from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyspiel
import torch
from open_spiel.python import policy as os_policy

DTYPE = torch.float64

DECISION, CHANCE, TERMINAL = 0, 1, 2


@dataclass(frozen=True, eq=False)
class Evaluation:

    edge_prob: torch.Tensor
    reach: torch.Tensor
    values: torch.Tensor
    root_values: torch.Tensor
    regrets: torch.Tensor
    avg_contrib: torch.Tensor
    cf_values: torch.Tensor


class TensorGame:

    def __init__(self, game: pyspiel.Game):
        gtype = game.get_type()
        if game.num_players() != 2:
            raise ValueError("TensorGame supports two-player games only")
        if gtype.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
            raise ValueError("TensorGame supports sequential games only")
        if gtype.utility != pyspiel.GameType.Utility.ZERO_SUM:
            raise ValueError("TensorGame supports zero-sum games only")

        self.game = game
        self.num_players = game.num_players()
        self.num_actions = game.num_distinct_actions()

        tabular = os_policy.TabularPolicy(game)
        self.infoset_keys: tuple[str, ...] = tuple(tabular.state_lookup)
        assert all(tabular.state_lookup[k] == i for i, k in enumerate(self.infoset_keys))
        self.infoset_index: dict[str, int] = dict(tabular.state_lookup)
        self.num_infosets = len(self.infoset_keys)
        self.legal_mask = torch.as_tensor(np.asarray(tabular.legal_actions_mask), dtype=torch.bool)
        self.infoset_player = torch.full((self.num_infosets,), -1, dtype=torch.long)

        self._build(game)

    def _build(self, game: pyspiel.Game) -> None:
        P = self.num_players
        deals: list[tuple[int, ...]] = []
        deal_probs: list[float] = []
        deal_root_states: list[pyspiel.State] = []

        def find_deals(state, outcomes, prob):
            if state.is_chance_node():
                for a, p in state.chance_outcomes():
                    find_deals(state.child(a), outcomes + (a,), prob * p)
            else:
                deals.append(outcomes)
                deal_probs.append(prob)
                deal_root_states.append(state)

        find_deals(game.new_initial_state(), (), 1.0)

        kind, player, infoset, depth, deal_of = [], [], [], [], []
        parent, action, owner, chance_p, returns, path = [], [], [], [], [], []
        child_lists: list[list[tuple[int, int]]] = []
        max_branch = self.num_actions

        def add_node(state, d, dep, par, act, own, cp, parent_path):
            nonlocal max_branch
            n = len(kind)
            deal_of.append(d)
            depth.append(dep)
            parent.append(par)
            action.append(act)
            owner.append(own)
            chance_p.append(cp)
            path.append(parent_path + ([n] if par >= 0 else []))
            child_lists.append([])
            if state.is_terminal():
                kind.append(TERMINAL)
                player.append(-1)
                infoset.append(-1)
                returns.append(list(state.returns()))
                return n
            returns.append([0.0] * P)
            if state.is_chance_node():
                kind.append(CHANCE)
                player.append(-1)
                infoset.append(-1)
                outs = state.chance_outcomes()
                max_branch = max(max_branch, len(outs))
                for b, (a, p) in enumerate(outs):
                    c = add_node(state.child(a), d, dep + 1, n, a, P, p, path[n])
                    child_lists[n].append((b, c))
                return n
            cur = state.current_player()
            key = state.information_state_string(cur)
            i = self.infoset_index[key]
            if self.infoset_player[i] < 0:
                self.infoset_player[i] = cur
            assert self.infoset_player[i] == cur
            legal = state.legal_actions()
            if self.legal_mask[i].nonzero().flatten().tolist() != list(legal):
                raise AssertionError(f"legal actions disagree with TabularPolicy at {key!r}")
            kind.append(DECISION)
            player.append(cur)
            infoset.append(i)
            for a in legal:
                c = add_node(state.child(a), d, dep + 1, n, a, cur, 1.0, path[n])
                child_lists[n].append((a, c))
            return n

        roots = [add_node(s, d, 0, -1, -1, -1, 1.0, []) for d, s in enumerate(deal_root_states)]

        N = len(kind)
        self.num_nodes = N
        self.num_deals = len(deals)
        self.deals: tuple[tuple[int, ...], ...] = tuple(deals)
        self.deal_prob = torch.tensor(deal_probs, dtype=DTYPE)
        self.deal_root = torch.tensor(roots, dtype=torch.long)

        self.node_kind = torch.tensor(kind, dtype=torch.long)
        self.node_player = torch.tensor(player, dtype=torch.long)
        self.node_infoset = torch.tensor(infoset, dtype=torch.long)
        self.node_depth = torch.tensor(depth, dtype=torch.long)
        self.node_deal = torch.tensor(deal_of, dtype=torch.long)
        self.node_returns = torch.tensor(returns, dtype=DTYPE)
        self.edge_parent = torch.tensor(parent, dtype=torch.long)
        self.edge_action = torch.tensor(action, dtype=torch.long)
        self.edge_owner = torch.tensor(owner, dtype=torch.long)
        self.edge_chance_prob = torch.tensor(chance_p, dtype=DTYPE)

        self.max_depth = max(depth)
        pe = torch.full((N, max(self.max_depth, 1)), N, dtype=torch.long)
        for n, pth in enumerate(path):
            if pth:
                pe[n, : len(pth)] = torch.tensor(pth)
        self.path_edges = pe

        ch = torch.full((N, max_branch), N, dtype=torch.long)
        for n, lst in enumerate(child_lists):
            for b, c in lst:
                ch[n, b] = c
        self.children = ch
        self.internal_by_depth = [
            torch.nonzero((self.node_depth == L) & (self.node_kind != TERMINAL)).flatten()
            for L in range(self.max_depth + 1)
        ]

        self.decision_nodes = torch.nonzero(self.node_kind == DECISION).flatten()
        dn = self.decision_nodes
        pairs = torch.stack([self.node_deal[dn], self.node_infoset[dn]], 1)
        if torch.unique(pairs, dim=0).shape[0] != pairs.shape[0]:
            raise ValueError("an infoset occurs more than once within a single deal")
        node_of = torch.full((self.num_deals, self.num_infosets), -1, dtype=torch.long)
        node_of[pairs[:, 0], pairs[:, 1]] = dn
        self.node_of = node_of

        self._edge_is_decision = (self.edge_owner >= 0) & (self.edge_owner < P)
        self._parent_infoset = torch.where(
            self._edge_is_decision, self.node_infoset[self.edge_parent.clamp(min=0)], 0
        )
        self._decision_action = torch.where(self._edge_is_decision, self.edge_action, 0)

        ext_owner = torch.cat([self.edge_owner, self.edge_owner.new_full((1,), -1)])
        own = (ext_owner[self.path_edges[dn]] == self.node_player[dn].unsqueeze(1)).sum(1)
        base = torch.full((self.num_infosets,), -1, dtype=torch.long)
        lo = base.scatter_reduce(0, self.node_infoset[dn], own, reduce="amin", include_self=False)
        hi = base.scatter_reduce(0, self.node_infoset[dn], own, reduce="amax", include_self=False)
        if not torch.equal(lo, hi):
            raise ValueError("game does not have perfect recall")
        self.infoset_own_depth = lo

    def edge_probabilities(self, sigma: torch.Tensor) -> torch.Tensor:
        p_dec = sigma[self._parent_infoset, self._decision_action]
        return torch.where(self._edge_is_decision, p_dec, self.edge_chance_prob.to(sigma.dtype))

    def reach_probabilities(self, edge_prob: torch.Tensor) -> torch.Tensor:
        ext_p = torch.cat([edge_prob, edge_prob.new_ones(1)])
        ext_o = torch.cat([self.edge_owner, self.edge_owner.new_full((1,), -1)])
        g = ext_p[self.path_edges]
        o = ext_o[self.path_edges]
        cols = [torch.where(o == q, g, 1.0).prod(1) for q in range(self.num_players + 1)]
        return torch.stack(cols, 1)

    def node_values(self, edge_prob: torch.Tensor) -> torch.Tensor:
        ext_p = torch.cat([edge_prob, edge_prob.new_zeros(1)])
        v = torch.cat([self.node_returns, self.node_returns.new_zeros(1, self.num_players)]).to(edge_prob.dtype)
        for L in reversed(range(self.max_depth + 1)):
            idx = self.internal_by_depth[L]
            if idx.numel() == 0:
                continue
            ch = self.children[idx]
            v = v.index_put((idx,), (ext_p[ch].unsqueeze(-1) * v[ch]).sum(1))
        return v[:-1]

    def evaluate(self, sigma: torch.Tensor) -> Evaluation:
        P, A = self.num_players, self.num_actions
        edge_prob = self.edge_probabilities(sigma)
        reach = self.reach_probabilities(edge_prob)
        values = self.node_values(edge_prob)

        h = self.decision_nodes
        p = self.node_player[h]
        I = self.node_infoset[h]
        d = self.node_deal[h]
        legal = self.legal_mask[I]

        child = self.children[h, :A]
        ext_v = torch.cat([values, values.new_zeros(1, P)])
        q = ext_v[child, p.unsqueeze(1).expand(-1, A)]
        v_h = values[h, p]
        r_h = reach[h]
        col = torch.arange(P + 1)
        cf_reach = torch.where(col.unsqueeze(0) == p.unsqueeze(1), 1.0, r_h).prod(1)
        own_reach = r_h.gather(1, p.unsqueeze(1)).squeeze(1)

        cfv_node = torch.where(legal, cf_reach.unsqueeze(1) * q, 0.0)
        reg_node = torch.where(legal, cf_reach.unsqueeze(1) * (q - v_h.unsqueeze(1)), 0.0)
        avg_node = own_reach.unsqueeze(1) * sigma[I]

        zeros = sigma.new_zeros((self.num_deals, self.num_infosets, A))
        return Evaluation(
            edge_prob=edge_prob,
            reach=reach,
            values=values,
            root_values=values[self.deal_root],
            regrets=zeros.index_put((d, I), reg_node, accumulate=True),
            avg_contrib=zeros.index_put((d, I), avg_node, accumulate=True),
            cf_values=zeros.index_put((d, I), cfv_node, accumulate=True),
        )

    def expected_value(self, sigma: torch.Tensor, deal_weights: torch.Tensor | None = None) -> torch.Tensor:
        w = self.deal_prob if deal_weights is None else deal_weights
        values = self.node_values(self.edge_probabilities(sigma))
        return (w.to(values.dtype).unsqueeze(1) * values[self.deal_root]).sum(0)
