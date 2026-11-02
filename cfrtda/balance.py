from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla
import torch

from .data import Hands
from .game import DTYPE, TensorGame


class LogitMap:

    def __init__(self, game: TensorGame):
        legal = game.legal_mask
        ref = legal.to(torch.int64).argmax(1)
        free = legal.clone()
        free[torch.arange(game.num_infosets), ref] = False
        self.game = game
        self.ref = ref
        self.free_index = torch.nonzero(free)
        self.size = int(self.free_index.shape[0])

    def theta(self, z: torch.Tensor) -> torch.Tensor:
        g = self.game
        th = torch.zeros((g.num_infosets, g.num_actions), dtype=z.dtype)
        th = th.index_put((self.free_index[:, 0], self.free_index[:, 1]), z)
        return th.masked_fill(~g.legal_mask, float("-inf"))

    def sigma(self, z: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.theta(z), dim=1)


def soft_q_values(
    game: TensorGame, sigma: torch.Tensor, q: torch.Tensor, tau: float, return_visited: bool = False
):
    P = game.num_players
    edge_prob = game.edge_probabilities(sigma)
    reach = game.reach_probabilities(edge_prob)
    dn = game.decision_nodes
    p_dn = game.node_player[dn]
    I_dn = game.node_infoset[dn]

    ent = -(torch.where(game.legal_mask, sigma * torch.log(sigma.clamp(min=1e-300)), 0.0)).sum(1)
    bonus = torch.zeros((game.num_nodes + 1, P), dtype=sigma.dtype).index_put((dn, p_dn), tau * ent[I_dn])
    ext_p = torch.cat([edge_prob, edge_prob.new_zeros(1)])
    v = torch.cat([game.node_returns, game.node_returns.new_zeros(1, P)]).to(sigma.dtype)
    for L in reversed(range(game.max_depth + 1)):
        idx = game.internal_by_depth[L]
        if idx.numel():
            ch = game.children[idx]
            v = v.index_put((idx,), (ext_p[ch].unsqueeze(-1) * v[ch]).sum(1) + bonus[idx])

    A = game.num_actions
    child = game.children[dn, :A]
    qa = v[child, p_dn.unsqueeze(1).expand(-1, A)]
    col = torch.arange(P + 1)
    cf_reach = torch.where(col.unsqueeze(0) == p_dn.unsqueeze(1), 1.0, reach[dn]).prod(1)
    wgt = q.to(sigma.dtype)[game.node_deal[dn]] * cf_reach
    num = torch.zeros((game.num_infosets, A), dtype=sigma.dtype).index_put((I_dn,), wgt.unsqueeze(1) * qa, accumulate=True)
    den = torch.zeros(game.num_infosets, dtype=sigma.dtype).index_put((I_dn,), wgt, accumulate=True)
    Q = num / den.clamp(min=1e-300).unsqueeze(1)
    return (Q, den > 0) if return_visited else Q


def balance_residual(lm: LogitMap, z: torch.Tensor, q: torch.Tensor, tau: float) -> torch.Tensor:
    Q = soft_q_values(lm.game, lm.sigma(z), q, tau)
    I, a = lm.free_index[:, 0], lm.free_index[:, 1]
    return z - (Q[I, a] - Q[I, lm.ref[I]]) / tau


def solve_equilibrium(
    lm: LogitMap,
    q: torch.Tensor,
    tau: float,
    z0: torch.Tensor | None = None,
    *,
    tol: float = 1e-12,
    max_newton: int = 100,
    homotopy_steps: int = 25,
) -> torch.Tensor:
    if z0 is None:
        taus = list(np.geomspace(max(1.0, tau), tau, homotopy_steps)) if tau < 1.0 else [tau]
        z = torch.zeros(lm.size, dtype=DTYPE)
    else:
        taus, z = [tau], z0.detach().clone()
    for t in taus:
        z = _newton(lm, q, float(t), z, tol, max_newton)
    return z


def _newton(lm, q, tau, z, tol, max_iter):
    F = lambda x: balance_residual(lm, x, q, tau)
    f = F(z)
    for _ in range(max_iter):
        if float(f.abs().max()) < tol:
            return z
        J = torch.autograd.functional.jacobian(F, z)
        dz = torch.linalg.solve(J, -f)
        step, base = 1.0, float(f.norm())
        while step > 1e-6:
            z_new = z + step * dz
            f_new = F(z_new)
            if bool(torch.isfinite(f_new).all()) and float(f_new.norm()) < (1 - 1e-4 * step) * base:
                break
            step *= 0.5
        z, f = z_new, f_new
    if float(f.abs().max()) > 1e-8:
        raise RuntimeError(f"Newton did not converge at tau={tau}: |F|={float(f.abs().max()):.2e}")
    return z


@dataclass(eq=False)
class BalanceInfluence:

    game: TensorGame
    hands: Hands
    tau: float = 0.03

    def __post_init__(self):
        from .influence import target_functions

        g = self.game
        self.lm = LogitMap(g)
        self.counts = self.hands.deal_counts(None, g.num_deals)
        self.N = float(self.counts.sum())
        self.q = self.counts / self.N
        self.z = solve_equilibrium(self.lm, self.q, self.tau)
        self.sigma = self.lm.sigma(self.z).detach()
        train_q = self.q if self.hands.regime == "fixed" else None
        self.fns = target_functions(g, train_q)
        self.targets = tuple(self.fns)

        self.J = torch.autograd.functional.jacobian(lambda z: balance_residual(self.lm, z, self.q, self.tau), self.z)
        self.Fq = torch.autograd.functional.jacobian(lambda q: balance_residual(self.lm, self.z, q, self.tau), self.q)
        grads = []
        for fn in self.fns.values():
            z = self.z.clone().requires_grad_(True)
            (gz,) = torch.autograd.grad(fn(self.lm.sigma(z)), z)
            grads.append(gz)
        self.grad_z = torch.stack(grads)
        lam = torch.linalg.solve(self.J.T, self.grad_z.T)
        self.dfdq = -(lam.T @ self.Fq)
        eye = torch.eye(g.num_deals, dtype=DTYPE)
        self.per_deal_add = self.dfdq @ ((eye - self.q.unsqueeze(0)) / self.N).T
        self.per_hand = self.per_deal_add[:, self.hands.deal]
        self.base_targets = {k: float(fn(self.sigma)) for k, fn in self.fns.items()}

    def predict_removal(self, ids, target: str) -> float:
        k = self.targets.index(target)
        return -float(self.per_hand[k, torch.as_tensor(list(ids), dtype=torch.long)].sum()) if len(ids) else 0.0

    def predict_add(self, deal: int, count: float, target: str) -> float:
        return float(count * self.per_deal_add[self.targets.index(target), deal])

    def resolve(self, counts: torch.Tensor) -> torch.Tensor:
        q_new = counts / counts.sum()
        try:
            z = solve_equilibrium(self.lm, q_new, self.tau, z0=self.z)
        except RuntimeError:
            try:
                z, s, ds = self.z, 0.0, 0.1
                while s < 1.0:
                    s_try = min(1.0, s + ds)
                    try:
                        z = solve_equilibrium(self.lm, (1 - s_try) * self.q + s_try * q_new, self.tau, z0=z)
                        s, ds = s_try, min(2 * ds, 0.5)
                    except RuntimeError:
                        ds /= 2
                        if ds < 1e-3:
                            raise
            except RuntimeError:
                z = solve_equilibrium(self.lm, q_new, self.tau, homotopy_steps=60)
        return self.lm.sigma(z).detach()


class BalanceExact(BalanceInfluence):

    def _delta(self, counts: torch.Tensor, target: str) -> float:
        if not hasattr(self, "_cache"):
            self._cache = {}
        key = tuple(counts.tolist())
        if key not in self._cache:
            self._cache[key] = self.resolve(counts)
        return float(self.fns[target](self._cache[key])) - self.base_targets[target]

    def predict_removal(self, ids, target: str) -> float:
        c = self.counts.clone()
        for i in ids:
            c[int(self.hands.deal[i])] -= 1
        return self._delta(c, target)

    def predict_add(self, deal: int, count: float, target: str) -> float:
        c = self.counts.clone()
        c[deal] += count
        return self._delta(c, target)


def linear_solver_study(J: torch.Tensor, g: torch.Tensor) -> dict[str, float]:
    A, b = J.T.numpy(), g.numpy()
    exact = np.linalg.solve(A, b)

    def err(x):
        return float(np.linalg.norm(x - exact) / np.linalg.norm(exact))

    x_g, _ = spla.gmres(A, b, rtol=1e-12, atol=0.0, restart=A.shape[0], maxiter=1000)
    x_c, _ = spla.cg(A, b, rtol=1e-12, atol=0.0, maxiter=10 * A.shape[0])
    return {
        "asymmetry": float(np.linalg.norm((A - A.T) / 2) / np.linalg.norm(A)),
        "complex_eigs": int(np.sum(np.abs(np.linalg.eigvals(A).imag) > 1e-9)),
        "gmres_err": err(x_g),
        "cg_err": err(x_c),
    }


