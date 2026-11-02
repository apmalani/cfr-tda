from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .data import Hands
from .game import DTYPE, TensorGame
from .reference import exploitability, kuhn_targets
from .solver import CFRConfig, average_increment, average_strategy, current_strategy, regret_update

MODES = ("A0", "A1", "A2")
OUTPUTS = ("average", "last")


def step_parts(
    game: TensorGame, cfg: CFRConfig, R_prev: torch.Tensor, W_t: torch.Tensor, t: int, mode: str = "A2"
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    sigma = current_strategy(game, cfg, R_prev)
    s_reg = sigma if mode == "A2" else sigma.detach()
    s_avg = sigma.detach() if mode == "A0" else sigma
    e_avg = game.evaluate(s_avg)
    R = regret_update(game, cfg, R_prev, s_reg, W_t, e_avg if s_reg is s_avg else None)
    dS = average_increment(game, cfg, s_avg, W_t, t, e_avg)
    return R, dS


def target_functions(game, train_q: torch.Tensor | None = None) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    from .removal import TARGETS

    fns: dict[str, Callable] = {}
    for name in TARGETS:
        if name == "exploitability_train":
            if train_q is not None:
                fns[name] = lambda avg, q=train_q: exploitability(game, avg, q)
        elif name == "exploitability":
            fns[name] = lambda avg: exploitability(game, avg)
        else:
            fns[name] = lambda avg, n=name: kuhn_targets(game, avg, with_exploitability=False)[n]
    return fns


def strategy_sum_seeds(game: TensorGame, strategy_sum: torch.Tensor, fns: dict[str, Callable]) -> torch.Tensor:
    S = strategy_sum.detach().clone().requires_grad_(True)
    avg = average_strategy(game, S)
    seeds = []
    for fn in fns.values():
        (g,) = torch.autograd.grad(fn(avg), S, retain_graph=True)
        seeds.append(g)
    return torch.stack(seeds)


def last_iterate_seeds(game: TensorGame, cfg: CFRConfig, regrets: torch.Tensor, fns: dict[str, Callable]) -> torch.Tensor:
    R = regrets.detach().clone().requires_grad_(True)
    sigma = current_strategy(game, cfg, R)
    seeds = []
    for fn in fns.values():
        (g,) = torch.autograd.grad(fn(sigma), R, retain_graph=True, allow_unused=True)
        seeds.append(torch.zeros_like(R) if g is None else g)
    return torch.stack(seeds)


def adjoint(
    game: TensorGame,
    cfg: CFRConfig,
    W: torch.Tensor,
    states_R: torch.Tensor,
    seeds: torch.Tensor,
    mode: str = "A2",
    seeds_R: torch.Tensor | None = None,
) -> torch.Tensor:
    T = W.shape[0]
    K = seeds.shape[0]
    Rbar = torch.zeros((K,) + tuple(states_R.shape[1:]), dtype=DTYPE) if seeds_R is None else seeds_R.clone()
    Wbar = torch.empty((K, T, W.shape[1]), dtype=DTYPE)
    for k in reversed(range(T)):
        R_prev = states_R[k].detach().requires_grad_(True)
        W_t = W[k].detach().requires_grad_(True)
        with torch.enable_grad():
            R, dS = step_parts(game, cfg, R_prev, W_t, k + 1, mode)
            out = torch.cat([R.flatten(), dS.flatten()])
            cot = torch.cat([Rbar.flatten(1), seeds.flatten(1)], 1)
            gR, gW = torch.autograd.grad(
                out, (R_prev, W_t), grad_outputs=cot, is_grads_batched=True, allow_unused=True
            )
        Rbar = torch.zeros_like(Rbar) if gR is None else gR.detach()
        Wbar[:, k] = torch.zeros_like(Wbar[:, k]) if gW is None else gW.detach()
    return Wbar


@dataclass(eq=False)
class Influence:

    mode: str
    targets: tuple[str, ...]
    Wbar: torch.Tensor
    per_hand: torch.Tensor
    per_deal_add: torch.Tensor

    def target_index(self, name: str) -> int:
        return self.targets.index(name)

    def predict_removal(self, ids, target: str) -> float:
        k = self.target_index(target)
        return -float(self.per_hand[k, torch.as_tensor(list(ids), dtype=torch.long)].sum()) if len(ids) else 0.0

    def predict_add(self, deal: int, count: float, target: str) -> float:
        return float(count * self.per_deal_add[self.target_index(target), deal])


def hand_influence(hands: Hands, Wbar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B = hands.batch_size
    if hands.regime == "fixed":
        per_deal = Wbar.sum(1) / B
        return per_deal[:, hands.deal], per_deal
    per_hand = Wbar[:, hands.iteration - 1, hands.deal] / B
    return per_hand, torch.full((Wbar.shape[0], Wbar.shape[2]), float("nan"), dtype=DTYPE)


def unrolled_influence(
    experiment, mode: str = "A2", targets: tuple[str, ...] | None = None, output: str = "average"
) -> Influence:
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}")
    game, hands, cfg = experiment.game, experiment.hands, experiment.cfg
    fns = target_functions(game, experiment.train_q)
    if targets is not None:
        fns = {k: fns[k] for k in targets}
    if output == "average":
        seeds = strategy_sum_seeds(game, experiment.states_S[-1], fns)
        seeds_R = None
    else:
        seeds_R = last_iterate_seeds(game, cfg, experiment.states_R[-1], fns)
        seeds = torch.zeros_like(seeds_R)
    Wbar = adjoint(game, cfg, experiment.W0, experiment.states_R, seeds, mode, seeds_R)
    per_hand, per_deal = hand_influence(hands, Wbar)
    return Influence(mode=mode, targets=tuple(fns), Wbar=Wbar, per_hand=per_hand, per_deal_add=per_deal)
