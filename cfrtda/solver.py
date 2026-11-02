from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .game import DTYPE, Evaluation, TensorGame

UPDATES = ("rm", "rm+", "softplus", "hedge", "reg")
AVERAGING = ("uniform", "linear")
_LOG_SOFTPLUS_CUTOFF = -30.0


@dataclass(frozen=True)
class CFRConfig:

    update: str = "rm"
    averaging: str = "uniform"
    beta: float = 1.0
    eta: float = 0.1
    tau: float = 0.03

    def __post_init__(self):
        if self.update not in UPDATES:
            raise ValueError(f"update must be one of {UPDATES}, got {self.update!r}")
        if self.averaging not in AVERAGING:
            raise ValueError(f"averaging must be one of {AVERAGING}, got {self.averaging!r}")
        if not (self.beta > 0 and self.eta > 0 and self.tau > 0):
            raise ValueError("beta, eta and tau must be positive")

    @property
    def reg_step(self) -> float:
        return self.eta * self.tau / (1.0 + self.eta * self.tau)

    def averaging_weight(self, t: int) -> float:
        return 1.0 if self.averaging == "uniform" else float(t)


class CFRState(NamedTuple):
    regrets: torch.Tensor
    strategy_sum: torch.Tensor


def initial_state(game: TensorGame) -> CFRState:
    z = torch.zeros((game.num_infosets, game.num_actions), dtype=DTYPE)
    return CFRState(z, z.clone())


def _log_softplus(x: torch.Tensor) -> torch.Tensor:
    safe = torch.where(x > _LOG_SOFTPLUS_CUTOFF, x, torch.zeros_like(x))
    return torch.where(x > _LOG_SOFTPLUS_CUTOFF, torch.log(F.softplus(safe)), x)


def current_strategy(game: TensorGame, cfg: CFRConfig, regrets: torch.Tensor) -> torch.Tensor:
    legal = game.legal_mask
    if cfg.update in ("rm", "rm+"):
        pos = torch.where(legal, regrets.clamp(min=0.0), 0.0)
        total = pos.sum(1, keepdim=True)
        uniform = legal.to(regrets.dtype) / legal.sum(1, keepdim=True)
        return torch.where(total > 0, pos / torch.where(total > 0, total, 1.0), uniform)
    if cfg.update == "softplus":
        logits = _log_softplus(cfg.beta * regrets)
    elif cfg.update == "reg":
        logits = regrets
    else:
        logits = cfg.eta * regrets
    return torch.softmax(logits.masked_fill(~legal, float("-inf")), dim=1)


def average_strategy(game: TensorGame, strategy_sum: torch.Tensor) -> torch.Tensor:
    legal = game.legal_mask
    s = torch.where(legal, strategy_sum, 0.0)
    total = s.sum(1, keepdim=True)
    uniform = legal.to(s.dtype) / legal.sum(1, keepdim=True)
    return torch.where(total > 0, s / torch.where(total > 0, total, 1.0), uniform)


def regret_update(
    game: TensorGame,
    cfg: CFRConfig,
    R_prev: torch.Tensor,
    sigma: torch.Tensor,
    deal_weight_t: torch.Tensor,
    e: Evaluation | None = None,
) -> torch.Tensor:
    if cfg.update == "reg":
        from .balance import soft_q_values

        Q, visited = soft_q_values(game, sigma, deal_weight_t, cfg.tau, return_visited=True)
        b = cfg.reg_step
        moved = (1.0 - b) * R_prev + b * Q / cfg.tau
        return torch.where(visited.unsqueeze(1) & game.legal_mask, moved, R_prev)
    e = game.evaluate(sigma) if e is None else e
    R = R_prev + (deal_weight_t.view(-1, 1, 1) * e.regrets).sum(0)
    return R.clamp(min=0.0) if cfg.update == "rm+" else R


def average_increment(
    game: TensorGame, cfg: CFRConfig, sigma: torch.Tensor, deal_weight_t: torch.Tensor, t: int, e: Evaluation | None = None
) -> torch.Tensor:
    e = game.evaluate(sigma) if e is None else e
    return cfg.averaging_weight(t) * (deal_weight_t.view(-1, 1, 1) * e.avg_contrib).sum(0)


def step(game: TensorGame, cfg: CFRConfig, state: CFRState, deal_weight_t: torch.Tensor, t: int) -> CFRState:
    sigma = current_strategy(game, cfg, state.regrets)
    e = game.evaluate(sigma)
    regrets = regret_update(game, cfg, state.regrets, sigma, deal_weight_t, e)
    strategy_sum = state.strategy_sum + average_increment(game, cfg, sigma, deal_weight_t, t, e)
    return CFRState(regrets, strategy_sum)


@dataclass(eq=False)
class CFRResult:
    config: CFRConfig
    average: torch.Tensor
    state: CFRState
    num_iterations: int
    history: list[dict] = field(default_factory=list)


def _record(game, cfg, state, t, eval_deal_weights) -> dict:
    from .reference import exploitability

    with torch.no_grad():
        avg = average_strategy(game, state.strategy_sum)
        row = {"t": t, "exploitability": float(exploitability(game, avg))}
        if eval_deal_weights is not None:
            row["exploitability_empirical"] = float(exploitability(game, avg, eval_deal_weights))
    return row


def solve(
    game: TensorGame,
    cfg: CFRConfig,
    deal_weights: torch.Tensor,
    *,
    state: CFRState | None = None,
    start_iteration: int = 1,
    record_every: int | None = None,
    eval_deal_weights: torch.Tensor | None = None,
    checkpoint_segments: int | None = None,
) -> CFRResult:
    T = deal_weights.shape[0]
    if deal_weights.shape != (T, game.num_deals):
        raise ValueError(f"deal_weights must be [T, {game.num_deals}], got {tuple(deal_weights.shape)}")
    state = initial_state(game) if state is None else state
    history: list[dict] = []

    if checkpoint_segments:
        if record_every:
            raise ValueError("record_every is not supported together with checkpoint_segments")

        def run_segment(regrets, strategy_sum, w_seg, t0):
            s = CFRState(regrets, strategy_sum)
            for k in range(w_seg.shape[0]):
                s = step(game, cfg, s, w_seg[k], t0 + k)
            return s.regrets, s.strategy_sum

        for a in range(0, T, checkpoint_segments):
            b = min(a + checkpoint_segments, T)
            out = checkpoint(
                run_segment, state.regrets, state.strategy_sum, deal_weights[a:b], start_iteration + a, use_reentrant=False
            )
            state = CFRState(*out)
    else:
        for k in range(T):
            t = start_iteration + k
            state = step(game, cfg, state, deal_weights[k], t)
            if record_every and (t % record_every == 0 or k == T - 1):
                history.append(_record(game, cfg, state, t, eval_deal_weights))

    return CFRResult(
        config=cfg,
        average=average_strategy(game, state.strategy_sum),
        state=state,
        num_iterations=start_iteration + T - 1,
        history=history,
    )


