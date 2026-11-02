from __future__ import annotations

import torch

from .game import DTYPE, TensorGame
from .kuhn import BET, Kuhn


def _deal_distribution(game: TensorGame, deal_weights: torch.Tensor | None) -> torch.Tensor:
    if deal_weights is None:
        return game.deal_prob
    w = torch.as_tensor(deal_weights, dtype=DTYPE)
    if w.shape != (game.num_deals,) or bool((w < 0).any()) or float(w.sum()) <= 0:
        raise ValueError("deal_weights must be non-negative with shape [num_deals] and positive sum")
    return w / w.sum()


def best_response(
    game: TensorGame, sigma: torch.Tensor, player: int, deal_weights: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    q = _deal_distribution(game, deal_weights).view(-1, 1, 1)
    mine = game.infoset_player == player
    legal = game.legal_mask
    onehot = torch.zeros((game.num_infosets, game.num_actions), dtype=DTYPE)
    work = sigma.detach().clone()
    for level in sorted(set(game.infoset_own_depth[mine].tolist()), reverse=True):
        rows = mine & (game.infoset_own_depth == level)
        cfv = (q * game.evaluate(work).cf_values).sum(0)
        best = cfv.masked_fill(~legal, float("-inf")).argmax(1)
        choice = torch.nn.functional.one_hot(best, game.num_actions).to(DTYPE)
        onehot = torch.where(rows.unsqueeze(1), choice, onehot)
        work = torch.where(rows.unsqueeze(1), choice, work)
    strategy = torch.where(mine.unsqueeze(1), onehot, sigma)
    value = game.expected_value(strategy, q.flatten())[player]
    return strategy, value


def best_response_values(
    game: TensorGame, sigma: torch.Tensor, deal_weights: torch.Tensor | None = None
) -> torch.Tensor:
    return torch.stack([best_response(game, sigma, p, deal_weights)[1] for p in range(game.num_players)])


def nash_conv(game: TensorGame, sigma: torch.Tensor, deal_weights: torch.Tensor | None = None) -> torch.Tensor:
    q = _deal_distribution(game, deal_weights)
    return best_response_values(game, sigma, q).sum() - game.expected_value(sigma, q).sum()


def exploitability(game: TensorGame, sigma: torch.Tensor, deal_weights: torch.Tensor | None = None) -> torch.Tensor:
    return nash_conv(game, sigma, deal_weights) / game.num_players


KUHN_ACTION_TARGETS: dict[str, tuple[str, int]] = {
    "alpha": ("J", BET),
    "K_bet": ("K", BET),
    "Qpb_call": ("Qpb", BET),
    "Qb_call": ("Qb", BET),
    "Jp_bet": ("Jp", BET),
}


def kuhn_targets(
    kuhn: Kuhn, sigma: torch.Tensor, deal_weights: torch.Tensor | None = None, with_exploitability: bool = True
) -> dict[str, torch.Tensor]:
    out = {name: sigma[kuhn.infoset(lab), act] for name, (lab, act) in KUHN_ACTION_TARGETS.items()}
    q = _deal_distribution(kuhn, deal_weights)
    out["game_value"] = kuhn.expected_value(sigma, q)[0]
    if with_exploitability:
        out["exploitability"] = exploitability(kuhn, sigma, q)
    return out


