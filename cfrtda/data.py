from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .game import DTYPE, TensorGame

REGIMES = ("fixed", "streaming")
FORMAT = "cfrtda.hands/v1"
FIXED_NAMESPACE = 0
SHUFFLE_NAMESPACE = 2**31 - 1


def draw_uniform(seed: int, t: int, k: int) -> float:
    return float(np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, t, k]))).random())


def deal_from_uniform(u: float, deal_prob: torch.Tensor) -> int:
    cdf = np.cumsum(deal_prob.numpy())
    return int(min(np.searchsorted(cdf, u, side="right"), len(cdf) - 1))


def stratified_uniforms(seed: int, n: int) -> list[float]:
    perm = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, SHUFFLE_NAMESPACE, n]))).permutation(n)
    return [(int(j) + draw_uniform(seed, FIXED_NAMESPACE, int(j))) / n for j in perm]


@dataclass(frozen=True, eq=False)
class Hands:

    game_name: str
    regime: str
    seed: int
    deal: torch.Tensor
    iteration: torch.Tensor
    slot: torch.Tensor
    uniform: torch.Tensor
    batch_size: int
    num_iterations: int | None
    deal_labels: tuple[str, ...] | None = None
    stratified: bool = False

    def __post_init__(self):
        if self.regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}, got {self.regime!r}")

    def __len__(self) -> int:
        return int(self.deal.numel())

    @property
    def ids(self) -> torch.Tensor:
        return torch.arange(len(self))

    @property
    def num_deals(self) -> int:
        return len(self.deal_labels) if self.deal_labels is not None else int(self.deal.max()) + 1

    def ones(self) -> torch.Tensor:
        return torch.ones(len(self), dtype=DTYPE)

    def without(self, remove) -> torch.Tensor:
        w = self.ones()
        w[torch.as_tensor(remove, dtype=torch.long)] = 0.0
        return w

    def _check_w(self, w: torch.Tensor | None) -> torch.Tensor:
        if w is None:
            return self.ones()
        if w.shape != (len(self),):
            raise ValueError(f"hand weights must have shape [{len(self)}], got {tuple(w.shape)}")
        return w

    def deal_counts(self, w: torch.Tensor | None = None, num_deals: int | None = None) -> torch.Tensor:
        w = self._check_w(w)
        D = num_deals or self.num_deals
        return w.new_zeros(D).index_add(0, self.deal, w)

    def empirical_deal_probs(self, w: torch.Tensor | None = None, num_deals: int | None = None) -> torch.Tensor:
        c = self.deal_counts(w, num_deals)
        return c / c.sum()

    def deal_weights(
        self, w: torch.Tensor | None = None, num_iterations: int | None = None, num_deals: int | None = None
    ) -> torch.Tensor:
        w = self._check_w(w)
        D = num_deals or self.num_deals
        if self.regime == "fixed":
            if num_iterations is None:
                raise ValueError("fixed regime: pass num_iterations")
            return self.deal_counts(w, D).unsqueeze(0).expand(num_iterations, D)
        T = self.num_iterations if num_iterations is None else num_iterations
        if T != self.num_iterations:
            raise ValueError(f"streaming hands span {self.num_iterations} iterations, got {T}")
        flat = (self.iteration - 1) * D + self.deal
        return w.new_zeros(T * D).index_add(0, flat, w).view(T, D)

    def summary(self) -> str:
        labels = self.deal_labels or tuple(str(d) for d in range(self.num_deals))
        counts = self.deal_counts().long().tolist()
        per = ", ".join(f"{lab}:{c}" for lab, c in zip(labels, counts))
        span = f"T={self.num_iterations}, B={self.batch_size}" if self.regime == "streaming" else "replayed every iteration"
        kind = "stratified " if self.stratified else ""
        return f"{len(self)} {kind}{self.regime} hands of {self.game_name} (seed {self.seed}; {span}); per deal {per}"

    def to_json(self) -> dict:
        labels = self.deal_labels
        return {
            "format": FORMAT,
            "game": self.game_name,
            "regime": self.regime,
            "seed": self.seed,
            "stratified": self.stratified,
            "batch_size": self.batch_size,
            "num_iterations": self.num_iterations,
            "deal_labels": list(labels) if labels is not None else None,
            "hands": [
                {
                    "id": i,
                    "iteration": int(self.iteration[i]),
                    "slot": int(self.slot[i]),
                    "deal": int(self.deal[i]),
                    "deal_label": labels[int(self.deal[i])] if labels is not None else None,
                    "u": float(self.uniform[i]),
                }
                for i in range(len(self))
            ],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1))
        return path

    @classmethod
    def from_json(cls, data: dict) -> "Hands":
        if data.get("format") != FORMAT:
            raise ValueError(f"unknown hands format {data.get('format')!r}")
        hs = data["hands"]
        if [h["id"] for h in hs] != list(range(len(hs))):
            raise ValueError("hand ids must be 0..n-1 in order")
        return cls(
            game_name=data["game"],
            regime=data["regime"],
            seed=int(data["seed"]),
            deal=torch.tensor([h["deal"] for h in hs], dtype=torch.long),
            iteration=torch.tensor([h["iteration"] for h in hs], dtype=torch.long),
            slot=torch.tensor([h["slot"] for h in hs], dtype=torch.long),
            uniform=torch.tensor([h["u"] for h in hs], dtype=DTYPE),
            batch_size=int(data["batch_size"]),
            num_iterations=data["num_iterations"],
            deal_labels=tuple(data["deal_labels"]) if data["deal_labels"] is not None else None,
            stratified=bool(data.get("stratified", False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Hands":
        return cls.from_json(json.loads(Path(path).read_text()))


def make_hands(
    game: TensorGame,
    regime: str,
    *,
    seed: int = 0,
    num_hands: int | None = None,
    num_iterations: int | None = None,
    batch_size: int = 1,
    stratified: bool = False,
) -> Hands:
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    if stratified and regime != "fixed":
        raise ValueError("stratified sampling is only supported for the fixed regime")
    if regime == "fixed":
        if not num_hands or num_hands < 1:
            raise ValueError("fixed regime needs num_hands >= 1")
        slots = [(FIXED_NAMESPACE, k) for k in range(num_hands)]
        batch, T = num_hands, None
    else:
        if not num_iterations or num_iterations < 1 or batch_size < 1:
            raise ValueError("streaming regime needs num_iterations >= 1 and batch_size >= 1")
        slots = [(t, k) for t in range(1, num_iterations + 1) for k in range(batch_size)]
        batch, T = batch_size, num_iterations
    us = stratified_uniforms(seed, len(slots)) if stratified else [draw_uniform(seed, t, k) for t, k in slots]
    deals = [deal_from_uniform(u, game.deal_prob) for u in us]
    labels = getattr(game, "deal_labels", None)
    return Hands(
        game_name=game.game.get_type().short_name,
        regime=regime,
        seed=seed,
        deal=torch.tensor(deals, dtype=torch.long),
        iteration=torch.tensor([t for t, _ in slots], dtype=torch.long),
        slot=torch.tensor([k for _, k in slots], dtype=torch.long),
        uniform=torch.tensor(us, dtype=DTYPE),
        batch_size=batch,
        num_iterations=T,
        deal_labels=tuple(labels) if labels is not None else None,
        stratified=stratified,
    )


def main(argv=None) -> None:
    import argparse

    from .kuhn import Kuhn

    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=REGIMES, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-hands", type=int)
    ap.add_argument("--num-iterations", type=int)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--stratified", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    hands = make_hands(
        Kuhn(),
        a.regime,
        seed=a.seed,
        num_hands=a.num_hands,
        num_iterations=a.num_iterations,
        batch_size=a.batch_size,
        stratified=a.stratified,
    )
    if a.out:
        out = a.out
    elif a.stratified:
        out = Path("data") / f"kuhn_{a.regime}_n{len(hands)}_balanced{'' if a.seed == 0 else f'_seed{a.seed}'}.json"
    else:
        out = Path("data") / f"kuhn_{a.regime}_n{len(hands)}_seed{a.seed}.json"
    hands.save(out)


if __name__ == "__main__":
    main()
