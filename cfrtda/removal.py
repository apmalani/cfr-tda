from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .data import Hands, make_hands
from .game import DTYPE
from .kuhn import Kuhn
from .reference import exploitability, kuhn_targets
from .solver import CFRConfig, CFRState, average_strategy, initial_state, solve, step

TARGETS = (
    "alpha",
    "K_bet",
    "Qpb_call",
    "Qb_call",
    "Jp_bet",
    "game_value",
    "exploitability",
    "exploitability_train",
)

CONFIGS: dict[str, CFRConfig] = {
    "rm": CFRConfig("rm"),
    "rm+": CFRConfig("rm+"),
    "rm+_linear": CFRConfig("rm+", "linear"),
    "softplus": CFRConfig("softplus", beta=1.0),
    "hedge": CFRConfig("hedge", eta=0.5),
}

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASETS: dict[str, tuple[str, int | None]] = {
    "fixed_balanced": ("kuhn_fixed_n30_balanced.json", 3000),
    "fixed_seed0": ("kuhn_fixed_n30_seed0.json", 3000),
    "streaming": ("kuhn_streaming_n3000_seed0.json", None),
}


class Experiment:

    def __init__(self, game: Kuhn, hands: Hands, cfg: CFRConfig, num_iterations: int | None = None):
        self.game, self.hands, self.cfg = game, hands, cfg
        self.T = hands.num_iterations if hands.regime == "streaming" else num_iterations
        if self.T is None:
            raise ValueError("fixed regime needs num_iterations")
        self.W0 = self.deal_weights()
        self.train_q = hands.empirical_deal_probs(None, game.num_deals) if hands.regime == "fixed" else None

        state = initial_state(game)
        R, S = [state.regrets], [state.strategy_sum]
        with torch.no_grad():
            for k in range(self.T):
                state = step(game, cfg, state, self.W0[k], k + 1)
                R.append(state.regrets)
                S.append(state.strategy_sum)
        self.states_R = torch.stack(R)
        self.states_S = torch.stack(S)
        self.base_average = average_strategy(game, state.strategy_sum)
        self.base_targets = self.measure(self.base_average)

    def deal_weights(self, w: torch.Tensor | None = None, extra_counts: torch.Tensor | None = None) -> torch.Tensor:
        W = self.hands.deal_weights(w, self.T, self.game.num_deals)
        if extra_counts is not None:
            if self.hands.regime != "fixed":
                raise ValueError("extra_counts is only defined for the fixed regime")
            W = W + extra_counts.to(DTYPE).unsqueeze(0)
        return W / self.hands.batch_size

    @torch.no_grad()
    def run(self, w: torch.Tensor | None = None, extra_counts: torch.Tensor | None = None) -> torch.Tensor:
        return self.run_W(self.deal_weights(w, extra_counts))

    @torch.no_grad()
    def run_W(self, W: torch.Tensor) -> torch.Tensor:
        changed = (W != self.W0).any(1)
        if not bool(changed.any()):
            return self.base_average.clone()
        t0 = int(changed.nonzero()[0])
        start = CFRState(self.states_R[t0], self.states_S[t0])
        return solve(self.game, self.cfg, W[t0:], state=start, start_iteration=t0 + 1).average

    @torch.no_grad()
    def measure(self, average: torch.Tensor) -> dict[str, float]:
        t = {k: float(v) for k, v in kuhn_targets(self.game, average).items()}
        t["exploitability_train"] = (
            float(exploitability(self.game, average, self.train_q)) if self.train_q is not None else math.nan
        )
        return {k: t[k] for k in TARGETS}


@dataclass(eq=False)
class Results:
    dataset: str
    config_name: str
    config: dict
    regime: str
    num_iterations: int
    hands_summary: str
    base_average: torch.Tensor
    base_targets: dict[str, float]
    records: list[dict] = field(default_factory=list)
    averages: list[torch.Tensor] = field(default_factory=list)

    def add(self, average: torch.Tensor, targets: dict[str, float], **meta) -> None:
        meta.setdefault("removed", [])
        meta.setdefault("size", len(meta["removed"]))
        self.records.append({**meta, "targets": targets})
        self.averages.append(average)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.dataset}__{self.config_name}.pt"
        payload = {k: v for k, v in asdict(self).items() if k not in ("averages", "base_average")}
        payload["base_average"] = self.base_average
        payload["averages"] = torch.stack(self.averages) if self.averages else torch.empty(0)
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Results":
        p = torch.load(path, weights_only=True)
        averages = list(p.pop("averages")) if p["averages"].numel() else []
        return cls(**p, averages=averages)

def streaming_sample(hands: Hands, size: int, seed: int = 0) -> list[int]:
    n = len(hands)
    size = min(size, n)
    edges = np.linspace(0, n, size + 1).round().astype(int)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 15, n, size])))
    return [int(rng.integers(a, b)) for a, b in zip(edges[:-1], edges[1:])]


def run_suite(
    game: Kuhn,
    hands: Hands,
    cfg: CFRConfig,
    *,
    dataset: str,
    config_name: str,
    num_iterations: int | None = None,
    streaming_loo: int = 200,
    group_sizes: tuple[int, ...] = (2, 5, 10, 20),
    groups_per_size: int = 20,
    windows: int = 10,
    add_deal: str = "JK",
    add_counts: tuple[int, ...] = (1, 5),
    noise_runs: int = 5,
    noise_eps: float = 1e-12,
    resample_seeds: tuple[int, ...] = tuple(range(1, 11)),
    seed: int = 0,
) -> Results:
    ex = Experiment(game, hands, cfg, num_iterations)
    res = Results(
        dataset=dataset,
        config_name=config_name,
        config=asdict(cfg),
        regime=hands.regime,
        num_iterations=ex.T,
        hands_summary=hands.summary(),
        base_average=ex.base_average,
        base_targets=ex.base_targets,
    )
    labels = hands.deal_labels or tuple(str(d) for d in range(game.num_deals))
    fixed = hands.regime == "fixed"

    def record(kind, average, **meta):
        res.add(average, ex.measure(average), kind=kind, **meta)

    loo_ids = list(range(len(hands))) if fixed else streaming_sample(hands, streaming_loo, seed)
    for i in loo_ids:
        record(
            "loo",
            ex.run(hands.without([i])),
            label=f"hand {i}",
            removed=[i],
            deal=labels[int(hands.deal[i])],
            iteration=int(hands.iteration[i]),
        )

    if fixed:
        for d, lab in enumerate(labels):
            ids = torch.nonzero(hands.deal == d).flatten().tolist()
            if ids:
                record("group_deal", ex.run(hands.without(ids)), label=f"all {lab}", removed=ids, deal=lab)
    else:
        edges = np.linspace(1, ex.T + 1, windows + 1).round().astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            ids = torch.nonzero((hands.iteration >= a) & (hands.iteration < b)).flatten().tolist()
            record("group_window", ex.run(hands.without(ids)), label=f"iterations {a}-{b - 1}", removed=ids)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 16, len(hands)])))
    for k in group_sizes:
        if k > len(loo_ids):
            continue
        for g in range(groups_per_size):
            ids = sorted(int(x) for x in rng.choice(loo_ids, size=k, replace=False))
            record("group_random", ex.run(hands.without(ids)), label=f"random {k} #{g}", removed=ids)

    if fixed and add_deal in labels:
        d = labels.index(add_deal)
        for c in add_counts:
            extra = torch.zeros(game.num_deals, dtype=DTYPE)
            extra[d] = c
            record("add_deal", ex.run(extra_counts=extra), label=f"+{c} {add_deal}", deal=add_deal, added=c)

    g = torch.Generator().manual_seed(seed + 1_000)
    for r in range(noise_runs):
        u = torch.rand(len(hands), generator=g, dtype=DTYPE) * 2 - 1
        record("noise_fp", ex.run(hands.ones() + noise_eps * u), label=f"fp noise #{r} (eps={noise_eps:g})")

    for s in resample_seeds:
        if fixed:
            other = make_hands(game, "fixed", num_hands=len(hands), seed=s)
        else:
            other = make_hands(game, "streaming", num_iterations=ex.T, batch_size=hands.batch_size, seed=s)
        W = other.deal_weights(None, ex.T, game.num_deals) / other.batch_size
        avg = solve(game, cfg, W).average
        record("resample", avg, label=f"redrawn set (seed {s})", seed=s)

    return res


def _task(args) -> Path:
    dataset, config_name, out_dir, kwargs = args
    torch.set_num_threads(1)
    file, T = DATASETS[dataset]
    kuhn = Kuhn()
    hands = Hands.load(DATA_DIR / file)
    res = run_suite(
        kuhn, hands, CONFIGS[config_name], dataset=dataset, config_name=config_name, num_iterations=T, **kwargs
    )
    return res.save(out_dir)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--streaming-loo", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("results/ground_truth"))
    a = ap.parse_args(argv)
    tasks = [(d, c, a.out, {"streaming_loo": a.streaming_loo}) for d in a.datasets.split(",") for c in a.configs.split(",")]
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(_task, tasks))


if __name__ == "__main__":
    main()
