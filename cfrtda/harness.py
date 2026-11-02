from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from .compare import evaluate_combo
from .data import Hands
from .kuhn import Kuhn
from .removal import CONFIGS, DATA_DIR, DATASETS, Experiment, Results

METHODS = ("A0", "A1", "A2", "B", "Bx")
FRACTIONS = {"fixed": (0.1, 0.25, 0.5), "streaming": (0.01, 0.05, 0.1, 0.25, 0.5)}
LDS_M = 100
B_TAU = 0.03


def lds_subsets(n: int, fraction: float, M: int, seed: int = 0) -> list[list[int]]:
    k = max(1, int(round(fraction * n)))
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 17, n, int(round(fraction * 1e6))])))
    return [sorted(int(x) for x in rng.choice(n, size=k, replace=False)) for _ in range(M)]


def lds_ground_truth(dataset: str, config_name: str, out_dir: Path, M: int = LDS_M) -> Path:
    path = out_dir / f"{dataset}__{config_name}.pt"
    if path.exists():
        return path
    torch.set_num_threads(1)
    file, T = DATASETS[dataset]
    kuhn = Kuhn()
    hands = Hands.load(DATA_DIR / file)
    ex = Experiment(kuhn, hands, CONFIGS[config_name], T)
    res = Results(
        dataset=dataset, config_name=config_name, config={}, regime=hands.regime, num_iterations=ex.T,
        hands_summary=hands.summary(), base_average=ex.base_average, base_targets=ex.base_targets,
    )
    for frac in FRACTIONS[hands.regime]:
        for j, ids in enumerate(lds_subsets(len(hands), frac, M)):
            avg = ex.run(hands.without(ids))
            res.add(avg, ex.measure(avg), kind="lds", label=f"remove {frac:g} #{j}", removed=ids, fraction=frac)
    out_dir.mkdir(parents=True, exist_ok=True)
    return res.save(out_dir)


def _sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def spearman(a, p) -> float:
    a, p = np.asarray(a, float), np.asarray(p, float)
    if len(a) < 3 or a.std() == 0 or p.std() == 0:
        return math.nan
    return float(stats.spearmanr(a, p).statistic)


def lds_table(rows, method: str, target: str) -> dict[float, float]:
    out = {}
    lds = _sel(rows, kind="lds", target=target)
    for frac in sorted({r["fraction"] for r in lds}):
        sub = [r for r in lds if r["fraction"] == frac]
        out[frac] = spearman([r["actual"] for r in sub], [r[f"pred_{method}"] for r in sub])
    return out


def _lds_task(args):
    return lds_ground_truth(*args)


def _eval_task(args):
    d, c, gt_dir, lds_dir = args
    torch.set_num_threads(1)
    rows = evaluate_combo(d, c, gt_dir, methods=METHODS, b_tau=B_TAU, extra=(lds_dir / f"{d}__{c}.pt",))
    return (d, c), rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", type=Path, default=Path("results/ground_truth"))
    ap.add_argument("--lds", type=Path, default=Path("results/lds"))
    ap.add_argument("--out", type=Path, default=Path("results/harness"))
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    combos = [(d, c) for d in DATASETS for c in CONFIGS]
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(_lds_task, [(d, c, a.lds) for d, c in combos]))
        all_rows = dict(pool.map(_eval_task, [(d, c, a.ground_truth, a.lds) for d, c in combos]))
    a.out.mkdir(parents=True, exist_ok=True)
    flat = [r for rows in all_rows.values() for r in rows]
    with open(a.out / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    lds = {f"{d}|{c}|{m}|{t}": {str(f): v for f, v in lds_table(all_rows[(d, c)], m, t).items()}
           for d, c in combos for m in METHODS for t in ("alpha", "Qb_call", "exploitability")}
    (a.out / "lds.json").write_text(json.dumps(lds, indent=1))


if __name__ == "__main__":
    main()
