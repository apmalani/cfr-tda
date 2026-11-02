from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .balance import BalanceInfluence
from .consistency import eta_for_step, optimal_step
from .data import Hands
from .influence import target_functions, unrolled_influence
from .kuhn import Kuhn
from .removal import DATA_DIR, DATASETS, Experiment
from .solver import CFRConfig, current_strategy, solve

PANELS = ("rm+", "hedge", "reg")
TS = (1000, 3000, 10000)
REG_TAU = 0.1


def _config(name: str, b: BalanceInfluence | None) -> CFRConfig:
    if name == "rm+":
        return CFRConfig("rm+")
    if name == "hedge":
        return CFRConfig("hedge", eta=0.5)
    step, _ = optimal_step(b.J)
    return CFRConfig("reg", tau=REG_TAU, eta=eta_for_step(step, REG_TAU))


def sweep(panel: str, dataset: str = "fixed_seed0", deal: str = "KQ", target: str = "Qb_call", points: int = 41) -> dict:
    torch.set_num_threads(1)
    kuhn = Kuhn()
    hands = Hands.load(DATA_DIR / DATASETS[dataset][0])
    d = kuhn.deal(deal)
    i = int(torch.nonzero(hands.deal == d).flatten()[0])
    tau_b = REG_TAU if panel == "reg" else 0.03
    b = BalanceInfluence(kuhn, hands, tau=tau_b)
    cfg = _config(panel, b)
    last = panel == "reg"
    s = np.linspace(0.0, 1.0, points)
    out = {"panel": panel, "dataset": dataset, "deal": deal, "hand": i, "target": target, "s": s.tolist(),
           "B_slope": float(b.per_deal_add[b.targets.index(target), d]), "B_tau": tau_b, "curves": []}
    for T in TS:
        ex = Experiment(kuhn, hands, cfg, T)
        inf = unrolled_influence(ex, "A2", (target,), output="last" if last else "average")
        f = []
        for x in s:
            w = hands.ones()
            w[i] -= x
            if last:
                sig = current_strategy(kuhn, cfg, solve(kuhn, cfg, ex.deal_weights(w)).state.regrets)
                f.append(float(target_functions(kuhn, ex.train_q)[target](sig)))
            else:
                f.append(ex.measure(ex.run(w))[target])
        f = np.asarray(f) - f[0]
        df = np.diff(f)
        out["curves"].append({
            "T": T,
            "delta": f.tolist(),
            "A2_slope": float(inf.per_hand[0, i]),
            "slope_sign_changes": int(np.sum(np.sign(df[1:]) != np.sign(df[:-1]))),
            "range": float(f.max() - f.min()),
        })
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/harness"))
    a = ap.parse_args(argv)
    with ProcessPoolExecutor(max_workers=len(PANELS)) as pool:
        results = list(pool.map(sweep, PANELS))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "phase.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
