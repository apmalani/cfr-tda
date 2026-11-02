from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .balance import BalanceExact, BalanceInfluence
from .data import Hands
from .influence import target_functions, unrolled_influence
from .kuhn import Kuhn
from .removal import DATA_DIR, DATASETS, Experiment
from .solver import CFRConfig, current_strategy, solve

TARGETS = ("alpha", "Qb_call", "Jp_bet", "game_value")
DEFAULT_COMBOS = (("fixed_balanced", 0.1), ("fixed_seed0", 0.1), ("fixed_balanced", 0.03), ("fixed_seed0", 0.03))


def optimal_step(J: torch.Tensor) -> tuple[float, float]:
    mu = np.linalg.eigvals(J.numpy())
    if mu.real.min() <= 0:
        return math.nan, math.inf
    betas = np.geomspace(1e-5, 2.0, 20_000)
    rho = np.array([np.abs(1 - b * mu).max() for b in betas])
    i = int(rho.argmin())
    return float(betas[i]), float(rho[i])


def eta_for_step(b: float, tau: float) -> float:
    return b / (tau * (1.0 - b))


def run(dataset: str, tau: float, *, tol: float = 1e-13, points: int = 7) -> dict:
    torch.set_num_threads(1)
    file, _ = DATASETS[dataset]
    kuhn = Kuhn()
    hands = Hands.load(DATA_DIR / file)
    b = BalanceInfluence(kuhn, hands, tau=tau)
    kk = [b.targets.index(t) for t in TARGETS]
    B_ref = b.per_deal_add[kk]
    mu = np.linalg.eigvals(b.J.numpy())
    step, rho = optimal_step(b.J)
    cfg = CFRConfig("reg", tau=tau, eta=eta_for_step(step, tau))
    T_need = int(math.ceil(math.log(tol) / math.log(rho)))
    Ts = sorted({int(round(x)) for x in np.geomspace(100, 1.3 * T_need, points)})

    def rel(X):
        return float((X - B_ref).abs().max() / B_ref.abs().max())

    rows = []
    for T in Ts:
        ex = Experiment(kuhn, hands, cfg, T)
        last = current_strategy(kuhn, cfg, ex.states_R[-1])
        inf_last = unrolled_influence(ex, "A2", TARGETS, output="last")
        inf_avg = unrolled_influence(ex, "A2", TARGETS, output="average")
        rows.append({
            "T": T,
            "strategy_err_last": float((last - b.sigma).abs().max()),
            "strategy_err_average": float((ex.base_average - b.sigma).abs().max()),
            "influence_err_last": rel(inf_last.per_deal_add),
            "influence_err_average": rel(inf_avg.per_deal_add),
            "predicted_rate": rho**T,
        })

    T = Ts[-1]
    ex = Experiment(kuhn, hands, cfg, T)
    last = current_strategy(kuhn, cfg, ex.states_R[-1]).detach()
    inf_last = unrolled_influence(ex, "A2", TARGETS, output="last")
    bx = BalanceExact(kuhn, hands, tau=tau)
    fns = target_functions(kuhn, b.q)
    loo = []
    for d, lab in enumerate(hands.deal_labels):
        ids = torch.nonzero(hands.deal == d).flatten().tolist()
        if not ids:
            continue
        i = ids[0]
        sig = current_strategy(kuhn, cfg, solve(kuhn, cfg, ex.deal_weights(hands.without([i]))).state.regrets).detach()
        for t in TARGETS:
            loo.append({
                "deal": lab,
                "target": t,
                "actual": float(fns[t](sig) - fns[t](last)),
                "A2_last": inf_last.predict_removal([i], t),
                "B": b.predict_removal([i], t),
                "Bx": bx.predict_removal([i], t),
            })
    return {
        "dataset": dataset,
        "tau": tau,
        "step_b": step,
        "eta": cfg.eta,
        "rho": rho,
        "min_re_mu": float(mu.real.min()),
        "max_abs_mu": float(np.abs(mu).max()),
        "T_needed": T_need,
        "rows": rows,
        "loo": loo,
    }


def _task(args):
    return run(*args)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/harness"))
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        results = list(pool.map(_task, DEFAULT_COMBOS))
    (a.out / "consistency.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
