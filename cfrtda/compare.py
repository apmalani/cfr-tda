from __future__ import annotations

import math
from pathlib import Path

import torch

from .balance import BalanceExact, BalanceInfluence
from .data import Hands
from .influence import MODES, unrolled_influence
from .kuhn import Kuhn
from .removal import CONFIGS, DATA_DIR, DATASETS, TARGETS, Experiment, Results

PRED_KINDS = ("loo", "group_deal", "group_random", "group_window", "add_deal", "lds")


def predict(method, rec: dict, target: str, deal_labels) -> float:
    if rec["kind"] == "add_deal":
        return method.predict_add(deal_labels.index(rec["deal"]), rec["added"], target)
    return method.predict_removal(rec["removed"], target)


def evaluate_combo(
    dataset: str,
    config_name: str,
    gt_dir: Path,
    *,
    methods: tuple[str, ...] = (*MODES, "B", "Bx"),
    b_tau: float = 0.03,
    extra: tuple[Path, ...] = (),
) -> list[dict]:
    file, T = DATASETS[dataset]
    kuhn = Kuhn()
    hands = Hands.load(DATA_DIR / file)
    res = Results.load(gt_dir / f"{dataset}__{config_name}.pt")
    records = list(res.records)
    for p in extra:
        other = Results.load(p)
        if repr(other.base_targets) != repr(res.base_targets):
            raise RuntimeError(f"{p} was computed from a different baseline")
        records += other.records
    ex = Experiment(kuhn, hands, CONFIGS[config_name], T)
    if not torch.equal(ex.base_average, res.base_average):
        raise RuntimeError("baseline does not reproduce the saved ground truth")
    impl = {}
    for m in methods:
        if m in MODES:
            impl[m] = unrolled_influence(ex, m)
        elif m in ("B", "Bx"):
            impl[m] = (BalanceInfluence if m == "B" else BalanceExact)(kuhn, hands, tau=b_tau)
        else:
            raise ValueError(f"unknown method {m!r}")

    rows = []
    for rec in records:
        if rec["kind"] not in PRED_KINDS:
            continue
        for t in TARGETS:
            actual = rec["targets"][t] - res.base_targets[t]
            if math.isnan(actual):
                continue
            row = {
                "dataset": dataset, "config": config_name, "kind": rec["kind"], "label": rec.get("label", ""),
                "size": rec.get("size", 0), "deal": rec.get("deal", ""), "iteration": rec.get("iteration", ""),
                "fraction": rec.get("fraction", ""), "target": t, "actual": actual,
            }
            for m in methods:
                row[f"pred_{m}"] = predict(impl[m], rec, t, hands.deal_labels) if t in impl[m].targets else math.nan
            rows.append(row)
    return rows
