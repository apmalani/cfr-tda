# cfrtda

Training data attribution (TDA) for counterfactual regret minimization (CFR) on Kuhn poker: which dealt hands caused the solver to make a given decision?

Standard influence functions assume training settles at a loss minimum. CFR solves a two-player game instead, so this repo adapts them two ways and checks both against exact retraining.

## Methods

- **Replay (A0, A1, A2)**: differentiates through the CFR run itself. A0 counts only a hand's direct vote into the average strategy, A1 adds its ripple through later regrets, and A2 is the full derivative.
- **Snapshot (B, Bx)**: softens the game with entropy (strength τ), then differentiates its equilibrium with the implicit function theorem, using the non-symmetric game Jacobian J in place of the Hessian. Bx re-solves the equilibrium instead of linearizing.
- **Ground truth**: every removal is retrained exactly, with common random numbers, so differences come from the removed hands alone.

## Layout

| Module | Purpose |
|---|---|
| `game.py`, `kuhn.py` | Tensorized Kuhn poker (via OpenSpiel) |
| `solver.py` | Deterministic, differentiable CFR (RM, RM+, softplus-RM, Hedge, regularized) |
| `data.py` | Training hands: fixed decks and streams |
| `influence.py` | Replay formulas A0–A2 |
| `balance.py` | Snapshot formulas B and Bx |
| `reference.py` | Exploitability and target behaviors (e.g. bluff rate) |
| `removal.py` | Ground-truth retraining |
| `compare.py`, `harness.py` | Predictions vs. truth, linear datamodeling score (LDS) |
| `phase.py` | Gradual single-hand removal (why replay can mislead) |
| `consistency.py` | Checks A2 → B on regularized CFR |

## Usage

```bash
pip install -e .
python -m cfrtda.removal       # ground-truth retrains -> results/ground_truth
python -m cfrtda.harness       # predictions.csv, lds.json -> results/harness
python -m cfrtda.phase         # phase.json
python -m cfrtda.consistency   # consistency.json
```

`results/harness` holds the saved outputs used in the write-up. Training sets are in `data/`.
