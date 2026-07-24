# Model Card — PyTorch MLP RUL Predictor

## Overview
- **Task:** Remaining Useful Life (RUL) regression for turbofan engines.
- **Model:** Feed-forward MLP (`MLPRULNet`, hidden = [256,128,64,32], dropout
  0.2, ~96k params), PyTorch.
- **Input:** 200-dim engineered feature vector per engine, built by the
  canonical [feature contract](../shared/feature_contract.py) from raw C-MAPSS
  sensor readings: 14 non-constant sensors × (raw + 4 lags + 6 rolling +
  3 EMA) + 3 op-settings + cycle.
- **Output:** RUL in cycles (piecewise-linear cap 125), plus a derived
  health-status band and a confidence interval.

## Training data
- **NASA C-MAPSS FD001** — *simulated* run-to-failure turbofan data.
- Per-engine train/val/test split, StandardScaler, seed 42.

## Measured performance (FD001 test split, local CPU)
Source: `artifacts/model_comparison.json` (reproducible via
`ml_pipeline/evaluate/baseline_comparison.py`).

| Model | MAE | RMSE | R² | within-15 |
|---|---|---|---|---|
| Mean baseline | 33.17 | 35.34 | −0.64 | 12.0% |
| Ridge (linear) | 12.97 | 16.22 | 0.65 | 64.3% |
| MLP — raw sensors | 10.81 | 16.19 | 0.66 | 75.5% |
| **MLP — + temporal features** | **9.09** | **14.66** | **0.72** | **80.1%** |

**Adding temporal features lowered MAE by 15.9%** (10.81 → 9.09) — the basis
for resume claim 1. vs Ridge baseline: 29.9%.

## Intended use / out of scope
- **Intended:** demonstrating production-style RUL serving + monitoring +
  guarded retraining on a public benchmark.
- **Out of scope:** any live aircraft / safety-critical maintenance decision.
  C-MAPSS is simulated; not fleet-validated.

## Serving
- Runtime device selected at load; CPU here (no CUDA). Single-request compute
  p99 ≈ 0.65 ms, local API p99 ≈ 2.1 ms — both < 300 ms SLO.
- Training-serving consistency enforced by a shared feature contract + a
  checkpoint contract validated at load (schema version/hash, feature count,
  scaler, input dim).

## Ethical / risk notes
Simulated data limits real-world bias analysis. A wrong RUL in a real setting
could cause premature or missed maintenance — hence the not-for-safety scope
and the guarded-promotion retraining design.
