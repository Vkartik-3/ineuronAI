# Operational Impact Report

Separates **directly measured** technical impact from **inferred** workflow
benefit and **intended** business value. No revenue, downtime, adoption, or
safety outcomes are claimed — none are measured.

## A. Directly measured (from artifacts / tests)

| Metric | Value | Source |
|---|---|---|
| MAE reduction from temporal features | 15.9% (10.81→9.09) | `model_comparison.json` |
| within-15-cycles rate (best model) | 80.1% | `model_comparison.json` |
| Model-compute p99 (CPU) | ~0.65 ms | `model_compute_latency.json` |
| Local API p99 (in-proc ASGI, single) | ~2.1 ms | `local_api_latency.json` |
| Local API p99 (batch-8) | ~9.5 ms | `local_api_latency.json` |
| Passing tests | 233 (22 skipped) | `pytest tests/` |
| Validation checks on prediction input | 12+ rejection cases | `schemas.py`, `error_handler.py` |
| Failure scenarios covered by tests | Redis outage, corrupt/incompatible checkpoint, model-timeout, MLflow outage, retrain/promotion failure, rollback | `test_audit_fixes.py`, `test_critical_fixes.py`, `test_retrain_guard.py` |
| Boot defect fixed | slowapi rate-limiter param naming (service now starts) | this pass |
| Test-isolation defect fixed | Prometheus duplicate-registration (suite now order-independent) | this pass |

## B. Inferred workflow benefit (reasonable, not dollar-quantified)

- One API call returns a **validated** RUL with model version, health status,
  confidence, and latency — replacing "rerun the notebook" for the downstream
  team.
- Repeated identical requests are cache-eligible (model/schema-aware key), so a
  hot equipment ID avoids recomputation (latency benefit real; *magnitude*
  unmeasured here — no Redis server).
- Quality degradation is **detectable** via `model_rolling_accuracy` and
  **acted on** by guarded retraining, reducing the window a degraded model
  serves unnoticed.

## C. Intended business value (NOT realized/measured)

- Supports predictive-maintenance planning; reduces dependence on manual ML
  execution; improves visibility into model degradation; reduces risk of
  serving stale/incompatible models; makes client evaluation more repeatable.

These are **intentions**, explicitly not converted into financial or downtime
claims. See [docs/limitations.md](../docs/limitations.md).
