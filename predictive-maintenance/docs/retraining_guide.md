# Retraining Guide — Guarded, Not Blind

Backs resume claim 2: retraining triggers when accuracy drops below 80%. The
key nuance to defend: **automatic retraining ≠ automatic promotion.**

## The 80% rule, precisely

"Accuracy" = **within-15-cycles rate** = fraction of labeled predictions whose
`|predicted_rul − true_rul| ≤ 15 cycles`. Config in
[`retrain_config.yaml`](../ml_pipeline/retrain/config/retrain_config.yaml):

- `accuracy_threshold: 0.80`, `accuracy_tolerance_cycles: 15`
- `min_samples_for_check: 50` — no decision on thin evidence
- Trigger fires **strictly below** 0.80; exactly 0.80 does **not** trigger.

Feedback is recorded via `POST /feedback` into the `PerformanceMonitor`
(Redis-backed sorted set when available, in-memory fallback with a warning).

## Guard states

`RetrainGuard.decide()` ([retrain_guard.py](../ml_pipeline/retrain/retrain_guard.py))
returns one of: `HEALTHY`, `INSUFFICIENT_DATA`, `EVALUATION_UNAVAILABLE`,
`DEGRADED` (→ should trigger), `ALREADY_ACTIVE` (a run is in-flight),
`COOLDOWN_ACTIVE` (within cooldown window). This prevents duplicate/looping
triggers.

## Guarded promotion workflow

1. Below-threshold accuracy → guard says **DEGRADED** → retraining begins.
2. The **current model is preserved**; a **challenger** is trained/loaded.
3. Challenger is **validated**, then **compared** to the incumbent.
4. Promotion requires the challenger to **beat the incumbent** on MAE by
   `min_improvement_pct` (5%). A weaker/equal challenger is **rejected**.
5. On promotion: atomic pointer update, post-swap health check + smoke
   prediction, cache isolates old vs new version. On failure: **rollback** to
   last known-good.

Tested in `tests/unit/test_retrain_guard.py` and
`tests/integration/test_controlled_retraining_event.py`: insufficient samples,
above/at/below 80%, challenger better/equal/worse, training/validation failure,
rollback, duplicate-trigger suppression.

## Honest scope

The feedback stream used here is a **controlled** C-MAPSS evaluation stream, not
live production labels with real delayed ground truth. It demonstrates the
mechanism; it is not evidence of live production drift handling. See
[limitations.md](limitations.md).
