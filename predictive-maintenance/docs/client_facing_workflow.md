# Client-Facing Workflow — Plain Language

For a non-ML stakeholder who wants to understand what the service does and how
to read a prediction.

## The domain in one paragraph

Each **engine unit** is one simulated turbofan engine. A **cycle** is one
flight/operating cycle. **RUL** (Remaining Useful Life) is how many more cycles
the engine is predicted to run before failure. Lower RUL = closer to failure =
sooner maintenance. The model reads a window of recent sensor readings and
predicts RUL.

## What one request looks like

The downstream team POSTs one engine's recent sensor readings to
`/predict/rul` and gets back a single validated JSON object — no notebook, no
rerunning training. See [api_reference.md](api_reference.md) for the exact
shape.

## How to read the response

| Field | Meaning |
|---|---|
| `predicted_rul_cycles` / `rul_cycles` | predicted cycles until failure |
| `health_status` | `healthy` / `warning` / `critical` / `imminent_failure`, derived from RUL thresholds |
| `confidence` / `confidence_interval` | model's uncertainty band around the RUL |
| `model_version` | which model produced it (changes after a promotion) |
| `recommendations` | plain-language next action |
| `timestamp` | when it was produced |

**Health status thresholds** (configurable, `inference_config.yaml`): warning
≤ 50 cycles, critical ≤ 30, imminent ≤ 10.

## What caching does

Identical repeated requests for the same engine + same model are served from a
short-lived (300 s) Redis cache, so they return faster. The cache key includes
the **model version and feature-schema version**, so a model update never
serves a stale prediction from the old model.

## What monitoring shows

Prometheus + Grafana expose latency (p50/p95/p99), error rate, cache hit ratio,
current model version, rolling within-15-cycles accuracy, and retraining state.
See [monitoring_guide.md](monitoring_guide.md).

## What happens when quality drops

If the rolling within-15-cycles accuracy falls **below 80%**, a **guarded
retraining** workflow starts. A challenger model is trained and must **beat the
current model** on MAE before it is promoted. A weaker challenger is rejected;
the last known-good model stays live. See [retraining_guide.md](retraining_guide.md).

## Honest boundaries

C-MAPSS is **simulated** data. This service is a demonstration of production-
style ML serving; it is **not** validated for real aircraft safety decisions.
See [limitations.md](limitations.md).
