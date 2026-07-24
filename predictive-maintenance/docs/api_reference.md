# API Reference — Predictive-Maintenance Inference Service

FastAPI service (`inference_service/api/main.py`). Interactive docs at `/docs`
when running. Auth: `X-API-Key` header when `API_KEYS` env is set; skipped in
dev/test. `/health`, `/`, `/docs`, `/metrics` are always public.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/` | public | service banner |
| GET | `/health` | public | status + dependency health (Redis/Kafka/DB) + `cuda_available` + per-model device |
| GET | `/metrics` | public | Prometheus exposition |
| GET | `/models` | key | loaded models + versions + device |
| POST | `/predict/rul` | key | RUL prediction for one engine sequence |
| POST | `/predict/batch` | key | RUL for many sequences (≤ 100) |
| POST | `/predict/health` | key | health-class prediction (random forest) |
| POST | `/feedback` | key | record ground-truth RUL for a prior prediction |
| POST | `/models/reload` | admin | reload models (rate-limited 5/hour) |
| POST | `/train` | admin | trigger guarded retraining in background |

> Rate limits: predict endpoints 100/min, feedback 200/min, reload 5/hour,
> train 2/hour (slowapi, per client IP).

## POST /predict/rul

Request:

```json
{
  "data": {
    "equipment_id": "engine_001",
    "sequence": [
      {"sensor_1": 518.67, "sensor_2": 641.8, "...": 0.0,
       "op_setting_1": 0.0, "op_setting_2": 0.0, "op_setting_3": 100.0,
       "time_cycle": 1.0}
    ]
  },
  "return_confidence": true
}
```

- `sequence`: ordered readings, oldest→newest. The last 50 are used; the
  temporal feature engineer needs enough history to fill lag/rolling/EMA
  features. All 21 raw sensors may be sent; 7 constant ones are dropped
  internally per the [feature contract](../shared/feature_contract.py).

Response (200):

```json
{
  "equipment_id": "engine_001",
  "rul_cycles": 87.5,
  "rul_hours": 43.75,
  "health_status": "warning",
  "confidence": 0.82,
  "confidence_interval": {"lower": 78.1, "upper": 96.9, "confidence": 0.82},
  "model_version": "v1.0.0",
  "recommendations": ["Schedule inspection within 40 cycles"],
  "timestamp": "2026-07-24T06:54:23Z"
}
```

### Validation errors (structured, non-fake)

Bad input returns a structured error (code, message, request_id, timestamp,
retriable flag) via `error_handler.py` — never a fabricated RUL. Rejected:
missing/non-numeric/NaN/Inf sensor values, empty sequence, wrong feature count,
malformed JSON, oversized batch. If no model is loaded the service returns
`503 model_unavailable` — it does not guess.

## POST /feedback

Query params: `equipment_id`, `true_rul`, `predicted_rul`, optional `timestamp`.
Stored in the process-level `PerformanceMonitor` (Redis-backed when available)
and folded into the rolling within-15-cycles accuracy that drives the 80%
retraining trigger. Bounds-validated (finite, non-negative, ≤ 1000).

## GET /health

Returns `status` (healthy/degraded), per-dependency latency, `models_loaded`,
`cuda_available`, and `model_devices` — a live evidence trail for where the MLP
actually runs (CPU vs CUDA), backing resume claim 3.
