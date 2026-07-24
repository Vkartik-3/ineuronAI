# Monitoring Guide

Backs resume claim 2: Prometheus tracks service **and** model-quality signals;
Grafana makes them visible.

## Metrics exposed (`GET /metrics`)

Instruments defined in [`api/metrics.py`](../inference_service/api/metrics.py)
(registration is idempotent — safe under multi-path import / worker reload):

**Service:** `inference_requests_total{model,status}`,
`inference_latency_seconds` (histogram), `models_loaded`,
`kafka_pipeline_running`, `service_uptime_seconds`.

**Cache:** `prediction_cache_hits_total`, `prediction_cache_misses_total`.

**Model quality / lifecycle:** `model_rolling_accuracy{model}` (the same number
compared to the 80% threshold), `feedback_records_total`,
`retraining_triggered_total{reason}`, `model_version_info{model,version}`,
`prediction_rul_hours{equipment_id}`.

## Grafana dashboard

[`dashboard/grafana/reliability_dashboard.json`](../dashboard/grafana/reliability_dashboard.json)
panels: rolling within-15 accuracy, retraining triggers, API latency
(p50/p95/p99), request errors, cache hit ratio, model version, Kafka consumer
status, service uptime. Each panel carries a PromQL target (asserted by
`tests/unit/test_monitoring_and_device.py`).

## Alert rules

[`infra/prometheus/alert_rules.yml`](../infra/prometheus/alert_rules.yml):
`RollingAccuracyBelowThreshold`, `ApiP99LatencyAbove300ms`, `ModelUnavailable`,
`KafkaPipelineDown`, `RedisUnavailable`, `ExcessiveCacheErrors` — each with an
`expr` and a `severity` label (validated by tests).

## Reading it during an incident

- p99 latency rising → check cache hit ratio + model-compute vs total split.
- `model_rolling_accuracy` < 0.80 → retraining will trigger; watch
  `retraining_triggered_total` and `model_version_info` for a promotion.
- `RedisUnavailable` firing → predictions still served (fallback to compute),
  but cache-hit latency benefit is lost.

## Scope

Metrics are wired and unit-tested. A live Grafana render against production
traffic is **not** included — dashboards/alerts are validated by config/unit
tests, not by a screenshot of live data. See [limitations.md](limitations.md).
