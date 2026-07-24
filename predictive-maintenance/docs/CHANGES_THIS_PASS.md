# What Was Done — Production-Evidence Hardening Pass

One consolidated record of everything changed in this pass: defects fixed, files
added, real measurements, and what remains unverified. Branch
`harden/production-evidence`, commit `3eac498`. **Resume bullets unchanged.**

---

## 1. Starting point

A prior session had already implemented most of the system (feature contract,
checkpoint validation, retrain guard, performance monitor, MLflow lifecycle,
Prometheus/Grafana configs, initial benchmark scripts). Baseline before this
pass: **230 passed, 22 skipped, 3 failed**. This pass hardened, measured, and
documented on top of that — it was not a rewrite.

---

## 2. Real defects found and fixed

### 2.1 Prometheus test-isolation crash (flaky suite)
- **Symptom:** 3 metrics tests failed in the full run but passed in isolation.
- **Cause:** `metrics.py` registered Prometheus collectors at import; when the
  module was imported under a second identity / reloaded, it re-registered →
  `Duplicated timeseries in CollectorRegistry`.
- **Fix:** `_get_or_create()` in
  [`inference_service/api/metrics.py`](../inference_service/api/metrics.py)
  reuses an already-registered collector. Suite is now order-independent.

### 2.2 Service could not boot (slowapi)
- **Symptom:** `uvicorn inference_service.api.main:app` raised at import.
- **Cause:** `slowapi` requires each rate-limited endpoint to have a parameter
  literally named `request`; several endpoints named it `req`, and the predict
  endpoints attached the limiter to a Pydantic param coincidentally named
  `request` (broken at call time).
- **Fix:** renamed params in
  [`inference_service/api/main.py`](../inference_service/api/main.py)
  (`payload` for the body model, `request` for the `Request`). The service now
  imports and starts with the MLP loaded. Defends claim 3 — a service that
  can't import can't deploy.

### 2.3 MLflow tracking-URI env override (12-factor)
- **Cause:** config hardcoded `http://mlflow:5000`; a local run stalled on DNS
  retries.
- **Fix:** `MLFLOW_TRACKING_URI` env now overrides the config default in
  [`inference_service/models/model_manager.py`](../inference_service/models/model_manager.py).

**Result:** **238 passed, 22 skipped, 0 failed** (added regression tests).

---

## 3. Real measurements (executed → `artifacts/*.json`)

| Metric | Value | Artifact |
|---|---|---|
| MAE reduction from temporal features | **15.9%** (10.81 → 9.09) | `model_comparison.json` |
| within-15-cycles rate (best model) | **80.1%** | `model_comparison.json` |
| MAE reduction vs Ridge baseline | 29.9% | `model_comparison.json` |
| Model-compute p99 (CPU, 500 runs) | **~0.65 ms** | `model_compute_latency.json` |
| Local API p99 (in-process ASGI, single) | **~2.1 ms** | `local_api_latency.json` |
| Local API p99 (batch-8) | ~9.5 ms | `local_api_latency.json` |
| CPU inference across batch 1/8/16/32 | measured | `cpu_vs_gpu_report.json` |
| Redis cache-hit p99 (real brew Redis) | **~0.14 ms** | `cache_latency.json` |
| Redis cache-miss get p99 | ~0.11 ms | `cache_latency.json` |
| Load test 10/25/50 users, **0 errors** | ~300 rps, p99 44→160→270 ms | `load_test_summary.{json,md}` |

All reproducible; commands are in each artifact and in the benchmark scripts.
C-MAPSS FD001, single seed, local CPU (simulated data). Redis + load numbers
were obtained after installing a real Redis server (brew) and running Locust
against a bound uvicorn socket with the rate limiter disabled for capacity.

---

## 4. Files added

**Config / SLO**
- `config/slo.yaml` — engineering targets (not production guarantees).

**Benchmarks** (`benchmarks/`)
- `benchmark_cpu_vs_gpu.py` — CPU vs GPU across batch sizes (GPU honestly skipped).
- `benchmark_cache.py` — Redis hit/miss latency (skips with reason if no server).
- `benchmark_prediction_api.py` — in-process ASGI local-API latency.

**Load tests** (`load_tests/`)
- `locustfile.py` + `scenarios/README.md` — concurrency/endurance matrix, SLO gates.

**Docs** (`docs/`)
- `service_level_objectives.md`, `client_facing_workflow.md`, `api_reference.md`,
  `model_card.md`, `monitoring_guide.md`, `retraining_guide.md`,
  `limitations.md`, `aws_architecture.md`, `aws_deployment_runbook.md`,
  `aws_teardown_runbook.md`, and this file.

**Artifacts / evidence** (`artifacts/`)
- `model_comparison.json`, `model_compute_latency.json`, `local_api_latency.json`,
  `cpu_vs_gpu_report.{json,md}`, `cache_latency.json` (skipped),
  `interview_evidence_report.{json,md}` (generated from the above),
  `operational_impact_report.md`, `aws_deployment_evidence.md` (not deployed),
  `generate_evidence_report.py`.

**Tests**
- `tests/unit/test_app_boot_and_benchmarks.py` — guards the boot + idempotency
  fixes and benchmark-module import.

**Files modified:** `metrics.py`, `main.py`, `model_manager.py`.

---

## 5. What is NOT proven here (blocked by environment)

| Item | Why | Status |
|---|---|---|
| GPU acceleration benefit | No CUDA device on this host | code path exists; **unverified** |
| Multi-worker / multi-node scale | Load test used 1 worker on loopback | single-worker measured; horizontal scale **unverified** |
| AWS deployment + deployed latency | No AWS credentials | defs validated; **not deployed** |
| Live drift / production feedback | Controlled C-MAPSS stream, not live labels | mechanism tested; **simulated** |

> Update: Redis cache latency and the bound-socket load test — previously
> blocked — were **completed** in a follow-up sub-pass (real brew Redis + Locust
> against a bound uvicorn socket). See §3.

Full boundary statements in [`limitations.md`](limitations.md). Nothing above
was faked or substituted with a different number.

---

## 6. Dependencies installed this pass
`redis`, `locust`, `slowapi` (into the project venv). Work is committed to
branch `harden/production-evidence` — **not pushed**.

---

## 7. To close the unverified gaps
1. Run `load_tests/` against a bound server with Redis up (unrestricted host).
2. Run `benchmarks/benchmark_cpu_vs_gpu.py` on a CUDA host.
3. Deploy per `docs/aws_deployment_runbook.md`, then fill
   `artifacts/deployed_api_latency.json` and `aws_deployment_evidence.md`.

---

## 8. Requirement-by-requirement traceability (the original 21-section prompt)

Legend: ✅ done · 🟡 partial · ⛔ blocked (environment) · ❌ not done.
"Prior" = existed before this pass; "This pass" = added/verified now.

### 1. Client-facing API flow — 🟡
- ✅ Endpoints present: `/predict/rul`, `/predict/batch`, `/predict/health`,
  `/health`, `/metrics`, `/models`, `/feedback`, `/models/reload`, `/train`.
- ✅ Input validation: missing/non-numeric/NaN/Inf, empty sequence, oversized
  batch, malformed JSON, bounds — `schemas.py` + feedback bounds in `main.py`.
- ✅ Structured errors (code, message, request_id, timestamp, retriable) —
  `error_handler.py`.
- ❌ **Not added:** `/ready`, `/version`, `/models/current`, `/retraining/status`,
  `/retraining/evaluate`. Response lacks some requested fields (`cycle_id`,
  `feature_schema_hash`, `cache_status`, `device`, `latency_ms`, `request_id`
  in the body). **Remaining:** add those read-only endpoints + response fields.

### 2. Under-300ms latency evidence — 🟡
- ✅ Model-compute, local-API (in-proc ASGI), CPU, **cache-hit/miss (real
  Redis)**, and **bound-socket load** benchmarks run → real artifacts with
  p50/p95/p99, env metadata, SLO gate.
- ⛔ Deployed-endpoint latency (not deployed).
- 🟡 **Remaining:** deployed benchmark on AWS; separate cold-start artifact.

### 3. CPU vs GPU — 🟡
- ✅ `benchmark_cpu_vs_gpu.py` measures CPU across batch 1/8/16/32, derives
  conclusion from data, honestly reports GPU skipped.
- ⛔ GPU half unmeasured (no CUDA). **Remaining:** run on a GPU host.

### 4. Load / concurrency / endurance — ✅ / 🟡
- ✅ `load_tests/` locustfile + scenario matrix + SLO gate.
- ✅ **Executed**: 10/25/50 users, real Redis, 0 errors, ~300 rps,
  p99 44→160→270 ms; saturation ~300 rps (1 worker, CPU-bound). Artifact:
  `load_test_summary.{json,md}`.
- 🟡 **Remaining:** higher concurrency, 30/60-min endurance, multi-worker/AWS.

### 5. SLOs — ✅
- ✅ `config/slo.yaml` + `docs/service_level_objectives.md`; benchmarks/load
  tests gate on it. Marked engineering-target, not production guarantee.

### 6. Redis cache production behavior — 🟡
- ✅ (Prior + verified) model+schema-aware key, TTL 300s, hit/miss/fallback,
  no-stale-across-version — covered by unit tests.
- ✅ **Latency measured** against a real Redis server: hit p99 ≈ 0.14 ms,
  miss-get p99 ≈ 0.11 ms (`cache_latency.json`).

### 7. Async processing & request control — 🟡
- ✅ Async FastAPI endpoints, non-blocking fallback, background retrain thread,
  rate limits/connection concerns present.
- 🟡 No dedicated measurement of async benefit / queue-saturation tests.
  **Remaining:** concurrency/timeout/backpressure tests + a measured writeup.

### 8. Prometheus & Grafana — ✅ (Prior, verified this pass)
- ✅ Metrics for requests/latency/cache/model-version/rolling-accuracy/
  retraining; `reliability_dashboard.json`; alert rules. Config + unit tested.
- 🟡 No live-traffic Grafana render (not a code gap).

### 9. Exact 80% quality rule — ✅ (Prior, verified)
- ✅ within-15-cycles definition, threshold 0.80 (strict-below), min samples,
  guard states, guarded promotion; unit + integration tests.
- 🟡 Feedback is a controlled stream, not live labels (documented).

### 10. MLflow versioning — ✅ (Prior) / 🟡
- ✅ Runs, params, metrics, dataset, lifecycle metadata tracked; local file
  store works. **Remaining:** managed Model Registry aliases/stages if a server
  is used.

### 11. Feature contract & training-serving consistency — ✅ (Prior, verified)
- ✅ Shared `feature_contract.py`, checkpoint contract validated at load,
  parity + failure tests (swapped features, wrong scaler, stale schema).

### 12. Production-style failure handling — ✅ (Prior, verified)
- ✅ Missing/corrupt/incompatible checkpoint, Redis outage/timeout, invalid
  request, MLflow outage, retrain/promotion failure, rollback — failure-
  injection tests. Errors structured; last-known-good preserved.

### 13. AWS deployment evidence — 🟡 / ⛔
- ✅ `infra/aws/` defs, IAM least-privilege, deploy/verify scripts;
  `docs/aws_architecture.md` + deploy/teardown runbooks;
  `artifacts/aws_deployment_evidence.md` (states NOT deployed).
- ⛔ No execution (no creds). **Remaining:** actually deploy + capture evidence.

### 14. Deployment safety & rollback — ✅ (Prior, verified)
- ✅ Current/challenger/known-good pointers, validation-before-activate,
  rollback, cache isolation by version; tests.

### 15. Security & privacy — ✅ (Prior, verified)
- ✅ API-key auth, rate limiting, security headers, request-size limit, secrets
  via env (none in source), safe errors (no stack traces), C-MAPSS=simulated
  noted. Dependency pinning present (`requirements-lock.txt`).

### 16. Client-facing documentation — ✅ (This pass)
- ✅ `client_facing_workflow.md`, `api_reference.md`, `model_card.md`,
  `monitoring_guide.md`, `retraining_guide.md`, `limitations.md` with examples.

### 17. Business/operational impact — ✅ (This pass)
- ✅ `artifacts/operational_impact_report.md` — measured vs inferred vs intended,
  no fabricated revenue/downtime/adoption.

### 18. Interview evidence report — ✅ (This pass)
- ✅ `artifacts/interview_evidence_report.{md,json}` generated from real
  artifacts, with a status matrix.

### 19. Testing requirements — ✅ / 🟡
- ✅ 238 passed / 22 skipped; unit, integration, contract, cache, monitoring,
  feedback, retraining, promotion, rollback, failure-injection, boot regression.
- 🟡 Coverage % not computed; live-service + load tests not run here.
  **Remaining:** coverage report + live/load tests on real infra.

### 20. Implementation process — ✅
- ✅ Inspected repo first, fixed correctness/boot before adding evidence,
  phased order followed.

### 21. Final report — ✅
- ✅ This document + commit message + evidence report.

---

## 9. Honest one-line summary of remaining work
Redis cache latency and the load test are now **done** (real server + bound
socket). What still needs **real infrastructure**: a **GPU host** (CPU-vs-GPU),
an **AWS account** (deploy + deployed-endpoint latency), and a bigger box for
**endurance / multi-worker / higher-concurrency** scale. Plus optional API
polish: the extra read-only endpoints (`/ready`, `/version`, `/models/current`,
`/retraining/status`) and the additional response fields in §1. No code logic
is blocking these — only environment access.
