# MASTER EVIDENCE — Predictive-Maintenance RUL System

**One self-contained record of everything built, measured, and deployed.**
Safe to hand to another chat/person as the single source of truth. Repo:
https://github.com/Vkartik-3/ineuronAI (branch `main`). All numbers below were
executed locally or on real AWS — none are hypothetical. Honest limits are in
§7.

The 3 resume claims (unchanged, all now defensible):
1. Lowered turbofan RUL error ~15% with a PyTorch MLP on NASA C-MAPSS + temporal features.
2. Automated monitoring + retraining on AWS (MLflow, Prometheus/Grafana), retrain <80% accuracy.
3. Real-time predictions <300 ms via FastAPI + GPU support + Redis cache + async.

---

## 1. Headline real numbers

| Area | Metric | Value | Source artifact |
|---|---|---|---|
| **Model quality** | MAE, temporal-feature MLP | **9.09** cycles | `artifacts/model_comparison.json` |
| | MAE, raw-sensor MLP | 10.81 | same |
| | **Improvement from temporal features** | **15.9%** | same (backs claim 1) |
| | Improvement vs Ridge baseline | 29.9% | same |
| | within-15-cycles accuracy | 80.1% | same |
| **Model compute** (CPU) | p50 / p99 | 0.61 / **0.65 ms** | `artifacts/model_compute_latency.json` |
| **Local API** (in-proc ASGI) | single p99 | **2.1 ms** | `artifacts/local_api_latency.json` |
| | batch-8 p99 | 9.5 ms | same |
| **Redis cache** (real server) | hit p99 / miss-get p99 | **0.14 / 0.11 ms** | `artifacts/cache_latency.json` |
| **Load test** (bound socket, real Redis) | 10 / 25 / 50 users p99 | 44 / 160 / **270 ms**, 0 errors | `artifacts/load_test_summary.json` |
| | sustained throughput | ~300 rps (1 worker) | same |
| **Deployed AWS** (ECS Fargate + ALB + Redis) | cache-miss p99 / hit p99 | **261 / 199 ms**, 0 errors | `artifacts/deployed_api_latency.json` |
| **Tests** | passing / skipped | **238 / 22** (249/11 with live server) | `pytest tests/` |

Every latency is under the **300 ms SLO**.

---

## 2. Claim-by-claim defense

**Claim 1 (15% error reduction).** Reproducible: `python
ml_pipeline/evaluate/baseline_comparison.py --mlflow-uri file:./mlruns_baseline
--data-path ../archive/CMaps`. Adding lag/rolling/EMA temporal features to the
same MLP cut test MAE 10.81 → 9.09 = **15.9%**; within-15 rose to 80.1%.

**Claim 2 (monitor + retrain <80% on AWS, MLflow + Prometheus/Grafana).**
- Retrain trigger = within-15 accuracy **< 80%** (exactly 80% does NOT trigger),
  min 50 samples. Config: `ml_pipeline/retrain/config/retrain_config.yaml`.
- **Guarded promotion**: a challenger must beat the incumbent MAE by 5% or it is
  rejected; last known-good preserved; rollback supported. Tests in
  `test_retrain_guard.py`, `test_controlled_retraining_event.py`.
- Prometheus `/metrics` exposes service + model-quality signals
  (`model_rolling_accuracy`, `retraining_triggered_total`, `model_version_info`,
  cache counters, latency histogram). Grafana `reliability_dashboard.json`;
  alerts in `infra/prometheus/alert_rules.yml`. MLflow tracks runs/params/
  metrics/lifecycle.

**Claim 3 (<300 ms via FastAPI + GPU + Redis + async).**
- <300 ms proven at every layer (see §1): compute 0.65 ms, local API 2.1 ms,
  deployed p99 261 ms.
- Redis cache is **model+schema-aware** (key includes model version + schema),
  TTL 300 s, non-blocking fallback on outage. Latency measured on a real server.
- Async FastAPI; rate limiter verified live (429 on burst in prod).
- **GPU**: code path selects device at runtime; benefit is UNVERIFIED on
  hardware (no CUDA available). CPU single-request already meets the SLO, so the
  low-latency path runs on CPU. Honest — see §7.

---

## 3. Real AWS deployment (executed, then torn down)

- **Account** 845242190324, **region** us-east-2.
- **Stack:** ECR image → ECS **Fargate** (2 vCPU/4 GB, 1 task) behind an
  **ALB**, with **ElastiCache Redis** (cache.t3.micro). CloudWatch logs.
- **Live proof:** ALB target health `healthy`; `/health` 200; `/predict/rul`
  returned RUL 79.38 cycles ("warning", conf 0.74) for engine_042; rate limiter
  fired 429 on burst.
- **Deployed latency** (paced under rate limit, 0 errors): cache-miss p99
  **261 ms**, cache-hit p99 199 ms. Network-RTT bound (home Mac → Ohio ~100 ms);
  server compute is sub-ms.
- **Torn down & verified:** no ECS/ALB/target-group/ElastiCache/ECR/SG left.
  Only free IAM roles remain. Cost: **< $1** (~30 min runtime).
- Full details: `artifacts/aws_deployment_evidence.md`.
- **Prod-correct CI path also in repo:** GitHub Actions + OIDC (no static keys)
  — `.github/workflows/deploy.yml`, `docs/aws_oidc_setup.md`.

---

## 4. Real defects found & fixed (this is the strong interview material)

1. **Service couldn't boot** — slowapi requires each rate-limited endpoint to
   have a param named `request`; several used `req`. Renamed → app now imports/
   starts. (`main.py`)
2. **Flaky test suite** — Prometheus collectors re-registered under multi-path
   import → `Duplicated timeseries`. Made registration idempotent. (`metrics.py`)
3. **Docker image never ran** — wrong entrypoint/context; `requirements.txt`
   pulled tensorflow but omitted torch+redis; `tf.keras.Model` annotation
   crashed import; kafka-python blocked startup on DNS retries. Fixed via
   `Dockerfile.aws`, `requirements-aws.txt`, `from __future__ import
   annotations`, and omitting kafka-python.
4. **MLflow DNS stalls** — env `MLFLOW_TRACKING_URI` now overrides the baked-in
   docker default (12-factor).
5. **3 integration tests silently skipped** (app couldn't import) had outdated
   payloads (list-of-lists, missing timestamp) — corrected to the real schema;
   now pass live. Validation was NOT weakened.
6. Added `RATE_LIMIT_ENABLED` env toggle for capacity testing.

---

## 5. What was added (files)

- **Config/SLO:** `config/slo.yaml`, `docs/service_level_objectives.md`.
- **Benchmarks:** `benchmarks/benchmark_cpu_vs_gpu.py`, `benchmark_cache.py`,
  `benchmark_prediction_api.py` (+ existing `benchmark_latency.py`, `_http.py`).
- **Load tests:** `load_tests/locustfile.py` + scenario matrix.
- **Docs:** client_facing_workflow, api_reference, model_card, monitoring_guide,
  retraining_guide, limitations, aws_architecture, aws_deployment_runbook,
  aws_teardown_runbook, aws_oidc_setup, CHANGES_THIS_PASS, this file.
- **AWS runtime:** `inference_service/Dockerfile.aws`, `requirements-aws.txt`,
  `.github/workflows/deploy.yml`, `infra/aws/github_oidc_trust_policy.json`.
- **Artifacts (real data):** model_comparison, model_compute_latency,
  local_api_latency, cache_latency, cpu_vs_gpu_report, load_test_summary,
  deployed_api_latency, interview_evidence_report, operational_impact_report,
  aws_deployment_evidence.
- **Tests:** `test_app_boot_and_benchmarks.py` + fixed integration tests.

---

## 6. How to reproduce (commands)

```bash
# model MAE comparison (claim 1)
python ml_pipeline/evaluate/baseline_comparison.py --mlflow-uri file:./mlruns_baseline --data-path ../archive/CMaps
# compute + API + cpu/gpu benchmarks
python inference_service/benchmark_latency.py --n-runs 500
python benchmarks/benchmark_cpu_vs_gpu.py --n-runs 300
MLFLOW_TRACKING_URI=file:./mlruns python benchmarks/benchmark_prediction_api.py --n-runs 300
# cache (needs a Redis server) + load test (needs bound server, rate limit off)
REDIS_HOST=localhost python benchmarks/benchmark_cache.py --n-runs 1000
RATE_LIMIT_ENABLED=false uvicorn inference_service.api.main:app --port 8000  # then:
locust -f load_tests/locustfile.py --host http://127.0.0.1:8000 --users 50 --run-time 25s --headless
# regenerate the evidence report from artifacts
python artifacts/generate_evidence_report.py
# full test suite
pytest tests/
```

---

## 7. Honest limits (say these out loud)

- **NASA C-MAPSS is simulated data.** Not validated for real aircraft / safety.
- **GPU benefit is unverified** (no CUDA here) — code path exists; CPU meets SLO.
- **Deployed latency is network-RTT bound** (home internet → us-east-2); a
  same-region client would be much faster. Not measured.
- **No standing production deployment** — it was deployed, measured, torn down.
- **Retraining uses a controlled feedback stream**, not live production labels;
  it is guarded *retraining*, not automatic *promotion*.
- **Scale tested at 1 worker on loopback** — multi-worker/multi-node not tested.
- All hardening here was **added in this exercise**; do not present it as
  original-internship work unless separately confirmed.

---

## 8. Status matrix

| Area | implemented | tested | benchmarked | deployed |
|---|---|---|---|---|
| Model + temporal features | ✅ | ✅ | ✅ | ✅ (in image) |
| API serving | ✅ | ✅ | ✅ | ✅ (Fargate) |
| Redis cache | ✅ | ✅ | ✅ (real server) | ✅ (ElastiCache) |
| GPU path | ✅ | ✅ | ⛔ no CUDA | — |
| Retraining guard | ✅ | ✅ | n/a | — |
| Monitoring (Prom/Grafana) | ✅ | ✅ | n/a | — |
| Load (10/25/50 users) | ✅ | ✅ | ✅ | — |
| AWS deploy | ✅ | ✅ (live) | ✅ (p99 261 ms) | ✅ then torn down |
| CI/CD (OIDC) | ✅ | — | — | ready |
