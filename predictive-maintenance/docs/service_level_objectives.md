# Service-Level Objectives (SLOs)

Source of truth for the numbers: [`config/slo.yaml`](../config/slo.yaml).
Every result below is tagged with **where it was measured** and **what is still
unverified**. These are **engineering targets**, not production guarantees —
see [limitations.md](limitations.md).

## Why each SLO exists

| SLO | Target | Backs resume claim | Measured where |
|---|---|---|---|
| Model-compute p99 | < 300 ms | 3 (under-300ms) | local CPU ✅ (`artifacts/model_compute_latency.json`) |
| Local API p99 | < 300 ms | 3 | in-process ASGI ✅ (`artifacts/local_api_latency.json`) |
| Cache-hit p99 | < 50 ms | 3 (Redis) | real brew Redis ✅ (`artifacts/cache_latency.json`, hit p99 ≈ 0.14 ms) |
| Deployed API p99 | (unset) | 3 (AWS) | ⛔ not deployed — no target enforced |
| Error rate | < 1% | 2 | load test ✅ 0 errors @ 10/25/50 users (`artifacts/load_test_summary.json`) |
| within-15-cycles | ≥ 80% to serve; retrain below | 2 | local eval ✅ (`artifacts/model_comparison.json` → 80.1%) |
| No stale cache across versions | invariant | 2/3 | unit tests ✅ |
| No weaker-model promotion | invariant | 2 | unit tests ✅ (`test_retrain_guard.py`) |

## Measured results (this environment)

- **Model compute** (sequence → feature engineering → MLP forward), CPU,
  500 runs: **p50 ≈ 0.61 ms, p99 ≈ 0.65 ms**. Full split (preprocess vs model)
  in the artifact. **PASS** (< 300 ms).
- **Local API** (in-process ASGI, cache-miss), 300 runs: **single p99 ≈ 2.1 ms**,
  batch-8 p99 ≈ 9.5 ms. **PASS** (< 300 ms).
- **Cache hit / miss** (real brew Redis, 1000 runs): **hit p99 ≈ 0.14 ms**,
  miss-get p99 ≈ 0.11 ms. **PASS** (< 50 ms).
- **Load test** (bound socket, real Redis, 1 uvicorn worker): 10/25/50 users,
  **0 errors**, ~300 rps sustained, p99 44 → 160 → 270 ms. All within SLO;
  saturation begins ~300 rps. See `artifacts/load_test_summary.md`.
- **Deployed endpoint**: **NOT MEASURED** — no AWS deployment executed.

## Reporting contract

Every SLO report — in benchmarks, load tests, or the evidence report — must
state: the **SLO**, the **workload**, the **environment**, the **observed
value**, **pass/fail**, and the **limitation**. A number without those five is
not accepted as evidence.

## What would upgrade these to production SLOs

Measure p99 / error rate / availability against a **deployed** endpoint under
**real** traffic over a defined window, with Redis and autoscaling in the loop.
Until then `meta.status` in `slo.yaml` stays `engineering-target`.
