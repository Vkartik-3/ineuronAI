# Load-test scenarios

All scenarios drive `load_tests/locustfile.py` against a **running** inference
service. Every result must be recorded with its concurrency, environment, and
SLO pass/fail (see `config/slo.yaml`). A bare throughput number is not evidence.

Run the whole matrix and collect CSVs into `artifacts/`:

```bash
scripts/run_load_matrix.sh http://localhost:8000
```

| Scenario | Command sketch | What it proves |
|---|---|---|
| `steady_state` | `--users 25 --spawn-rate 5 --run-time 5m` | sustained p95/p99 + error rate at fixed load |
| `ramp_up` | `--users 100 --spawn-rate 2 --run-time 5m` | latency/error curve as concurrency climbs; first-SLO-violated point |
| `cache_heavy` | reuse ~5 equipment IDs so requests repeat | Redis cache-hit ratio and its p99 impact |
| `cache_miss` | unique equipment ID per request | worst-case (no cache) p99 |
| `batch` | `/predict/batch` weighted heavier | batch throughput + per-item latency |
| `redis_failure` | stop Redis mid-run | service must fall back to compute, stay < error-rate SLO |
| `model_reload` | POST `/models/reload` mid-run | predictions continue; cache isolates old vs new version |

## Concurrency steps (record each)

1, 5, 10, 25, 50, 100 concurrent clients. For each, record: RPS, p50/p95/p99,
max latency, error rate, and which SLO (if any) broke first.

## Endurance

- 30-minute steady-state at the highest concurrency that held all SLOs.
- 60-minute run when the environment allows.
- Record: latency drift, memory growth, Redis errors, error-rate trend.

## Honesty rules (from the project spec)

- Do **not** invent a scale target; record the **maximum verified** sustained
  workload only.
- Local numbers are local. They are **not** AWS-endpoint numbers.
- If Redis / the server / a dependency is down, record the scenario as
  **skipped/unverified with the reason** — never fabricate the run.
