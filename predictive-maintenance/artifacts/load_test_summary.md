# Load Test Summary (bound socket, real Redis)

Generated 2026-07-24T07:28:18.448869+00:00 | commit 3eac498
Env: uvicorn 1 worker, localhost:6379 (real, brew redis), rate-limit disabled. Loopback single-host — **not** an AWS/multi-node number.

| Users | RPS | p50 | p95 | p99 | max | reqs | errors | err% |
|---|---|---|---|---|---|---|---|---|
| 10 | 238.8 | 13.0 | 34.0 | 44.0 | 170.0 | 5737 | 0 | 0.00% |
| 25 | 304.2 | 52.0 | 89.0 | 160.0 | 210.0 | 7307 | 0 | 0.00% |
| 50 | 301.3 | 130.0 | 220.0 | 270.0 | 360.0 | 7240 | 0 | 0.00% |

## Findings
- Max sustained throughput: **~304.2 rps** (single uvicorn worker).
- Highest concurrency within SLO (p99<300ms, err<1%): **50 users**.
- Saturation: throughput plateaus ~300 rps at 25->50 users while p99 rises 160->270 ms; single uvicorn worker is CPU-bound on the sync inference path.
- First SLO at risk: p99 latency (approaches 300 ms at 50 users); error rate stayed at 0.

## Limitations
1 worker, loopback, simulated data; higher concurrency and multi-worker/AWS not tested here