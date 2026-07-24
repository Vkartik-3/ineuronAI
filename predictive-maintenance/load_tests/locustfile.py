"""
Locust load test for the Predictive-Maintenance inference API.

Drives the real HTTP prediction path (/predict/rul, /predict/batch, /health)
so concurrency, throughput, p95/p99 and error rate can be measured against the
SLOs in config/slo.yaml. Requires a RUNNING inference service.

Quick start (local):
    # 1. start the service (models load from models/checkpoints)
    uvicorn inference_service.api.main:app --port 8000
    # 2. run a smoke test, 10 users for 30s, headless
    locust -f load_tests/locustfile.py --host http://localhost:8000 \
           --users 10 --spawn-rate 5 --run-time 30s --headless \
           --csv artifacts/load_smoke

The scenarios/ modules document the specific concurrency steps, endurance
runs, and Redis-failure / model-reload cases required by the SLO suite. Every
result MUST be reported with its concurrency, environment, and SLO pass/fail —
a bare RPS number is not evidence.

If X-API-Key auth is enabled server-side, set LOCUST_API_KEY in the env.
"""

import os
import random

from locust import HttpUser, task, between, events

_API_KEY = os.environ.get("LOCUST_API_KEY")
_SENSORS = [f"sensor_{i}" for i in range(1, 22)]


def _reading(t: int) -> dict:
    r = {s: round(random.uniform(0.5, 1.5), 4) for s in _SENSORS}
    r.update({"op_setting_1": 0.1, "op_setting_2": 0.2, "op_setting_3": 80.0,
              "time_cycle": float(t + 1)})
    return r


def _sequence(n: int = 50) -> list:
    return [_reading(t) for t in range(n)]


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if _API_KEY:
        h["X-API-Key"] = _API_KEY
    return h


class RULPredictionUser(HttpUser):
    """Realistic client: mostly single RUL predictions, some batches + health."""
    wait_time = between(0.0, 0.05)

    @task(10)
    def predict_rul(self):
        eq = f"engine_{random.randint(1, 100):03d}"
        payload = {"data": {"equipment_id": eq, "sequence": _sequence()},
                   "return_confidence": False}
        with self.client.post("/predict/rul", json=payload, headers=_headers(),
                              catch_response=True, name="/predict/rul") as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}")

    @task(2)
    def predict_batch(self):
        seqs = [{"equipment_id": f"engine_{i:03d}", "sequence": _sequence()}
                for i in range(1, 9)]
        with self.client.post("/predict/batch", json={"sequences": seqs},
                              headers=_headers(), catch_response=True,
                              name="/predict/batch") as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")


@events.quitting.add_listener
def _assert_slos(environment, **_kw):
    """Fail the run (non-zero exit) when the p99/error-rate SLOs are violated,
    so CI cannot silently pass a degraded service. Thresholds mirror
    config/slo.yaml (local_api.p99_ms=300, error_rate_max=0.01)."""
    stats = environment.stats.total
    p99 = stats.get_response_time_percentile(0.99)
    fail_ratio = stats.fail_ratio
    if p99 is not None and p99 > 300:
        environment.process_exit_code = 1
        print(f"SLO VIOLATION: p99={p99}ms > 300ms")
    if fail_ratio > 0.01:
        environment.process_exit_code = 1
        print(f"SLO VIOLATION: error_rate={fail_ratio:.3%} > 1%")
