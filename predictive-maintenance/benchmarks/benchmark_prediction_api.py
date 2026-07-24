"""
In-process ASGI (local API) latency benchmark.

Drives the REAL FastAPI app object through Starlette's TestClient, so the
measured path includes: HTTP parsing, Pydantic validation, routing, the
prediction handler (feature engineering + MLP forward), and JSON
serialization. It excludes ONLY the kernel TCP socket and any network hop.

Why in-process: this repository's sandbox blocks loopback TCP, so a bound-port
Locust/HTTP run cannot execute here. This benchmark is therefore labeled
`transport: "in_process_asgi"` and is explicitly NOT a deployed-endpoint or
bound-socket number. Those remain unverified (see docs/limitations.md).

Redis is typically unavailable here too, so this measures the cache-MISS path
(full inference every call) unless a real Redis is reachable.

Output: artifacts/local_api_latency.json

Usage:
    MLFLOW_TRACKING_URI=file:./mlruns python benchmarks/benchmark_prediction_api.py --n-runs 300
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
_PM_ROOT = _HERE.parent
_ARTIFACTS = _PM_ROOT / "artifacts"
_ARTIFACTS.mkdir(exist_ok=True)
sys.path.insert(0, str(_PM_ROOT))

os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")

_SENSORS = [f"sensor_{i}" for i in range(1, 22)]


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(_PM_ROOT), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _sequence(n=50):
    seq = []
    for t in range(n):
        r = {s: 1.0 + 0.01 * t for s in _SENSORS}
        r.update({"op_setting_1": 0.1, "op_setting_2": 0.2, "op_setting_3": 80.0,
                  "time_cycle": float(t + 1)})
        seq.append(r)
    return seq


def _pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def _stats(xs):
    return {"p50_ms": round(_pct(xs, 50), 3), "p95_ms": round(_pct(xs, 95), 3),
            "p99_ms": round(_pct(xs, 99), 3), "mean_ms": round(statistics.mean(xs), 3),
            "max_ms": round(max(xs), 3), "min_ms": round(min(xs), 3), "count": len(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-runs", type=int, default=300)
    args = ap.parse_args()

    os.chdir(_PM_ROOT)  # config + checkpoint paths are cwd-relative
    from fastapi.testclient import TestClient
    from inference_service.api.main import app, limiter
    # Latency benchmark: disable the per-IP rate limiter so it does not cap the
    # request rate. Rate-limit behavior is validated separately by the load
    # tests; here we want raw served-request latency.
    limiter.enabled = False

    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "transport": "in_process_asgi",
        "note": "Starlette TestClient — full ASGI stack minus kernel TCP socket. "
                "NOT a bound-socket or deployed-endpoint measurement.",
    }

    payload = {"data": {"equipment_id": "engine_001", "sequence": _sequence()},
               "return_confidence": False}
    batch_payload = {"sequences": [{"equipment_id": f"engine_{i:03d}",
                                    "sequence": _sequence()} for i in range(8)]}

    report: Dict = {"benchmark": "local_api_latency", "environment": env,
                    "config": {"n_warmup": args.n_warmup, "n_runs": args.n_runs}}

    with TestClient(app) as client:
        # sanity — confirm model is actually loaded and predictions are 200
        probe = client.post("/predict/rul", json=payload)
        if probe.status_code != 200:
            report["status"] = "skipped"
            report["reason"] = (f"/predict/rul returned {probe.status_code}: "
                                f"{probe.text[:200]} — model likely not loaded")
            (_ARTIFACTS / "local_api_latency.json").write_text(json.dumps(report, indent=2))
            print("SKIPPED:", report["reason"])
            return

        for _ in range(args.n_warmup):
            client.post("/predict/rul", json=payload)

        single_ms: List[float] = []
        for _ in range(args.n_runs):
            t0 = time.perf_counter()
            resp = client.post("/predict/rul", json=payload)
            single_ms.append((time.perf_counter() - t0) * 1000)
            assert resp.status_code == 200

        batch_ms: List[float] = []
        for _ in range(max(1, args.n_runs // 5)):
            t0 = time.perf_counter()
            resp = client.post("/predict/batch", json=batch_payload)
            batch_ms.append((time.perf_counter() - t0) * 1000)
            assert resp.status_code == 200

    report["status"] = "ok"
    report["single_request"] = _stats(single_ms)
    report["batch_8_request"] = _stats(batch_ms)
    report["slo_p99_ms"] = 300
    report["slo_pass"] = _pct(single_ms, 99) < 300
    (_ARTIFACTS / "local_api_latency.json").write_text(json.dumps(report, indent=2))

    print(f"local API (in-process ASGI) single  p50={report['single_request']['p50_ms']}ms "
          f"p99={report['single_request']['p99_ms']}ms  SLO<300ms: {report['slo_pass']}")
    print(f"local API batch-8 p99={report['batch_8_request']['p99_ms']}ms")
    print(f"Wrote {_ARTIFACTS/'local_api_latency.json'}")
    if not report["slo_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
