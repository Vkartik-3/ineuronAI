"""
Full HTTP end-to-end latency benchmark (distinct from the local compute benchmark).

Measures the COMPLETE serving path against a RUNNING FastAPI service:
HTTP parse -> Pydantic validation -> API-key auth -> rate limiting -> Redis
lookup/write -> OnlineFeatureEngineer -> scaler.transform -> MLP forward ->
health derivation -> JSON serialization.

This is deliberately separate from benchmark_latency.py (the local compute
benchmark), which excludes HTTP/Redis and measures only in-process inference.

Prereqs (not launched by this script):
    # 1. Redis
    docker compose -f infra/monitoring/docker-compose.redis.yml up -d
    # 2. inference service with the canonical checkpoint loaded
    API_KEY=devkey uvicorn inference_service.api.main:app --port 8000
    # 3. run:
    .venv/bin/python inference_service/benchmark_http.py \
        --url http://localhost:8000 --api-key devkey \
        --scenarios cache_miss cache_hit --concurrency 1 10 50 \
        --output benchmark_http_results.json

Exits nonzero if any measured scenario's p99 exceeds 300 ms (the SLO).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # package root

try:
    import requests
except ImportError:
    requests = None

from shared.feature_contract import OP_SETTINGS, SENSORS  # noqa: E402


def _reading(i: int) -> dict:
    r = {"time_cycle": i + 1}
    for j, s in enumerate(SENSORS):
        r[s] = 100.0 + j + 0.1 * i
    for o in OP_SETTINGS:
        r[o] = 0.5
    return r


def _payload(equipment_id: str, seq_len: int = 50) -> dict:
    return {
        "equipment_id": equipment_id,
        "sequence": [_reading(i) for i in range(seq_len)],
    }


def _percentiles(xs):
    xs = sorted(xs)
    def pct(p):
        if not xs:
            return None
        k = min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))
        return xs[k] * 1000.0  # ms
    return {
        "p50": pct(50), "p95": pct(95), "p99": pct(99),
        "max": max(xs) * 1000.0 if xs else None,
    }


def run_scenario(url, api_key, name, n_requests, concurrency, warmup, same_equipment):
    endpoint = f"{url}/predict/rul"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    def one(i):
        eq = "EQ-FIXED" if same_equipment else f"EQ-{i}"
        t0 = time.perf_counter()
        try:
            resp = requests.post(endpoint, json=_payload(eq), headers=headers, timeout=10)
            ok = resp.status_code == 200
        except Exception:
            ok = False
        return time.perf_counter() - t0, ok

    for i in range(warmup):
        one(i)

    latencies, errors = [], 0
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for dt, ok in ex.map(one, range(n_requests)):
            latencies.append(dt)
            errors += 0 if ok else 1
    wall = time.perf_counter() - t_start

    pcts = _percentiles(latencies)
    return {
        "scenario": name,
        "concurrency": concurrency,
        "n_requests": n_requests,
        "warmup": warmup,
        "errors": errors,
        "error_rate": errors / max(1, n_requests),
        "rps": n_requests / wall if wall else None,
        **pcts,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--api-key", default="devkey")
    ap.add_argument("--n-requests", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 10, 50])
    ap.add_argument("--scenarios", nargs="+",
                    default=["cache_miss", "cache_hit"],
                    help="cache_miss=unique equipment, cache_hit=fixed equipment")
    ap.add_argument("--output", default="benchmark_http_results.json")
    ap.add_argument("--slo-p99-ms", type=float, default=300.0)
    args = ap.parse_args()

    if requests is None:
        print("FATAL: `requests` not installed (pip install requests).", file=sys.stderr)
        return 2

    env = {
        "machine": platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": __import__("os").cpu_count(),
    }
    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        env["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
        )
    except Exception:
        env["cuda_available"] = None

    results, slo_violations = [], []
    for scenario in args.scenarios:
        same_eq = scenario == "cache_hit"
        for c in args.concurrency:
            r = run_scenario(
                args.url, args.api_key, scenario, args.n_requests, c,
                args.warmup, same_eq,
            )
            results.append(r)
            print(f"  {scenario:<12} c={c:<3} p50={r['p50']:.1f} p95={r['p95']:.1f} "
                  f"p99={r['p99']:.1f} rps={r['rps']:.0f} err={r['errors']}")
            if r["p99"] is not None and r["p99"] > args.slo_p99_ms:
                slo_violations.append(r)

    out = {
        "benchmark_type": "full_http_end_to_end",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "slo_p99_ms": args.slo_p99_ms,
        "results": results,
        "slo_violations": len(slo_violations),
    }
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved {args.output}. SLO violations: {len(slo_violations)}")
    return 1 if slo_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
