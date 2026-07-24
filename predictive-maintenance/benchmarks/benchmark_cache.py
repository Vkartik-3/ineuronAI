"""
Redis prediction-cache benchmark.

Measures cache-hit vs cache-miss(+set) latency for the model-aware prediction
cache that backs resume claim 3 ("Redis caching"). Requires a REAL Redis
server. If none is reachable, the benchmark records status="skipped" with the
reason and exits 0 — it NEVER substitutes model-compute latency for a cache
number (that substitution is explicitly forbidden by the project rules).

Connect target: REDIS_HOST/REDIS_PORT env (default localhost:6379).

Outputs:
    artifacts/cache_latency.json

Usage:
    REDIS_HOST=localhost python benchmarks/benchmark_cache.py --n-runs 500
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
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_PM_ROOT = _HERE.parent
_ARTIFACTS = _PM_ROOT / "artifacts"
_ARTIFACTS.mkdir(exist_ok=True)
sys.path.insert(0, str(_PM_ROOT))


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(_PM_ROOT),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _environment() -> Dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "redis_host": os.environ.get("REDIS_HOST", "localhost"),
        "redis_port": int(os.environ.get("REDIS_PORT", 6379)),
    }


def _pct(xs: List[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def _stats(xs: List[float]) -> Dict[str, float]:
    return {
        "p50_ms": round(_pct(xs, 50), 4),
        "p95_ms": round(_pct(xs, 95), 4),
        "p99_ms": round(_pct(xs, 99), 4),
        "mean_ms": round(statistics.mean(xs), 4),
        "max_ms": round(max(xs), 4),
        "count": len(xs),
    }


def _skip(reason: str, env: Dict) -> None:
    report = {"benchmark": "cache_latency", "status": "skipped",
              "reason": reason, "environment": env}
    (_ARTIFACTS / "cache_latency.json").write_text(json.dumps(report, indent=2))
    print(f"SKIPPED (recorded, not faked): {reason}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=500)
    args = ap.parse_args()
    env = _environment()

    try:
        from inference_service.cache.prediction_cache import PredictionCache
    except Exception as exc:
        _skip(f"could not import PredictionCache: {exc}", env)
        return

    cache = PredictionCache(host=env["redis_host"], port=env["redis_port"], ttl=300)
    if not cache.is_available:
        _skip(
            f"no reachable Redis server at {env['redis_host']}:{env['redis_port']} "
            "— cache latency NOT measured (not substituted with any other number)",
            env,
        )
        return

    sample_value = {
        "equipment_id": "engine_001", "rul_cycles": 87.5, "rul_hours": 43.75,
        "health_status": "warning", "model_version": "v1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    key = cache.make_key("engine_001", [{"sensor_2": 1.0}], model_name="mlp",
                         model_version="v1.0.0")

    # Prime one entry, then measure hit latency.
    cache.set(key, sample_value)
    hit_ms: List[float] = []
    for _ in range(args.n_runs):
        t0 = time.perf_counter()
        _ = cache.get(key)
        hit_ms.append((time.perf_counter() - t0) * 1000)

    # Miss latency: unique key each time (get -> None), then set.
    miss_get_ms: List[float] = []
    miss_set_ms: List[float] = []
    for i in range(args.n_runs):
        k = cache.make_key(f"engine_miss_{i}", [{"sensor_2": float(i)}],
                           model_name="mlp", model_version="v1.0.0")
        t0 = time.perf_counter()
        got = cache.get(k)
        t1 = time.perf_counter()
        cache.set(k, sample_value)
        t2 = time.perf_counter()
        assert got is None
        miss_get_ms.append((t1 - t0) * 1000)
        miss_set_ms.append((t2 - t1) * 1000)

    hit_ratio = 1.0  # by construction in the hit loop; real hit ratio is workload-dependent
    report = {
        "benchmark": "cache_latency",
        "status": "ok",
        "environment": env,
        "config": {"n_runs": args.n_runs, "ttl_seconds": cache.ttl},
        "cache_hit": _stats(hit_ms),
        "cache_miss_get": _stats(miss_get_ms),
        "cache_miss_set": _stats(miss_set_ms),
        "note": ("hit_ratio here is 1.0 by construction of the hit loop; the "
                 "production hit ratio is workload-dependent and is measured "
                 "separately by the load tests via Prometheus cache counters."),
        "slo_p99_hit_ms": 50,
        "slo_pass": _pct(hit_ms, 99) < 50,
    }
    (_ARTIFACTS / "cache_latency.json").write_text(json.dumps(report, indent=2))
    # cleanup miss keys
    for i in range(args.n_runs):
        cache.delete(cache.make_key(f"engine_miss_{i}", [{"sensor_2": float(i)}],
                                    model_name="mlp", model_version="v1.0.0"))
    cache.delete(key)

    print(f"cache hit  p99 = {report['cache_hit']['p99_ms']} ms  (SLO<50ms: {report['slo_pass']})")
    print(f"cache miss get p99 = {report['cache_miss_get']['p99_ms']} ms")
    print(f"Wrote {_ARTIFACTS/'cache_latency.json'}")
    if not report["slo_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
