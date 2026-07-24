"""
CPU-vs-GPU inference benchmark for the PyTorch MLP RUL model.

Defends resume claim 3 ("GPU acceleration") HONESTLY: it does not assume GPU
helps. It measures model-compute latency and throughput on CPU and (when a
CUDA device is present) on GPU, across batch sizes 1/8/16/32, and reports the
crossover batch size where GPU begins to win — or states plainly that GPU was
unavailable so only CPU was measured.

For single-request (batch=1) inference the host<->device transfer and kernel
launch overhead typically dominate, so CPU is often faster; GPU tends to win
only once the batch is large enough to amortize that overhead. The service
therefore selects the device from evidence/config, not by assumption.

Outputs:
    artifacts/cpu_vs_gpu_report.json
    artifacts/cpu_vs_gpu_report.md

Usage:
    python benchmarks/benchmark_cpu_vs_gpu.py --n-runs 300
"""

import argparse
import json
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

# Reuse the (namespace-safe) model builder + sequence generator from the
# existing single-path latency benchmark rather than re-deriving them.
sys.path.insert(0, str(_PM_ROOT / "inference_service"))
import benchmark_latency as bl  # noqa: E402

import numpy as np  # noqa: E402

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(_PM_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _environment() -> Dict:
    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch": torch.__version__ if _TORCH else None,
        "cuda_available": bool(_TORCH and torch.cuda.is_available()),
        "gpu_name": None,
    }
    if env["cuda_available"]:
        try:
            env["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            env["gpu_name"] = "unknown CUDA device"
    return env


def _percentiles(xs: List[float]) -> Dict[str, float]:
    xs_sorted = sorted(xs)
    def pct(p):
        return xs_sorted[min(len(xs_sorted) - 1, int(round(p / 100 * (len(xs_sorted) - 1))))]
    return {
        "p50_ms": round(pct(50), 4),
        "p95_ms": round(pct(95), 4),
        "p99_ms": round(pct(99), 4),
        "mean_ms": round(statistics.mean(xs), 4),
        "max_ms": round(max(xs), 4),
        "throughput_rps": round(1000.0 / statistics.mean(xs), 1),
    }


def _bench_device(device_str: str, batch_sizes: List[int], n_warmup: int,
                  n_runs: int, input_dim: int) -> Dict:
    """Model-compute-only benchmark on a fixed device across batch sizes."""
    device = torch.device(device_str)
    model = bl.MLPRULNet(input_dim=input_dim, hidden_sizes=[256, 128, 64, 32],
                         dropout=0.2).to(device).eval()

    results: Dict[str, Dict] = {}
    rng = np.random.default_rng(7)
    for bs in batch_sizes:
        x_host = torch.from_numpy(
            rng.standard_normal((bs, input_dim)).astype("float32")
        )
        # warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                xb = x_host.to(device)
                _ = model(xb)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        compute_ms, h2d_ms, d2h_ms = [], [], []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                xb = x_host.to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                out = model(xb)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t2 = time.perf_counter()
                _ = out.cpu()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t3 = time.perf_counter()
                h2d_ms.append((t1 - t0) * 1000)
                compute_ms.append((t2 - t1) * 1000)
                d2h_ms.append((t3 - t2) * 1000)

        entry = {"model_compute": _percentiles(compute_ms)}
        entry["model_compute"]["throughput_samples_per_s"] = round(
            bs * 1000.0 / statistics.mean(compute_ms), 1
        )
        if device.type == "cuda":
            entry["host_to_device_ms_mean"] = round(statistics.mean(h2d_ms), 4)
            entry["device_to_host_ms_mean"] = round(statistics.mean(d2h_ms), 4)
        results[f"batch_{bs}"] = entry
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-runs", type=int, default=300)
    args = ap.parse_args()

    if not _TORCH:
        report = {"status": "skipped", "reason": "torch not installed",
                  "environment": _environment()}
        (_ARTIFACTS / "cpu_vs_gpu_report.json").write_text(json.dumps(report, indent=2))
        print("SKIPPED: torch not installed")
        return

    input_dim = bl._ENGINEERED_INPUT_DIM
    batch_sizes = [1, 8, 16, 32]
    env = _environment()

    report: Dict = {
        "benchmark": "cpu_vs_gpu_model_compute",
        "environment": env,
        "config": {"batch_sizes": batch_sizes, "n_warmup": args.n_warmup,
                   "n_runs": args.n_runs, "input_dim": input_dim,
                   "measures": "model compute only (forward pass); "
                               "h2d/d2h transfer reported for GPU"},
        "devices": {},
    }

    print("Benchmarking CPU ...")
    report["devices"]["cpu"] = _bench_device("cpu", batch_sizes, args.n_warmup,
                                             args.n_runs, input_dim)

    if env["cuda_available"]:
        print("Benchmarking GPU ...")
        report["devices"]["cuda"] = _bench_device("cuda", batch_sizes, args.n_warmup,
                                                  args.n_runs, input_dim)
    else:
        report["devices"]["cuda"] = {
            "status": "skipped",
            "reason": "No CUDA device available on this host — GPU path NOT "
                      "measured. GPU acceleration claim is unverified here and "
                      "must be measured on a CUDA host before being asserted.",
        }

    # --- Honest conclusion, derived from data ---
    conclusion = _derive_conclusion(report)
    report["conclusion"] = conclusion

    (_ARTIFACTS / "cpu_vs_gpu_report.json").write_text(json.dumps(report, indent=2))
    (_ARTIFACTS / "cpu_vs_gpu_report.md").write_text(_render_md(report))
    print("\n" + conclusion["summary"])
    print(f"\nWrote {_ARTIFACTS/'cpu_vs_gpu_report.json'}")
    print(f"Wrote {_ARTIFACTS/'cpu_vs_gpu_report.md'}")


def _derive_conclusion(report: Dict) -> Dict:
    cpu = report["devices"]["cpu"]
    gpu = report["devices"].get("cuda", {})
    if gpu.get("status") == "skipped":
        return {
            "gpu_measured": False,
            "crossover_batch": None,
            "summary": (
                "GPU was UNAVAILABLE on this host, so only CPU was measured. "
                "The single-request CPU p99 is well under the 300 ms SLO "
                f"(batch_1 p99 = {cpu['batch_1']['model_compute']['p99_ms']} ms), "
                "so the low-latency path is served on CPU. The 'GPU acceleration' "
                "claim remains a code-supported capability that is UNVERIFIED on "
                "hardware here — it must be benchmarked on a CUDA host to assert a "
                "throughput benefit."
            ),
        }
    # Both measured — find crossover where GPU compute mean < CPU compute mean.
    crossover = None
    for bs in [1, 8, 16, 32]:
        c = cpu[f"batch_{bs}"]["model_compute"]["mean_ms"]
        g = gpu[f"batch_{bs}"]["model_compute"]["mean_ms"]
        if g < c:
            crossover = bs
            break
    if crossover is None:
        summary = ("CPU was faster than GPU at every tested batch size for this "
                   "small MLP — transfer + launch overhead dominates. The service "
                   "uses CPU for the low-latency path.")
    elif crossover == 1:
        summary = "GPU was faster even at batch=1; GPU acceleration benefits all tested workloads."
    else:
        summary = (f"GPU became faster than CPU at batch size {crossover}; below that, "
                   "CPU wins because host<->device transfer dominates. GPU acceleration "
                   "improved throughput for larger batches only.")
    return {"gpu_measured": True, "crossover_batch": crossover, "summary": summary}


def _render_md(report: Dict) -> str:
    env = report["environment"]
    lines = [
        "# CPU vs GPU — Model-Compute Benchmark", "",
        f"- Timestamp: {env['timestamp']}",
        f"- Git commit: {env['git_commit']}",
        f"- Host: {env['platform']} / {env['processor']}",
        f"- torch: {env['torch']}  |  CUDA available: {env['cuda_available']}"
        f"  |  GPU: {env['gpu_name']}",
        f"- Measures: {report['config']['measures']}",
        "",
        "## Results (model compute)", "",
        "| device | batch | p50 ms | p95 ms | p99 ms | throughput (samples/s) |",
        "|---|---|---|---|---|---|",
    ]
    for dev in ("cpu", "cuda"):
        d = report["devices"].get(dev, {})
        if d.get("status") == "skipped":
            lines.append(f"| {dev} | — | — | — | — | SKIPPED: {d['reason'][:60]}… |")
            continue
        for bs in [1, 8, 16, 32]:
            mc = d[f"batch_{bs}"]["model_compute"]
            lines.append(f"| {dev} | {bs} | {mc['p50_ms']} | {mc['p95_ms']} | "
                         f"{mc['p99_ms']} | {mc['throughput_samples_per_s']} |")
    lines += ["", "## Conclusion", "", report["conclusion"]["summary"], ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
