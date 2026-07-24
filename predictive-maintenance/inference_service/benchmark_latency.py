"""
Latency Benchmark — PyTorch MLP Inference (post-feature-engineering fix)

Measures end-to-end prediction latency for the full serving path:

    sequence (50 readings)
        → OnlineFeatureEngineer.engineer()   [lag/rolling/EMA features]
        → InferenceEngine.predict_rul()      [MLP forward pass, GPU if available]
        → get_health_status_from_rul()

This matches the ACTUAL serving path after the training-serving feature
consistency fix.  The old benchmark used preprocess_flat_features(sequence[-1])
which skipped all temporal feature engineering and understated serving cost.

Does NOT include Redis cache or HTTP overhead — pure inference latency.
Cache-hit path is dominated by Redis RTT (~1 ms) and JSON deserialization.

Usage:
    python benchmark_latency.py
    python benchmark_latency.py --n-warmup 20 --n-runs 500 --seq-len 50

Results are saved to benchmark_results.json for reproducibility.
"""

import argparse
import json
import logging
import sys
import time
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent                  # .../inference_service
_PREDICTIVE = _HERE.parent                              # .../predictive-maintenance
_TRAIN = _PREDICTIVE / "ml_pipeline" / "train"
# Add train path first — gives 'models.mlp_model' priority over
# inference_service/models/ package, avoiding namespace collision
sys.path.insert(0, str(_TRAIN))
# ...and the package root, so `shared.feature_contract` (the canonical feature
# contract shared by training and serving) resolves regardless of cwd.
sys.path.insert(0, str(_PREDICTIVE))

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

import importlib.util as _ilu

# Load inference_engine directly to avoid namespace collision with
# inference_service/models/ package shadowing train/models/ package
_ie_spec = _ilu.spec_from_file_location(
    "inference_engine", _HERE / "models" / "inference_engine.py"
)
_ie_mod = _ilu.module_from_spec(_ie_spec)
_ie_spec.loader.exec_module(_ie_mod)
InferenceEngine = _ie_mod.InferenceEngine

# Load feature_engineering from inference_service (not the train package)
_fe_spec = _ilu.spec_from_file_location(
    "feature_engineering", _HERE / "feature_engineering.py"
)
_fe_mod = _ilu.module_from_spec(_fe_spec)
_fe_spec.loader.exec_module(_fe_mod)
OnlineFeatureEngineer = _fe_mod.OnlineFeatureEngineer
CONSTANT_SENSORS = _fe_mod.CONSTANT_SENSORS

# Load MLPRULNet directly to avoid inference_service/models/ shadowing
# ml_pipeline/train/models/ — same pattern already used for inference_engine above.
_mlp_spec = _ilu.spec_from_file_location(
    "mlp_model", _TRAIN / "models" / "mlp_model.py"
)
_mlp_mod = _ilu.module_from_spec(_mlp_spec)
_mlp_spec.loader.exec_module(_mlp_mod)
MLPRULNet = _mlp_mod.MLPRULNet


# ---------------------------------------------------------------------------
# C-MAPSS FD001 sensor names (all 21; 7 constant ones are excluded by
# OnlineFeatureEngineer automatically)
# ---------------------------------------------------------------------------
_ALL_SENSORS = [f"sensor_{i}" for i in range(1, 22)]

# Non-constant sensors after CONSTANT_SENSORS exclusion: 14 sensors
# Features per sensor: 1 raw + 4 lag + 6 rolling(mean+std×3) + 3 EMA = 14
# Total: 14 × 14 + 3 op_settings + 1 time_cycle = 200
_NON_CONSTANT_SENSORS = [s for s in _ALL_SENSORS if s not in CONSTANT_SENSORS]
_ENGINEERED_INPUT_DIM = len(_NON_CONSTANT_SENSORS) * 14 + 3 + 1  # = 200


# ---------------------------------------------------------------------------
# Build a minimal MLP for benchmarking (no checkpoint needed)
# ---------------------------------------------------------------------------

def _build_benchmark_model(input_dim: int = _ENGINEERED_INPUT_DIM) -> "torch.nn.Module":
    """
    Build an MLPRULNet with production architecture for latency measurement.
    Uses random weights — only latency matters here, not prediction quality.
    input_dim must match the OnlineFeatureEngineer output dimension (200).
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is required for this benchmark")
    model = MLPRULNet(
        input_dim=input_dim,
        hidden_sizes=[256, 128, 64, 32],
        dropout=0.2,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    model._device = device
    return model


def _make_sequence(seq_len: int, rng: np.random.Generator) -> List[Dict[str, float]]:
    """
    Build a realistic C-MAPSS-style sensor sequence of length seq_len.

    All 21 sensors are included (OnlineFeatureEngineer drops the 7 constant
    ones internally). Values drift slightly over time to exercise the temporal
    feature computation (lags, rolling stats, EMA).
    """
    base = {s: rng.uniform(0.5, 1.5) for s in _ALL_SENSORS}
    base.update({
        "op_setting_1": rng.uniform(0.0, 0.5),
        "op_setting_2": rng.uniform(0.0, 0.5),
        "op_setting_3": rng.uniform(60.0, 100.0),
    })
    sequence = []
    for t in range(seq_len):
        reading = {k: v + rng.normal(0, 0.01) for k, v in base.items()}
        reading["time_cycle"] = float(t + 1)
        sequence.append(reading)
        # Gradual degradation drift on non-constant sensors
        for s in _NON_CONSTANT_SENSORS:
            base[s] += rng.uniform(0.0, 0.005)
    return sequence


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    n_warmup: int = 10,
    n_runs: int = 200,
    seq_len: int = 50,
) -> Dict:
    """
    Run end-to-end inference latency benchmark using the full serving path:

        sequence → OnlineFeatureEngineer.engineer()
                 → InferenceEngine.predict_rul()
                 → get_health_status_from_rul()

    Returns dict with p50, p95, p99, mean, min, max latencies in milliseconds.
    """
    rng = np.random.default_rng(42)
    feat_eng = OnlineFeatureEngineer()
    engine = InferenceEngine(n_features=_ENGINEERED_INPUT_DIM)

    # Probe actual output dim from the feature engineer so model matches
    probe_seq = _make_sequence(seq_len, rng)
    probe_features = feat_eng.engineer(probe_seq, "probe")
    actual_input_dim = probe_features.shape[1]

    model = _build_benchmark_model(input_dim=actual_input_dim)

    # ------------------------------------------------------------------
    # GPU-acceleration evidence — explicit, unambiguous fields so the
    # benchmark output can be cited as proof of *runtime* device placement,
    # not just that CUDA code paths exist in the source.
    # ------------------------------------------------------------------
    cuda_available = bool(torch.cuda.is_available())
    device = torch.device("cuda" if cuda_available else "cpu")
    device_label = str(device)
    run_type = "cuda" if cuda_available else "cpu"
    gpu_name: Optional[str] = None
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(device)
        except Exception:
            gpu_name = "unknown CUDA device"
    else:
        logger.info("CUDA unavailable, running on CPU fallback.")

    # Pre-generate all sequences so generation cost is excluded from timing
    sequences = [_make_sequence(seq_len, rng) for _ in range(n_warmup + n_runs)]

    print(f"\nBenchmark config:")
    print(f"  torch.cuda.is_available() : {cuda_available}")
    print(f"  Selected device           : {device_label}  (run_type={run_type})")
    if gpu_name:
        print(f"  GPU name                  : {gpu_name}")
    elif not cuda_available:
        print(f"  GPU name                  : n/a — CUDA unavailable, running on CPU fallback.")
    print(f"  Model params      : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Sequence length   : {seq_len} readings")
    print(f"  Engineered dim    : {actual_input_dim} features")
    print(f"  Non-const sensors : {len(_NON_CONSTANT_SENSORS)}")
    print(f"  Warmup runs       : {n_warmup}")
    print(f"  Timed runs        : {n_runs}")
    print(f"  Timing path       : sequence → engineer() → predict_rul() → health_status")
    print()

    # Warmup — JIT trace / CUDA warm
    print("Warming up ...", end="", flush=True)
    for i in range(n_warmup):
        features = feat_eng.engineer(sequences[i], "warmup")
        engine.predict_rul(model, features, return_confidence=False)
    print(" done")

    # Timed runs — measure the full serving path with split timings.
    # preprocess_ms: OnlineFeatureEngineer.engineer() — lag/rolling/EMA over sequence
    # model_ms:      MLP forward pass + health_status derivation
    # total_ms:      preprocess_ms + model_ms (end-to-end wall time)
    preprocess_ms_list: List[float] = []
    model_ms_list: List[float] = []
    total_ms_list: List[float] = []
    print("Benchmarking ...", end="", flush=True)
    for i in range(n_runs):
        seq = sequences[n_warmup + i]
        eq_id = f"eq_{i % 10}"

        t0 = time.perf_counter()
        features = feat_eng.engineer(seq, eq_id)          # preprocessing
        t1 = time.perf_counter()
        rul, _ = engine.predict_rul(model, features, return_confidence=False)
        _ = engine.get_health_status_from_rul(rul)        # model + health
        t2 = time.perf_counter()

        preprocess_ms_list.append((t1 - t0) * 1000.0)
        model_ms_list.append((t2 - t1) * 1000.0)
        total_ms_list.append((t2 - t0) * 1000.0)
    print(" done\n")

    def _stats(values: List[float]) -> Dict:
        s = sorted(values)
        n = len(s)
        return {
            "mean":   round(statistics.mean(s), 3),
            "median": round(statistics.median(s), 3),
            "p50":    round(s[int(n * 0.50)], 3),
            "p95":    round(s[int(n * 0.95)], 3),
            "p99":    round(s[int(n * 0.99)], 3),
            "min":    round(min(s), 3),
            "max":    round(max(s), 3),
        }

    results = {
        "cuda_available": cuda_available,
        "device": device_label,
        "gpu_name": gpu_name,
        "run_type": run_type,
        "n_runs": n_runs,
        "n_warmup": n_warmup,
        "seq_len": seq_len,
        "engineered_input_dim": actual_input_dim,
        "model_params": sum(p.numel() for p in model.parameters()),
        "preprocessing": "OnlineFeatureEngineer (lag/rolling/EMA, ddof=1 std)",
        "total_ms":      _stats(total_ms_list),
        "preprocess_ms": _stats(preprocess_ms_list),
        "model_ms":      _stats(model_ms_list),
        "under_300ms": all(l < 300.0 for l in total_ms_list),
        "under_50ms":  all(l < 50.0  for l in total_ms_list),
        "under_10ms":  all(l < 10.0  for l in total_ms_list),
    }
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference latency benchmark")
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--n-runs",   type=int, default=200)
    p.add_argument("--seq-len",  type=int, default=50,
                   help="Sensor sequence length fed to OnlineFeatureEngineer")
    p.add_argument("--output",   default="benchmark_results.json")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = run_benchmark(
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        seq_len=args.seq_len,
    )

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    tot = results["total_ms"]
    pre = results["preprocess_ms"]
    mdl = results["model_ms"]
    print("=" * 70)
    print("  PyTorch MLP Inference Latency Benchmark")
    print("  path: sequence → OnlineFeatureEngineer → MLP → health_status")
    print("=" * 70)
    print(f"  torch.cuda.is_available() : {results['cuda_available']}")
    print(f"  Selected device           : {results['device']}  (run_type={results['run_type']})")
    if results.get("gpu_name"):
        print(f"  GPU name                  : {results['gpu_name']}")
    elif not results["cuda_available"]:
        print(f"  GPU name                  : n/a — CUDA unavailable, running on CPU fallback.")
    print(f"  Model params      : {results['model_params']:,}")
    print(f"  Sequence length   : {results['seq_len']} readings")
    print(f"  Engineered dim    : {results['engineered_input_dim']} features")
    print(f"  Runs (timed)      : {results['n_runs']}")
    print()
    print(f"  {'Metric':<10}  {'Total':>9}  {'Preprocess':>12}  {'Model':>9}")
    print(f"  {'-'*10}  {'-'*9}  {'-'*12}  {'-'*9}")
    for k in ("mean", "p50", "p95", "p99", "min", "max"):
        print(f"  {k:<10}  {tot[k]:>8.3f}ms  {pre[k]:>10.3f}ms  {mdl[k]:>8.3f}ms")
    print()
    print(f"  All runs < 300 ms : {results['under_300ms']}")
    print(f"  All runs <  50 ms : {results['under_50ms']}")
    print(f"  All runs <  10 ms : {results['under_10ms']}")
    print("=" * 70)
    print(f"\nResults saved to {out_path}")
