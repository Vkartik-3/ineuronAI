"""
Regression guards for the hardening pass:

1. The FastAPI app must IMPORT cleanly. slowapi raises at import time if a
   rate-limited endpoint lacks a parameter named `request`; this test would have
   caught the boot bug that shipped in the uncommitted tree (endpoints named the
   param `req`). Defends resume claim 3 ("deploying a FastAPI service") — a
   service that cannot import cannot deploy.

2. The new benchmark modules must import (syntax/contract smoke).

3. metrics.py must be import-idempotent (no Duplicated-timeseries crash), which
   also keeps the wider test suite order-independent.
"""
import importlib
import sys
from pathlib import Path

import pytest

_PM_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PM_ROOT), str(_PM_ROOT / "inference_service")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.mark.unit
def test_inference_app_imports_and_has_rate_limited_routes():
    """App import must not raise (slowapi param-name contract)."""
    main = importlib.import_module("inference_service.api.main")
    routes = {r.path for r in main.app.routes}
    for path in ("/predict/rul", "/predict/batch", "/feedback", "/models/reload"):
        assert path in routes, f"expected route {path} registered"


@pytest.mark.unit
def test_metrics_module_import_is_idempotent():
    """Re-executing metrics.py must not raise Duplicated timeseries."""
    import inference_service.api.metrics as m
    importlib.reload(m)  # would raise ValueError before the _get_or_create fix
    assert hasattr(m, "INFERENCE_REQUESTS_TOTAL")


@pytest.mark.unit
@pytest.mark.parametrize("mod_path", [
    "benchmarks/benchmark_cpu_vs_gpu.py",
    "benchmarks/benchmark_cache.py",
    "benchmarks/benchmark_prediction_api.py",
])
def test_benchmark_scripts_are_importable(mod_path):
    """Compile the benchmark scripts (catches syntax/contract breakage)."""
    src = (_PM_ROOT / mod_path).read_text()
    compile(src, mod_path, "exec")
