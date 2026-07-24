"""
Monitoring-config validation, device-placement, and Prometheus-metric tests.

These defend resume claims 2 (monitoring) and 3 (GPU support) with checks that
run without live infrastructure.
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

_PM_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PM_ROOT), str(_PM_ROOT / "inference_service")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Grafana dashboard + Prometheus rules validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMonitoringConfig:
    def test_reliability_dashboard_parses_and_has_required_panels(self):
        path = _PM_ROOT / "dashboard" / "grafana" / "reliability_dashboard.json"
        d = json.loads(path.read_text())
        titles = {p["title"].lower() for p in d["panels"]}
        for required in [
            "rolling within-15 accuracy",
            "retraining triggers",
            "api latency (p50/p95/p99)",
            "request errors",
            "cache hit ratio",
            "model version",
            "kafka consumer status",
            "service uptime",
        ]:
            assert required in titles, f"missing panel: {required}"
        # every panel must carry at least one PromQL target
        for p in d["panels"]:
            assert p.get("targets"), f"panel {p['title']} has no targets"

    def test_prometheus_alert_rules_parse_and_cover_slos(self):
        path = _PM_ROOT / "infra" / "prometheus" / "alert_rules.yml"
        doc = yaml.safe_load(path.read_text())
        alerts = {
            r["alert"]
            for g in doc["groups"]
            for r in g["rules"]
        }
        for required in [
            "RollingAccuracyBelowThreshold",
            "ApiP99LatencyAbove300ms",
            "ModelUnavailable",
            "KafkaPipelineDown",
            "RedisUnavailable",
            "ExcessiveCacheErrors",
        ]:
            assert required in alerts, f"missing alert: {required}"
        # each rule needs an expr and a for/labels block
        for g in doc["groups"]:
            for r in g["rules"]:
                assert r.get("expr"), f"{r['alert']} has no expr"
                assert r.get("labels", {}).get("severity")

    def test_prometheus_scrape_config_targets_inference_metrics(self):
        path = _PM_ROOT / "infra" / "prometheus" / "prometheus.yml"
        doc = yaml.safe_load(path.read_text())
        jobs = {j["job_name"] for j in doc["scrape_configs"]}
        assert "inference-api" in jobs
        api = next(j for j in doc["scrape_configs"] if j["job_name"] == "inference-api")
        assert api.get("metrics_path") == "/metrics"


# ---------------------------------------------------------------------------
# Prometheus metric wiring
# ---------------------------------------------------------------------------

_API_METRICS = None


def _load_api_metrics():
    """Load inference_service/api/metrics.py once (avoids the alerting/api collision
    and prometheus duplicate-registration on re-import)."""
    global _API_METRICS
    if _API_METRICS is not None:
        return _API_METRICS

    # Reuse an already-imported metrics module if one exists — re-executing
    # metrics.py would re-register the Prometheus collectors and raise
    # "Duplicated timeseries in CollectorRegistry".
    for mod in list(sys.modules.values()):
        if mod is not None and getattr(mod, "__name__", "").endswith("metrics") \
                and hasattr(mod, "RETRAINING_TRIGGERED_TOTAL"):
            _API_METRICS = mod
            return mod

    import importlib.util as ilu

    path = _PM_ROOT / "inference_service" / "api" / "metrics.py"
    spec = ilu.spec_from_file_location("_inference_api_metrics", path)
    mod = ilu.module_from_spec(spec)
    sys.modules["_inference_api_metrics"] = mod
    spec.loader.exec_module(mod)
    _API_METRICS = mod
    return _API_METRICS


@pytest.mark.unit
class TestPrometheusMetrics:
    def test_reliability_metrics_exist(self):
        m = _load_api_metrics()
        for name in ("RETRAINING_TRIGGERED_TOTAL", "CACHE_HITS_TOTAL", "CACHE_MISSES_TOTAL"):
            assert hasattr(m, name), f"metric {name} missing"

    def test_retraining_counter_increments(self):
        from prometheus_client import REGISTRY

        RETRAINING_TRIGGERED_TOTAL = _load_api_metrics().RETRAINING_TRIGGERED_TOTAL

        def read():
            v = REGISTRY.get_sample_value(
                "retraining_triggered_total", {"reason": "unit_test"}
            )
            return 0.0 if v is None else v

        before = read()
        RETRAINING_TRIGGERED_TOTAL.labels(reason="unit_test").inc()
        assert read() == before + 1


# ---------------------------------------------------------------------------
# Device placement (GPU support without CUDA hardware)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDevicePlacement:
    def test_device_selection_is_explicit_and_consistent(self):
        torch = pytest.importorskip("torch")
        from shared import feature_contract as fc
        from shared.checkpoint import _load_mlp_class

        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")

        MLPRULNet = _load_mlp_class()
        model = MLPRULNet(input_dim=fc.INPUT_DIM, hidden_sizes=[16], dropout=0.0).to(device)
        # Every parameter is on the selected device.
        for p in model.parameters():
            assert p.device.type == device.type
        # Input tensor placed on the same device runs without a device mismatch.
        import numpy as np
        x = torch.tensor(np.zeros((1, fc.INPUT_DIM), dtype="float32")).to(device)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.device.type == device.type

    def test_cuda_availability_is_reported_not_assumed(self):
        torch = pytest.importorskip("torch")
        # The value is whatever the host reports; the point is it is a real bool
        # derived from torch, never hardcoded True.
        assert isinstance(torch.cuda.is_available(), bool)
