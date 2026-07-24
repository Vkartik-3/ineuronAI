"""
Controlled retraining event as a pytest integration test.

Deterministic, self-contained (temp local MLflow, in-memory monitor). Proves
the below-80% trigger drives the full retrain -> challenger -> checkpoint ->
compare -> promote -> counter cycle.

Run:
    .venv/bin/python -m pytest tests/integration/test_controlled_retraining_event.py -q
"""

import sys
from pathlib import Path

import pytest

_PM_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PM_ROOT), str(_PM_ROOT / "ml_pipeline" / "retrain")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("torch")
pytest.importorskip("mlflow")


@pytest.mark.integration
def test_controlled_retraining_event_full_cycle():
    from controlled_retraining_event import run_event

    ev = run_event()

    assert ev["below_threshold"] is True
    assert ev["measured_accuracy"] < ev["threshold"]
    assert ev["decision_state"] == "degraded"
    assert ev["retraining_triggered"] is True
    assert ev["challenger_mlflow_run_id"]
    assert ev["checkpoint_sha256"] and len(ev["checkpoint_sha256"]) == 64
    assert ev["comparison_winner"] == "challenger"
    assert ev["comparison_improvement_pct"] >= 5.0
    assert ev["comparison_ci"]["challenger_significantly_better"] is True
    assert ev["promotion_approved"] is True
    assert ev["prometheus_counter_after"] == (ev["prometheus_counter_before"] or 0) + 1
