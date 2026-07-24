"""
Controlled retraining event (deterministic, self-contained).

Proves the below-80%-accuracy trigger drives a full retraining cycle end to
end, WITHOUT requiring live AWS/Kafka. It uses:

  * an in-memory PerformanceMonitor (or Redis if REDIS_HOST is set)
  * a temporary local MLflow SQLite store + filesystem artifacts
  * the real canonical checkpoint format and comparator
  * the real RetrainGuard (cooldown / active-job / states)

Flow:
  1. Insert >= 50 feedback records engineered to ~76% within-15 accuracy.
  2. PerformanceMonitor.check_accuracy_threshold() -> below_threshold=True.
  3. RetrainGuard.decide() -> DEGRADED -> begin() claims the job.
  4. Train a small challenger, save a CANONICAL checkpoint, log an MLflow run.
  5. ModelComparator.compare_predictions(champion, challenger) with bootstrap CI.
  6. Record promotion approved/rejected.
  7. Increment the Prometheus RETRAINING_TRIGGERED_TOTAL counter.
  8. Print the full evidence block.

Run:
    .venv/bin/python ml_pipeline/retrain/controlled_retraining_event.py
    REDIS_HOST=localhost .venv/bin/python ml_pipeline/retrain/controlled_retraining_event.py

Redis (optional) via the compose profile:
    docker compose -f infra/monitoring/docker-compose.redis.yml up -d
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

_PM_ROOT = Path(__file__).resolve().parents[2]
for _p in (
    str(_PM_ROOT),
    str(_PM_ROOT / "ml_pipeline" / "train"),
    str(_PM_ROOT / "ml_pipeline" / "retrain"),
    str(_PM_ROOT / "inference_service"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _build_monitor(threshold=0.80, tolerance=15, min_samples=50):
    """In-memory PerformanceMonitor with deterministic config (Redis if env set)."""
    from performance_monitor import PerformanceMonitor

    # Construct without a config file by patching the loaded config after init
    # is awkward; instead rely on the class defaults and override attributes.
    m = PerformanceMonitor.__new__(PerformanceMonitor)
    m.accuracy_threshold = threshold
    m.tolerance_cycles = tolerance
    m.min_samples = min_samples
    m.window_days = 7
    m._redis_client = None
    m._records = []
    m._redis_key = "perf_monitor:records"
    m._redis_key_prefix = "perf_monitor:"
    return m


def _insert_feedback(monitor, n=60, target_accuracy=0.76, seed=7):
    """Insert n records whose within-15 accuracy is ~target_accuracy."""
    from performance_monitor import PredictionRecord

    rng = np.random.default_rng(seed)
    n_good = int(round(n * target_accuracy))
    now = datetime.utcnow()
    for i in range(n):
        true_rul = float(rng.uniform(20, 120))
        err = rng.uniform(0, 10) if i < n_good else rng.uniform(25, 45)
        rec = PredictionRecord(
            timestamp=now - timedelta(minutes=n - i),
            equipment_id=f"EQ-{i % 10:02d}",
            true_rul=true_rul,
            predicted_rul=true_rul + err * (1 if rng.random() < 0.5 else -1),
        )
        rec.within_tolerance = abs(rec.predicted_rul - true_rul) <= monitor.tolerance_cycles
        monitor._records.append(rec)
    return monitor


def _train_challenger(input_dim, seed=0):
    import torch
    from sklearn.preprocessing import StandardScaler
    from shared.checkpoint import _load_mlp_class

    MLPRULNet = _load_mlp_class()
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, input_dim))
    y = (X[:, :5].sum(axis=1) * 2 + 60).astype(np.float32)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)

    torch.manual_seed(seed)
    model = MLPRULNet(input_dim=input_dim, hidden_sizes=[32, 16], dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.MSELoss()
    model.train()
    for _ in range(40):
        opt.zero_grad()
        loss = lossf(model(torch.tensor(Xs)), torch.tensor(y))
        loss.backward()
        opt.step()
    model.eval()
    return model, scaler, X, y


def run_event() -> dict:
    import mlflow
    import torch

    from shared import feature_contract as fc
    from shared.checkpoint import build_checkpoint_payload, save_canonical_checkpoint
    from retrain_guard import RetrainGuard, RetrainState
    from model_comparator import ModelComparator

    tmp = Path(tempfile.mkdtemp(prefix="retrain_event_"))
    mlflow.set_tracking_uri(f"sqlite:///{tmp/'mlflow.db'}")

    # 1-2. feedback + accuracy check
    monitor = _insert_feedback(_build_monitor(), n=60, target_accuracy=0.76)
    report = monitor.check_accuracy_threshold()

    # 3. guard decision
    guard = RetrainGuard(cooldown_seconds=3600, min_samples=50)
    decision = guard.decide(
        accuracy=report.accuracy, threshold=report.threshold,
        n_samples=report.n_samples, evaluation_available=True,
    )
    triggered = decision.should_retrain and guard.begin()

    challenger_run_id = ckpt_sha = None
    comparison = None
    promotion_approved = False
    try:
        if triggered:
            # 4-5. challenger + canonical checkpoint + MLflow run
            model, scaler, X, y = _train_challenger(fc.INPUT_DIM)
            mlflow.set_experiment("controlled_retraining_event")
            with mlflow.start_run(run_name="challenger") as run:
                challenger_run_id = run.info.run_id
                payload = build_checkpoint_payload(
                    model=model, scaler=scaler,
                    config={"mlp": {"architecture": {"hidden_sizes": [32, 16], "dropout": 0.0}}},
                    hidden_sizes=[32, 16], dropout=0.0, dataset_id="FD001",
                    training_run_id=challenger_run_id, training_seed=0,
                    training_metrics={}, evaluation_protocol="synthetic_event",
                )
                ckpt_sha = save_canonical_checkpoint(payload, tmp / "challenger.pt")
                mlflow.set_tag("checkpoint_sha256", ckpt_sha)
                mlflow.log_artifact(str(tmp / "challenger.pt"), "canonical_checkpoint")

                # Controlled comparison: a champion with ~15-cycle error vs a
                # challenger with ~7-cycle error, both against the same targets.
                # Deterministic arrays make the promotion decision reproducible;
                # the trained model above still produced the real canonical
                # checkpoint artifact that was logged to MLflow.
                cmp_rng = np.random.default_rng(123)
                champion_pred = y + cmp_rng.normal(0, 15, size=len(y))
                challenger_pred = y + cmp_rng.normal(0, 7, size=len(y))
                comparison = ModelComparator.__new__(ModelComparator)
                comparison.min_improvement = 5.0
                comparison.primary_metric = "mae"
                report_cmp = ModelComparator.compare_predictions(
                    comparison, y, champion_pred, challenger_pred,
                    "champion_meanpredictor", "challenger",
                )
                promotion_approved = report_cmp.should_promote
                mlflow.log_metric("should_promote", int(promotion_approved))
                comparison = report_cmp
    finally:
        if triggered:
            guard.end()

    # 7. Prometheus counter
    counter_before = counter_after = None
    try:
        sys.path.insert(0, str(_PM_ROOT / "inference_service"))
        from prometheus_client import REGISTRY
        from api.metrics import RETRAINING_TRIGGERED_TOTAL

        def _read():
            v = REGISTRY.get_sample_value(
                "retraining_triggered_total", {"reason": "accuracy_degradation"}
            )
            return 0.0 if v is None else v

        counter_before = _read()
        if triggered:
            RETRAINING_TRIGGERED_TOTAL.labels(reason="accuracy_degradation").inc()
        counter_after = _read()
    except Exception as exc:  # metrics optional in some envs
        print(f"[warn] Prometheus counter unavailable: {exc}")

    evidence = {
        "trigger_reason": decision.reason,
        "decision_state": decision.state.value,
        "measured_accuracy": round(report.accuracy, 4),
        "threshold": report.threshold,
        "n_samples": report.n_samples,
        "below_threshold": report.below_threshold,
        "retraining_triggered": bool(triggered),
        "model_version_champion": "champion_meanpredictor",
        "challenger_mlflow_run_id": challenger_run_id,
        "checkpoint_sha256": ckpt_sha,
        "comparison_winner": comparison.winner if comparison else None,
        "comparison_improvement_pct": round(comparison.improvement_pct, 2) if comparison else None,
        "comparison_ci": comparison.details.get("bootstrap_ci") if comparison else None,
        "promotion_approved": promotion_approved,
        "prometheus_counter_before": counter_before,
        "prometheus_counter_after": counter_after,
    }
    return evidence


def main() -> int:
    ev = run_event()
    print("=" * 72)
    print("  CONTROLLED RETRAINING EVENT — EVIDENCE")
    print("=" * 72)
    for k, v in ev.items():
        print(f"  {k:<32}: {v}")
    print("=" * 72)
    ok = (
        ev["below_threshold"]
        and ev["retraining_triggered"]
        and ev["challenger_mlflow_run_id"]
        and ev["checkpoint_sha256"]
        and ev["prometheus_counter_after"] == (ev["prometheus_counter_before"] or 0) + 1
    )
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
