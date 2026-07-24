"""
Resume Evidence Printer — "Lowered turbofan engine RUL prediction error by 15%"

Reads the logged MLflow runs for the ``baseline_vs_mlp_cmapss`` experiment
straight out of the local SQLite tracking store (``mlruns.db``) and prints
the MAE comparison that backs the resume's 15% claim.

No live MLflow server, network access, or GPU is required — the SQLite file
is a self-contained MLflow backend store, so this script can be run anywhere
the repo (with ``mlruns.db``) is checked out.

Usage:
    python print_resume_evidence.py
    python print_resume_evidence.py --db /path/to/mlruns.db --experiment baseline_vs_mlp_cmapss
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DB = _REPO_ROOT / "mlruns.db"

# Run-name prefixes → human labels, in the order they should be printed.
_RUN_PREFIXES = {
    "mean_baseline": "Mean baseline (no model)",
    "ridge_baseline": "Ridge regression (linear baseline)",
    "mlp_raw_sensors": "PyTorch MLP — raw sensors only",
    "mlp_temporal_features": "PyTorch MLP + temporal features (resume model)",
}

_RESUME_CLAIM_PCT = 15.0


def _latest_finished_run(client: MlflowClient, experiment_id: str, run_name_prefix: str):
    """Most recent FINISHED run whose name starts with the given prefix."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["start_time DESC"],
        max_results=200,
    )
    for run in runs:
        name = run.data.tags.get("mlflow.runName", "")
        if name.startswith(run_name_prefix) and run.info.status == "FINISHED":
            return run
    return None


def load_resume_evidence(
    db_path: Optional[Path] = None,
    experiment_name: str = "baseline_vs_mlp_cmapss",
) -> Dict[str, Dict]:
    """
    Load the latest finished run of each baseline/MLP variant from the local
    MLflow SQLite store and return their key metrics + tags.

    Returns:
        {run_prefix: {"mae": float, "dataset": str, "run_id": str, "label": str}, ...}
    """
    db_path = Path(db_path) if db_path else _DEFAULT_DB
    if not db_path.exists():
        raise FileNotFoundError(
            f"MLflow SQLite store not found at {db_path}. "
            f"Run baseline_comparison.py first to generate it, or pass --db."
        )

    client = MlflowClient(tracking_uri=f"sqlite:///{db_path}")
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{experiment_name}' not found in {db_path}. "
            f"Run baseline_comparison.py to populate it."
        )

    evidence: Dict[str, Dict] = {}
    for prefix, label in _RUN_PREFIXES.items():
        run = _latest_finished_run(client, experiment.experiment_id, prefix)
        if run is None:
            continue
        evidence[prefix] = {
            "label": label,
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName"),
            "dataset": run.data.tags.get("dataset", "unknown"),
            "mae": run.data.metrics.get("mae"),
            "rmse": run.data.metrics.get("rmse"),
            "r2": run.data.metrics.get("r2"),
            "within_15_cycles_accuracy": run.data.metrics.get("within_15_cycles_accuracy"),
            "pct_mae_improvement_vs_ridge": run.data.metrics.get("pct_mae_improvement_vs_ridge"),
            "pct_mae_improvement_vs_raw_mlp": run.data.metrics.get("pct_mae_improvement_vs_raw_mlp"),
            # Provenance tags distinguishing corrected (scaled) runs from the
            # historical unscaled runs.
            "scaler_applied": run.data.tags.get("scaler_applied"),
            "pipeline_version": run.data.tags.get("pipeline_version"),
            "feature_schema_hash": run.data.tags.get("feature_schema_hash"),
            "canonical_checkpoint": run.data.tags.get("canonical_checkpoint"),
            "checkpoint_sha256": run.data.tags.get("checkpoint_sha256"),
            "evaluation_protocol": run.data.tags.get("evaluation_protocol"),
            "code_commit": run.data.tags.get("code_commit"),
        }
    return evidence


def load_best_available_evidence(db_path: Optional[Path] = None):
    """
    Return (experiment_name, evidence, is_corrected).

    Prefers the corrected scaler-applied experiment; falls back to the
    historical unscaled experiment, clearly flagged.
    """
    for name, corrected in (
        ("baseline_vs_mlp_cmapss_scaled", True),
        ("baseline_vs_mlp_cmapss", False),
    ):
        try:
            ev = load_resume_evidence(db_path, experiment_name=name)
        except RuntimeError:
            continue
        if ev.get("mlp_temporal_features"):
            return name, ev, corrected
    raise RuntimeError("No usable experiment found in the MLflow store.")


def print_resume_evidence(db_path: Optional[Path] = None) -> bool:
    """
    Print the resume-claim evidence table and verdict.

    Returns:
        True if the measured MAE improvement (MLP temporal vs Ridge) is
        >= the resume's 15% claim, False otherwise.
    """
    experiment_name, evidence, is_corrected = load_best_available_evidence(db_path)

    ridge = evidence.get("ridge_baseline")
    mlp_temporal = evidence.get("mlp_temporal_features")
    mlp_raw = evidence.get("mlp_raw_sensors")
    mean_base = evidence.get("mean_baseline")

    print("=" * 76)
    print('  Resume claim: "Lowered turbofan engine RUL prediction error by 15%"')
    print(f"  Source: MLflow experiment '{experiment_name}' (mlruns.db, SQLite)")
    if is_corrected:
        print("  Pipeline: CORRECTED — scaler fitted on train, applied before the")
        print("            network, persisted in the canonical checkpoint.")
    else:
        print("  Pipeline: HISTORICAL/UNSCALED — no corrected scaled experiment found.")
        print("            These numbers do NOT describe the serving path. Run")
        print("            baseline_comparison.py to regenerate corrected evidence.")
    print("=" * 76)

    if mlp_temporal:
        print(f"  Dataset                : {mlp_temporal['dataset']}")
        print(f"  MLflow run (MLP model) : {mlp_temporal['run_name']}  ({mlp_temporal['run_id'][:8]})")
        if mlp_temporal.get("feature_schema_hash"):
            print(f"  Feature schema hash    : {mlp_temporal['feature_schema_hash'][:16]}")
        if mlp_temporal.get("checkpoint_sha256"):
            print(f"  Canonical checkpoint   : sha256 {mlp_temporal['checkpoint_sha256'][:16]}")
        if mlp_temporal.get("evaluation_protocol"):
            print(f"  Evaluation protocol    : {mlp_temporal['evaluation_protocol']}")
        if mlp_temporal.get("code_commit"):
            print(f"  Code commit            : {mlp_temporal['code_commit'][:8]}")
        if mlp_temporal.get("scaler_applied"):
            print(f"  Scaler applied         : {mlp_temporal['scaler_applied']}")
    print()
    print(f"  {'Model':<48}  {'MAE (cycles)':>14}")
    print(f"  {'-'*48}  {'-'*14}")
    for key in ("mean_baseline", "ridge_baseline", "mlp_raw_sensors", "mlp_temporal_features"):
        row = evidence.get(key)
        if row and row["mae"] is not None:
            print(f"  {row['label']:<48}  {row['mae']:>14.2f}")
    print()

    if not (ridge and mlp_temporal and ridge.get("mae") and mlp_temporal.get("mae")):
        print("  Could not locate both 'ridge_baseline' and 'mlp_temporal_features' "
              "runs with MAE metrics — cannot verify the claim.")
        return False

    ridge_mae = ridge["mae"]
    mlp_mae = mlp_temporal["mae"]
    improvement_pct = (ridge_mae - mlp_mae) / ridge_mae * 100.0

    # Prefer the value MLflow logged at training time; fall back to recomputing.
    logged_improvement = mlp_temporal.get("pct_mae_improvement_vs_ridge")
    if logged_improvement is not None:
        improvement_pct = logged_improvement

    print(f"  Ridge baseline MAE                 : {ridge_mae:.2f} cycles")
    print(f"  MLP (+ temporal features) MAE      : {mlp_mae:.2f} cycles")
    if mlp_raw and mlp_raw.get("mae") is not None:
        print(f"  MLP (raw sensors, no engineering)  : {mlp_raw['mae']:.2f} cycles  (ablation reference)")
    if mean_base and mean_base.get("mae") is not None:
        print(f"  Mean-predictor baseline MAE        : {mean_base['mae']:.2f} cycles  (naive reference)")
    print()
    print(f"  MAE improvement (MLP temporal vs Ridge)    : {improvement_pct:+.1f}%")
    raw_impr = mlp_temporal.get("pct_mae_improvement_vs_raw_mlp")
    if raw_impr is None and mlp_raw and mlp_raw.get("mae"):
        raw_impr = (mlp_raw["mae"] - mlp_mae) / mlp_raw["mae"] * 100.0
    if raw_impr is not None:
        print(f"  MAE improvement (temporal features vs raw) : {raw_impr:+.1f}%")
    print("  Note: 'vs Ridge' measures MLP architecture + temporal features over")
    print("        a linear model; 'vs raw' isolates the temporal features alone.")
    print(f"  Resume claim                       : >= {_RESUME_CLAIM_PCT:.0f}%")

    meets_claim = improvement_pct >= _RESUME_CLAIM_PCT
    verdict = "PASS — measured improvement meets/exceeds the resume claim" \
        if meets_claim else \
        "FAIL — measured improvement is BELOW the resume claim"
    print(f"  Verdict                            : {verdict}")
    print("=" * 76)

    return meets_claim


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print MLflow-backed evidence for the resume's 15% MAE-improvement claim"
    )
    p.add_argument("--db", default=None, help="Path to mlruns.db (default: repo root)")
    p.add_argument(
        "--experiment", default="baseline_vs_mlp_cmapss",
        help="MLflow experiment name (default: baseline_vs_mlp_cmapss)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    db = Path(args.db) if args.db else None
    ok = print_resume_evidence(db)
    sys.exit(0 if ok else 1)
