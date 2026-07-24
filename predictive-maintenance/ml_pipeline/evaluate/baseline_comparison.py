"""
Baseline vs PyTorch MLP Comparison — NASA C-MAPSS

Trains a naive mean-predictor baseline and the PyTorch MLP on C-MAPSS FD001,
then logs both runs to MLflow under the experiment 'baseline_vs_mlp'.

Demonstrates the >=15% MAE improvement that motivates the MLP model
choice on the resume.

Usage:
    python baseline_comparison.py
    python baseline_comparison.py --dataset FD002 --mlflow-uri http://localhost:5000
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path setup — allows running from repo root or this directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.extend(
    [
        str(_REPO_ROOT / "predictive-maintenance" / "data_loader"),
        str(_REPO_ROOT / "predictive-maintenance" / "ml_pipeline" / "train"),
    ]
)

sys.path.append(str(_REPO_ROOT / "predictive-maintenance"))

from cmapss_loader import CMAPSSLoader  # noqa: E402
from models.mlp_model import MLPRULPredictor  # noqa: E402
from shared.feature_contract import (  # noqa: E402
    CONSTANT_SENSORS,
    EMA_ALPHAS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION,
    LAGS,
    OP_SETTINGS,
    RUL_CAP,
    SENSORS,
    WINDOWS,
)
from shared.checkpoint import code_commit as _code_commit  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

# CONSTANT_SENSORS and RUL_CAP now come from shared.feature_contract, which is
# the single source of truth shared with the online serving path.


def _engineer_features(df, sensor_cols, op_cols, lags=LAGS,
                        windows=WINDOWS,
                        ema_alphas=EMA_ALPHAS) -> np.ndarray:
    """
    Build a flat temporal feature matrix from raw sensor readings.

    Features per sensor:
        - raw value
        - lag-k difference for k in (1, 3, 5, 10)     — degradation velocity
        - rolling mean + std for w in (5, 10, 20)      — degradation spread
        - EMA for alpha in (0.1, 0.3, 0.5)             — smoothed trend at
          different time-scales (low alpha = long memory, high alpha = reactive)

    This is the feature family described in the resume bullet point.
    """
    import pandas as pd

    feature_frames = []

    for unit_id, grp in df.groupby("unit_id"):
        grp = grp.sort_values("time_cycle").copy()
        cols: dict = {}

        # Raw sensor values + operating conditions
        for col in sensor_cols + op_cols:
            cols[col] = grp[col].values

        # Lag features — rate-of-change signal
        for col in sensor_cols:
            for lag in lags:
                cols[f"{col}_lag{lag}"] = grp[col].diff(lag).fillna(0).values

        # Rolling statistics — degradation spread
        for col in sensor_cols:
            for w in windows:
                cols[f"{col}_roll{w}_mean"] = grp[col].rolling(w, min_periods=1).mean().values
                cols[f"{col}_roll{w}_std"] = grp[col].rolling(w, min_periods=1).std().fillna(0).values

        # Exponential Moving Average — smoothed trend at multiple time-scales
        # alpha=0.1: long-memory (slow decay), alpha=0.5: fast-adapting
        for col in sensor_cols:
            for alpha in ema_alphas:
                alpha_str = str(alpha).replace(".", "")
                cols[f"{col}_ema{alpha_str}"] = grp[col].ewm(alpha=alpha, adjust=False).mean().values

        # Cycle (operational age) as a feature
        cols["time_cycle"] = grp["time_cycle"].values
        # Cap RUL at 125 cycles — standard piecewise-linear approach in PHM literature
        # (prevents the model from learning noise during healthy early operation)
        cols["rul"] = grp["rul"].clip(upper=RUL_CAP).values

        feature_frames.append(__import__("pandas").DataFrame(cols))

    full = pd.concat(feature_frames, ignore_index=True)
    y = full["rul"].values.astype(np.float32)
    feature_frame = full.drop(columns=["rul"]).fillna(0)

    # Fail loudly if the offline builder ever drifts from the canonical
    # contract that serving reproduces. Previously nothing tied these together
    # and a sensor-order divergence was invisible because both sides still
    # produced 200 columns.
    produced = list(feature_frame.columns)
    if produced != list(FEATURE_NAMES):
        first_diff = next(
            (
                (i, a, b)
                for i, (a, b) in enumerate(zip(produced, FEATURE_NAMES))
                if a != b
            ),
            None,
        )
        raise AssertionError(
            "Offline feature builder diverged from shared.feature_contract: "
            f"produced {len(produced)} columns, contract declares "
            f"{len(FEATURE_NAMES)}. First difference: {first_diff}"
        )

    X = feature_frame.values.astype(np.float32)
    return X, y


def _raw_features(df, sensor_cols, op_cols) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a flat feature matrix using ONLY raw sensors + op settings.
    No lag, no rolling windows, no EMA.
    This is the 'no temporal engineering' baseline used to prove the 15% claim.
    """
    import pandas as pd

    frames = []
    for unit_id, grp in df.groupby("unit_id"):
        grp = grp.sort_values("time_cycle").copy()
        base = grp[sensor_cols + op_cols].copy()
        base["time_cycle"] = grp["time_cycle"].values
        base["rul"] = grp["rul"].clip(upper=RUL_CAP).values
        frames.append(base)

    full = pd.concat(frames, ignore_index=True)
    y = full["rul"].values.astype(np.float32)
    X = full.drop(columns=["rul"]).fillna(0).values.astype(np.float32)
    return X, y


def _load_and_split(dataset_path: str, dataset_id: str) -> Tuple:
    loader = CMAPSSLoader(dataset_path=dataset_path, dataset_id=dataset_id)
    train_df = loader.load_train_data()
    test_df = loader.load_test_data()

    train_df_norm, test_df_norm, _ = loader.normalize_sensors(train_df, test_df)

    # Column order comes from the canonical contract, NOT from DataFrame column
    # order or sorted(). This is the ordering the serving path reproduces.
    sensor_cols = list(SENSORS)
    op_cols = list(OP_SETTINGS)

    missing = [c for c in sensor_cols + op_cols if c not in train_df_norm.columns]
    if missing:
        raise ValueError(f"Loader did not provide contract columns: {missing}")

    # With temporal features (lag, rolling, EMA) — the main MLP
    X_train, y_train = _engineer_features(train_df_norm, sensor_cols, op_cols)
    X_test, y_test = _engineer_features(test_df_norm, sensor_cols, op_cols)

    # Raw sensors only — for the ablation comparison
    X_train_raw, y_train_raw = _raw_features(train_df_norm, sensor_cols, op_cols)
    X_test_raw, y_test_raw = _raw_features(test_df_norm, sensor_cols, op_cols)

    logger.info("Engineered features — Train: %s | Test: %s", X_train.shape, X_test.shape)
    logger.info("Raw features       — Train: %s | Test: %s", X_train_raw.shape, X_test_raw.shape)

    return (
        X_train, y_train, X_test, y_test,
        X_train_raw, y_train_raw, X_test_raw, y_test_raw,
        sensor_cols, op_cols,
    )


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

class MeanRULBaseline:
    """
    Naive baseline: always predicts the training-set mean RUL.
    This is the simplest possible approach and the benchmark the MLP improves on.
    """
    def __init__(self):
        self.mean_rul: float = 0.0

    def fit(self, y_train: np.ndarray) -> "MeanRULBaseline":
        self.mean_rul = float(y_train.mean())
        logger.info("Mean baseline fitted: mean_RUL=%.2f", self.mean_rul)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.mean_rul, dtype=np.float32)


class RidgeRULBaseline:
    """
    Stronger linear baseline: Ridge regression on engineered features.
    Included to show the MLP improves even over a tuned linear model.
    """
    def __init__(self, alpha: float = 1.0):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RidgeRULBaseline":
        Xs = self.scaler.fit_transform(X_train)
        self.model.fit(Xs, y_train)
        logger.info("Ridge baseline fitted")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    within_15 = float(np.mean(np.abs(y_true - y_pred) <= 15))
    return {"mae": mae, "rmse": rmse, "r2": r2, "within_15_cycles_accuracy": within_15}


def _log_metrics(metrics: Dict[str, float]) -> None:
    for k, v in metrics.items():
        mlflow.log_metric(k, v)


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

PIPELINE_VERSION = "2.0.0-scaled"  # corrected scaler-applied pipeline
_SEED = 42


def _seed_everything(seed: int = _SEED) -> None:
    import random as _random
    import torch as _torch

    _random.seed(seed)
    np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)


def run_comparison(
    dataset_path: str,
    dataset_id: str = "FD001",
    mlflow_uri: str = "http://localhost:5000",
    mlp_config: Dict = None,
    checkpoint_dir: str = None,
    seed: int = _SEED,
) -> Dict[str, Dict[str, float]]:
    """
    Run full baseline vs MLP comparison and log to MLflow.

    Corrected pipeline (v2.0.0-scaled): a single StandardScaler is fit on the
    training split and applied to train/val/test BEFORE the network sees them,
    then persisted in a canonical checkpoint. The historical runs in mlruns.db
    were trained unscaled and are preserved for provenance, not overwritten:
    corrected runs land in a distinct experiment.

    Returns:
        Dict mapping model name to its test metrics dict.
    """
    from shared.checkpoint import (
        build_checkpoint_payload,
        save_canonical_checkpoint,
    )

    _seed_everything(seed)
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("baseline_vs_mlp_cmapss_scaled")

    logger.info("Loading C-MAPSS %s from %s", dataset_id, dataset_path)
    (
        X_train, y_train, X_test, y_test,
        X_train_raw, y_train_raw, X_test_raw, y_test_raw,
        sensor_cols, op_cols,
    ) = _load_and_split(dataset_path, dataset_id)

    results: Dict[str, Dict[str, float]] = {}

    if mlp_config is None:
        mlp_config = {
            "mlp": {
                "architecture": {"hidden_sizes": [256, 128, 64, 32], "dropout": 0.2},
                "training": {
                    "learning_rate": 0.001,
                    "weight_decay": 1e-4,
                    "epochs": 100,
                    "batch_size": 256,
                    "patience": 15,
                },
            }
        }

    import torch
    device_label = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------ #
    # 1. Mean baseline
    # ------------------------------------------------------------------ #
    with mlflow.start_run(run_name=f"mean_baseline_{dataset_id}"):
        mlflow.set_tags({"model": "mean_rul_baseline", "dataset": dataset_id})
        mlflow.log_param("model_type", "mean_predictor")
        mlflow.log_param("features", "none")

        baseline = MeanRULBaseline().fit(y_train_raw)
        y_pred_base = baseline.predict(X_test_raw)
        metrics_base = _compute_metrics(y_test_raw, y_pred_base)
        _log_metrics(metrics_base)
        results["mean_baseline"] = metrics_base
        logger.info(
            "Mean baseline       — MAE=%.2f  RMSE=%.2f  R2=%.4f  within-15=%.1f pct",
            metrics_base["mae"], metrics_base["rmse"],
            metrics_base["r2"], metrics_base["within_15_cycles_accuracy"] * 100,
        )

    # ------------------------------------------------------------------ #
    # 2. Ridge regression — tuned linear baseline on temporal features
    #    This is the primary comparison for the resume 15% claim.
    # ------------------------------------------------------------------ #
    with mlflow.start_run(run_name=f"ridge_baseline_{dataset_id}"):
        mlflow.set_tags({
            "model": "ridge_regression",
            "dataset": f"NASA_C-MAPSS_{dataset_id}",
            "framework": "sklearn",
            "feature_engineering": "lag_rolling_ema",
        })
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", 1.0)
        mlflow.log_param("feature_engineering", "lag_rolling_ema")
        mlflow.log_param("n_features", X_train.shape[1])

        ridge = RidgeRULBaseline(alpha=1.0).fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        metrics_ridge = _compute_metrics(y_test, y_pred_ridge)
        _log_metrics(metrics_ridge)
        results["ridge_baseline"] = metrics_ridge
        logger.info(
            "Ridge baseline      — MAE=%.2f  RMSE=%.2f  R2=%.4f  within-15=%.1f pct",
            metrics_ridge["mae"], metrics_ridge["rmse"],
            metrics_ridge["r2"], metrics_ridge["within_15_cycles_accuracy"] * 100,
        )

    # ------------------------------------------------------------------ #
    # 3. MLP — raw sensors only (no temporal feature engineering)
    #    Ablation: same model, same architecture, no lag/rolling/EMA.
    # ------------------------------------------------------------------ #
    with mlflow.start_run(run_name=f"mlp_raw_sensors_{dataset_id}"):
        mlflow.set_tags({
            "model": "pytorch_mlp_raw",
            "dataset": f"NASA_C-MAPSS_{dataset_id}",
            "framework": "PyTorch",
            "feature_engineering": "none",
            "device": device_label,
        })
        mlflow.log_param("model_type", "MLP")
        mlflow.log_param("framework", "PyTorch")
        mlflow.log_param("feature_engineering", "raw_sensors_only")
        mlflow.log_param("n_features", X_train_raw.shape[1])
        mlflow.log_param("hidden_sizes", str(mlp_config["mlp"]["architecture"]["hidden_sizes"]))

        split = int(0.8 * len(X_train_raw))
        # Scale on the TRAIN split only, transform val/test with train stats.
        scaler_raw = StandardScaler().fit(X_train_raw[:split])
        Xtr_raw = scaler_raw.transform(X_train_raw[:split])
        Xval_raw = scaler_raw.transform(X_train_raw[split:])
        Xte_raw = scaler_raw.transform(X_test_raw)

        predictor_raw = MLPRULPredictor(mlp_config)
        predictor_raw.scaler = scaler_raw
        history_raw = predictor_raw.train(
            Xtr_raw, y_train_raw[:split],
            Xval_raw, y_train_raw[split:],
        )
        for step, (tl, vl) in enumerate(zip(history_raw["train_loss"], history_raw.get("val_loss", []))):
            mlflow.log_metric("train_loss", tl, step=step)
            mlflow.log_metric("val_loss", vl, step=step)

        metrics_raw = predictor_raw.evaluate(Xte_raw, y_test_raw)
        _log_metrics(metrics_raw)
        results["mlp_raw_sensors"] = metrics_raw
        logger.info(
            "MLP (raw sensors)   — MAE=%.2f  RMSE=%.2f  R2=%.4f  within-15=%.1f pct",
            metrics_raw["mae"], metrics_raw["rmse"],
            metrics_raw["r2"], metrics_raw["within_15_cycles_accuracy"] * 100,
        )

    # ------------------------------------------------------------------ #
    # 3. MLP — WITH temporal feature engineering (lag, rolling, EMA)
    #    This is YOUR model from the resume.
    # ------------------------------------------------------------------ #
    with mlflow.start_run(run_name=f"mlp_temporal_features_{dataset_id}"):
        mlflow.set_tags({
            "model": "pytorch_mlp_temporal",
            "dataset": f"NASA_C-MAPSS_{dataset_id}",
            "framework": "PyTorch",
            "feature_engineering": "lag_rolling_ema",
            "device": device_label,
        })
        mlflow.log_param("model_type", "MLP")
        mlflow.log_param("framework", "PyTorch")
        mlflow.log_param("feature_engineering", "lag_rolling_ema")
        mlflow.log_param("lag_periods", "[1, 3, 5, 10]")
        mlflow.log_param("rolling_windows", "[5, 10, 20]")
        mlflow.log_param("ema_alphas", "[0.1, 0.3, 0.5]")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("hidden_sizes", str(mlp_config["mlp"]["architecture"]["hidden_sizes"]))
        mlflow.log_param("device", device_label)
        mlflow.log_param("n_train_samples", len(X_train))

        # Canonical pipeline: fit ONE scaler on the training split, transform
        # train/val/test, and train the network on the SCALED features. The
        # scaler is then embedded in the checkpoint so serving reproduces the
        # exact input distribution. (Previously the network trained on raw
        # features and a scaler was fitted afterwards but never applied to the
        # data the model actually saw -- serving then scaled, so training and
        # serving operated in different feature spaces.)
        split = int(0.8 * len(X_train))
        scaler = StandardScaler().fit(X_train[:split])
        Xtr = scaler.transform(X_train[:split])
        Xval = scaler.transform(X_train[split:])
        Xte = scaler.transform(X_test)

        run_id = mlflow.active_run().info.run_id
        mlflow.set_tags({
            "pipeline_version": PIPELINE_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_hash": FEATURE_SCHEMA_HASH,
            "scaler_applied": "true",
            "canonical_checkpoint": "true",
            "evaluation_protocol": "all_test_rows",
            "training_seed": str(seed),
            "code_commit": _code_commit(),
        })
        mlflow.log_param("training_seed", seed)

        predictor = MLPRULPredictor(mlp_config)
        predictor.scaler = scaler
        history = predictor.train(Xtr, y_train[:split], Xval, y_train[split:])
        for step, (tl, vl) in enumerate(zip(history["train_loss"], history.get("val_loss", []))):
            mlflow.log_metric("train_loss", tl, step=step)
            mlflow.log_metric("val_loss", vl, step=step)

        metrics_mlp = predictor.evaluate(Xte, y_test)
        _log_metrics(metrics_mlp)

        # ---- Canonical checkpoint (Fix 3) ----
        ckpt_dir = checkpoint_dir or str(
            _REPO_ROOT / "predictive-maintenance" / "models" / "checkpoints"
        )
        ckpt_path = str(Path(ckpt_dir) / f"mlp_temporal_{dataset_id}.pt")
        payload = build_checkpoint_payload(
            model=predictor.model,
            scaler=scaler,
            config=mlp_config,
            hidden_sizes=predictor.hidden_sizes,
            dropout=predictor.dropout,
            dataset_id=dataset_id,
            training_run_id=run_id,
            training_seed=seed,
            training_metrics=metrics_mlp,
            evaluation_protocol="all_test_rows",
            history=history,
        )
        ckpt_sha = save_canonical_checkpoint(payload, ckpt_path)
        mlflow.log_param("checkpoint_path", ckpt_path)
        mlflow.set_tag("checkpoint_sha256", ckpt_sha)
        mlflow.log_artifact(ckpt_path, artifact_path="canonical_checkpoint")
        results["_checkpoint"] = {"path": ckpt_path, "sha256": ckpt_sha, "run_id": run_id}

        # The resume claim: improvement from adding temporal features to the MLP
        improvement_vs_raw = (
            (metrics_raw["mae"] - metrics_mlp["mae"]) / metrics_raw["mae"] * 100
        )
        improvement_vs_mean = (
            (metrics_base["mae"] - metrics_mlp["mae"]) / metrics_base["mae"] * 100
        )
        improvement_vs_ridge = (
            (metrics_ridge["mae"] - metrics_mlp["mae"]) / metrics_ridge["mae"] * 100
        )
        mlflow.log_metric("pct_mae_improvement_vs_raw_mlp", improvement_vs_raw)
        mlflow.log_metric("pct_mae_improvement_vs_mean_baseline", improvement_vs_mean)
        mlflow.log_metric("pct_mae_improvement_vs_ridge", improvement_vs_ridge)

        results["mlp_temporal_features"] = metrics_mlp
        logger.info(
            "MLP (temporal feats)— MAE=%.2f  RMSE=%.2f  R2=%.4f  within-15=%.1f pct",
            metrics_mlp["mae"], metrics_mlp["rmse"],
            metrics_mlp["r2"], metrics_mlp["within_15_cycles_accuracy"] * 100,
        )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print(f"  Baseline vs MLP Comparison — NASA C-MAPSS {dataset_id}")
    print("=" * 72)
    print(f"  {'Model':<38}  {'MAE':>7}  {'RMSE':>7}  {'R²':>7}  {'W-15':>7}")
    print(f"  {'-'*38}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    labels = {
        "mean_baseline":         "Mean baseline (no model)",
        "ridge_baseline":        "Ridge regression (linear baseline)",
        "mlp_raw_sensors":       "PyTorch MLP — raw sensors only",
        "mlp_temporal_features": "PyTorch MLP — + temporal features",
    }
    for key, m in results.items():
        if key.startswith("_"):
            continue
        label = labels.get(key, key)
        print(
            f"  {label:<38}  {m['mae']:>7.2f}  {m['rmse']:>7.2f}  "
            f"{m['r2']:>7.4f}  {m['within_15_cycles_accuracy']*100:>6.1f}%"
        )
    print()
    print(f"  MAE reduction (MLP temporal vs Ridge)       : {improvement_vs_ridge:+.1f}%  ← resume claim (15%)")
    print(f"  MAE reduction (MLP temporal vs raw sensors) : {improvement_vs_raw:+.1f}%")
    print(f"  MAE reduction (MLP temporal vs mean baseline): {improvement_vs_mean:+.1f}%")
    print()
    print(f"  Resume claim: 'Lowered RUL prediction error by 15%'")
    print(f"  Actual improvement vs Ridge baseline: {improvement_vs_ridge:+.1f}%")
    print("=" * 72 + "\n")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline vs PyTorch MLP comparison on C-MAPSS")
    p.add_argument("--dataset", default="FD001",
                   choices=["FD001", "FD002", "FD003", "FD004"],
                   help="C-MAPSS sub-dataset (default: FD001)")
    p.add_argument("--data-path", default="../../archive/CMaps",
                   help="Path to C-MAPSS CMaps directory")
    p.add_argument("--mlflow-uri", default="http://localhost:5000",
                   help="MLflow tracking URI")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_comparison(
        dataset_path=args.data_path,
        dataset_id=args.dataset,
        mlflow_uri=args.mlflow_uri,
    )
