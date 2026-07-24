"""
MLflow model lifecycle integration test (Fix 4).

Uses a temporary local MLflow SQLite store + filesystem artifact root -- no
network, no external MLflow server. Exercises the full production path:

    train -> canonical checkpoint -> log artifact -> register version
    -> set 'champion' alias -> download artifact -> load through ModelManager
    -> predict -> assert scaler/schema/version fidelity and prediction equality.

Run:
    .venv/bin/python -m pytest tests/integration/test_mlflow_lifecycle.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_PM_ROOT = Path(__file__).resolve().parents[2]
for _p in (
    str(_PM_ROOT),
    str(_PM_ROOT / "ml_pipeline" / "train"),
    str(_PM_ROOT / "inference_service"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

torch = pytest.importorskip("torch")
mlflow = pytest.importorskip("mlflow")

from sklearn.preprocessing import StandardScaler  # noqa: E402

from shared import feature_contract as fc  # noqa: E402
from shared.checkpoint import (  # noqa: E402
    _load_mlp_class,
    build_checkpoint_payload,
    save_canonical_checkpoint,
)

MLPRULNet = _load_mlp_class()
HIDDEN = [32, 16]


@pytest.mark.integration
def test_full_mlflow_lifecycle_preserves_scaler_schema_and_predictions(tmp_path):
    from mlflow.tracking import MlflowClient

    tracking_uri = f"sqlite:///{tmp_path/'mlflow.db'}"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    # --- train a tiny deterministic model on scaled features ---
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, fc.INPUT_DIM))
    y = (X[:, :5].sum(axis=1) * 3 + 50).astype(np.float32)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)

    torch.manual_seed(0)
    model = MLPRULNet(input_dim=fc.INPUT_DIM, hidden_sizes=HIDDEN, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.MSELoss()
    model.train()
    for _ in range(30):
        opt.zero_grad()
        loss = lossf(model(torch.tensor(Xs)), torch.tensor(y))
        loss.backward()
        opt.step()
    model.eval()

    ckpt_path = tmp_path / "mlp_temporal_FD001.pt"

    mlflow.set_experiment("lifecycle_test")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        payload = build_checkpoint_payload(
            model=model, scaler=scaler,
            config={"mlp": {"architecture": {"hidden_sizes": HIDDEN, "dropout": 0.0}}},
            hidden_sizes=HIDDEN, dropout=0.0, dataset_id="FD001",
            training_run_id=run_id, training_seed=0,
            training_metrics={"mae": float(loss.item())},
            evaluation_protocol="all_test_rows",
        )
        sha = save_canonical_checkpoint(payload, ckpt_path)
        mlflow.log_artifact(str(ckpt_path), artifact_path="canonical_checkpoint")
        mlflow.set_tag("checkpoint_sha256", sha)

    # --- register a version pointing at the artifact, alias it champion ---
    registry_name = "predictive_maintenance_model"
    client.create_registered_model(registry_name)
    src = f"runs:/{run_id}/canonical_checkpoint"
    mv = client.create_model_version(registry_name, source=src, run_id=run_id)
    client.set_registered_model_alias(registry_name, "champion", mv.version)

    # --- production resolution: alias -> version -> download artifact ---
    champion = client.get_model_version_by_alias(registry_name, "champion")
    assert champion.version == mv.version
    local_dir = mlflow.artifacts.download_artifacts(
        run_id=champion.run_id, artifact_path="canonical_checkpoint",
    )
    resolved_ckpt = Path(local_dir) / "mlp_temporal_FD001.pt"
    assert resolved_ckpt.exists()

    # --- load through the SAME ModelManager branch production uses ---
    from models.model_manager import ModelManager

    mm = ModelManager()
    mm._lock = __import__("threading").RLock()
    mm._models = {}
    mm._model_metadata = {}
    loaded = mm.load_canonical_mlp(
        str(resolved_ckpt), registry_name, champion.version,
        run_id=champion.run_id, expected_sha256=sha,
    )

    meta = mm._model_metadata["mlp"]
    assert meta["loaded"] is True
    assert meta["scaler_fitted"] is True
    assert meta["feature_schema_hash"] == fc.FEATURE_SCHEMA_HASH
    assert meta["input_dim"] == fc.INPUT_DIM
    assert meta["version"] == champion.version
    assert meta["checkpoint_sha256"] == sha
    assert getattr(loaded, "_scaler", None) is not None

    # --- predictions identical pre- and post-round-trip ---
    Xnew = rng.normal(size=(8, fc.INPUT_DIM))
    with torch.no_grad():
        expected = model(torch.tensor(scaler.transform(Xnew).astype(np.float32))).numpy()
        got = loaded(
            torch.tensor(loaded._scaler.transform(Xnew).astype(np.float32))
        ).numpy()
    np.testing.assert_allclose(expected, got, rtol=1e-5, atol=1e-5)


@pytest.mark.integration
def test_modelmanager_rejects_unsafe_checkpoint(tmp_path):
    """A legacy (non-canonical) checkpoint must be refused, not served unscaled."""
    from models.model_manager import ModelManager

    legacy = tmp_path / "legacy.pt"
    torch.manual_seed(0)
    m = MLPRULNet(input_dim=fc.INPUT_DIM, hidden_sizes=HIDDEN, dropout=0.0)
    torch.save({"model_state_dict": m.state_dict(), "input_dim": fc.INPUT_DIM}, legacy)

    mm = ModelManager()
    mm._lock = __import__("threading").RLock()
    mm._models = {}
    mm._model_metadata = {}
    with pytest.raises(Exception):
        mm.load_canonical_mlp(str(legacy), "predictive_maintenance_model", "1")
    assert mm._model_metadata["mlp"]["loaded"] is False
    assert mm._model_metadata["mlp"].get("unsafe_checkpoint_rejected") is True
