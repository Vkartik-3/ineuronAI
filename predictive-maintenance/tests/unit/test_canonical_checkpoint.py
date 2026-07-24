"""Canonical checkpoint round-trip, validation, and scaler-preservation tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

_PM_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PM_ROOT), str(_PM_ROOT / "ml_pipeline" / "train")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

torch = pytest.importorskip("torch")

from sklearn.preprocessing import StandardScaler  # noqa: E402

from shared import feature_contract as fc  # noqa: E402
from shared.checkpoint import (  # noqa: E402
    CHECKPOINT_FORMAT_VERSION,
    InvalidCheckpointError,
    apply_retention,
    build_checkpoint_payload,
    load_canonical_checkpoint,
    save_canonical_checkpoint,
)
from shared.checkpoint import _load_mlp_class  # noqa: E402

MLPRULNet = _load_mlp_class()

HIDDEN = [32, 16]


def _fitted_scaler(width=fc.INPUT_DIM, n=64):
    rng = np.random.default_rng(0)
    return StandardScaler().fit(rng.normal(size=(n, width)))


def _model(input_dim=fc.INPUT_DIM):
    torch.manual_seed(0)
    m = MLPRULNet(input_dim=input_dim, hidden_sizes=HIDDEN, dropout=0.1)
    m.eval()
    return m


def _payload(model=None, scaler=None):
    return build_checkpoint_payload(
        model=model or _model(),
        scaler=scaler if scaler is not None else _fitted_scaler(),
        config={"mlp": {"architecture": {"hidden_sizes": HIDDEN, "dropout": 0.1}}},
        hidden_sizes=HIDDEN,
        dropout=0.1,
        dataset_id="FD001",
        training_run_id="run-abc",
        training_seed=42,
        training_metrics={"mae": 9.4},
        evaluation_protocol="all_test_rows",
    )


@pytest.mark.unit
class TestCheckpointRoundTrip:
    def test_round_trip_preserves_predictions_scaler_and_schema(self, tmp_path):
        model = _model()
        scaler = _fitted_scaler()
        path = tmp_path / "ckpt.pt"
        sha = save_canonical_checkpoint(_payload(model, scaler), path)

        loaded = load_canonical_checkpoint(path, expected_sha256=sha)

        assert loaded.checkpoint_format_version == CHECKPOINT_FORMAT_VERSION
        assert loaded.input_dim == fc.INPUT_DIM
        assert loaded.feature_schema_hash == fc.FEATURE_SCHEMA_HASH
        assert loaded.feature_names == list(fc.FEATURE_NAMES)
        assert loaded.training_run_id == "run-abc"
        assert loaded.training_seed == 42

        # Scaler preserved and identical.
        np.testing.assert_allclose(loaded.scaler.mean_, scaler.mean_)
        np.testing.assert_allclose(loaded.scaler.scale_, scaler.scale_)

        # Predictions identical before/after serialization.
        rng = np.random.default_rng(1)
        X = rng.normal(size=(5, fc.INPUT_DIM)).astype(np.float32)
        Xs = scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            before = model(torch.tensor(Xs)).numpy()
            after = loaded.model(torch.tensor(loaded.transform(X))).numpy()
        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)

    def test_hash_mismatch_rejected(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_canonical_checkpoint(_payload(), path)
        with pytest.raises(InvalidCheckpointError, match="hash mismatch"):
            load_canonical_checkpoint(path, expected_sha256="0" * 64)

    def test_save_is_atomic_no_partial_file_on_error(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        bad = _payload()
        bad["unpicklable"] = lambda x: x  # lambdas are not picklable -> save fails
        with pytest.raises(Exception):
            save_canonical_checkpoint(bad, path)
        assert not path.exists()
        assert not list(tmp_path.glob(".*tmp*"))


@pytest.mark.unit
class TestCheckpointValidation:
    def test_unfitted_scaler_rejected_at_build(self):
        with pytest.raises(InvalidCheckpointError, match="FITTED scaler"):
            _payload(scaler=StandardScaler())

    def test_none_scaler_rejected_at_build(self):
        with pytest.raises(InvalidCheckpointError, match="FITTED scaler"):
            build_checkpoint_payload(
                model=_model(), scaler=None,
                config={}, hidden_sizes=HIDDEN, dropout=0.1, dataset_id="FD001",
            )

    def test_wrong_scaler_width_rejected_at_build(self):
        with pytest.raises(InvalidCheckpointError, match="width"):
            _payload(scaler=_fitted_scaler(width=fc.INPUT_DIM - 1))

    def test_wrong_model_input_dim_rejected_at_build(self):
        with pytest.raises(InvalidCheckpointError, match="input dim"):
            _payload(model=_model(input_dim=128), scaler=_fitted_scaler(width=128))

    def test_legacy_checkpoint_without_format_version_rejected(self, tmp_path):
        path = tmp_path / "legacy.pt"
        torch.save({"model_state_dict": _model().state_dict()}, path)
        with pytest.raises(InvalidCheckpointError, match="not a canonical checkpoint"):
            load_canonical_checkpoint(path)

    def test_scaler_stripped_after_save_rejected_on_load(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        payload = _payload()
        save_canonical_checkpoint(payload, path)
        # Simulate a corrupted artifact whose scaler was dropped.
        blob = torch.load(path, weights_only=False)
        blob["scaler"] = None
        torch.save(blob, path)
        with pytest.raises(InvalidCheckpointError, match="no fitted scaler"):
            load_canonical_checkpoint(path)


@pytest.mark.unit
class TestRetention:
    def test_keeps_only_n_most_recent(self, tmp_path):
        for i in range(7):
            p = tmp_path / f"m{i}.pt"
            p.write_bytes(b"x")
            import os
            os.utime(p, (i, i))  # deterministic mtime ordering
        removed = apply_retention(tmp_path, keep=3)
        remaining = sorted(p.name for p in tmp_path.glob("*.pt"))
        assert len(remaining) == 3
        assert remaining == ["m4.pt", "m5.pt", "m6.pt"]
        assert len(removed) == 4
