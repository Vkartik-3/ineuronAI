"""
Canonical model checkpoint format for the C-MAPSS RUL MLP.

This is the ONE serving artifact. A checkpoint carries everything needed to
reproduce the exact model input the network was trained on:

  * the network weights and architecture
  * the fitted final-feature StandardScaler
  * the feature contract (version, hash, ordered names, input dim)
  * training provenance (run id, seed, dataset, commit, metrics, protocol)

Why this exists
---------------
The previous artifact (``MLPRULPredictor.save_model``) stored a scaler but no
feature contract, and nothing validated that the scaler was fitted, that its
width matched the model input, or that the feature ordering the model was
trained on matched the ordering serving produces. A checkpoint could load
"successfully" and still produce systematically wrong predictions.

Loading is strict by default: any disagreement between the artifact and the
serving feature contract raises rather than degrading silently. Padding or
truncating a feature vector to make dimensions line up is never performed.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from shared.feature_contract import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION,
    INPUT_DIM,
    FeatureContractMismatch,
    validate_against_contract,
)

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT_VERSION = "1.0.0"

#: Key of the first Linear layer's weight in MLPRULNet -- used to recover the
#: true input dimension from the state dict rather than trusting metadata.
_FIRST_LAYER_WEIGHT_KEY = "network.0.block.0.weight"


class InvalidCheckpointError(ValueError):
    """A checkpoint is structurally invalid or unsafe to serve."""


def code_commit() -> str:
    """Current git commit, or 'unknown' outside a work tree."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a completed file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_mlp_class():
    """
    Import MLPRULNet by explicit file path.

    ``inference_service/models`` (which has no ``mlp_model``) and
    ``ml_pipeline/train/models`` are BOTH named ``models``; whichever is on
    ``sys.path`` first shadows the other, so a plain ``from models.mlp_model
    import ...`` is order-dependent and breaks inside the inference service.
    Loading by file location sidesteps the collision.
    """
    import importlib.util

    mlp_path = (
        Path(__file__).resolve().parent.parent
        / "ml_pipeline" / "train" / "models" / "mlp_model.py"
    )
    spec = importlib.util.spec_from_file_location("_canonical_mlp_model", mlp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MLPRULNet


def _scaler_is_fitted(scaler: Any) -> bool:
    return scaler is not None and hasattr(scaler, "mean_") and scaler.mean_ is not None


@dataclass(frozen=True)
class CanonicalCheckpoint:
    """A validated, loaded checkpoint. Immutable."""

    model: Any
    scaler: Any
    input_dim: int
    hidden_sizes: List[int]
    dropout: float
    feature_names: List[str]
    feature_schema_hash: str
    feature_schema_version: str
    checkpoint_format_version: str
    dataset_id: str
    training_run_id: Optional[str]
    training_seed: Optional[int]
    training_metrics: Dict[str, float]
    evaluation_protocol: str
    created_at: str
    code_commit: str
    checkpoint_path: str
    checkpoint_sha256: str
    raw: Dict[str, Any] = field(repr=False, default_factory=dict)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the persisted scaler exactly once.

        This is the ONLY place serving is permitted to scale features. The
        scaler is never refitted.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise InvalidCheckpointError(
                f"Expected (n, {self.input_dim}) features, got {X.shape}. "
                "Refusing to pad or truncate."
            )
        return self.scaler.transform(X).astype(np.float32)


def build_checkpoint_payload(
    *,
    model,
    scaler,
    config: Dict[str, Any],
    hidden_sizes: List[int],
    dropout: float,
    dataset_id: str,
    training_run_id: Optional[str] = None,
    training_seed: Optional[int] = None,
    training_metrics: Optional[Dict[str, float]] = None,
    evaluation_protocol: str = "unspecified",
    history: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """Assemble the canonical payload, validating it is safe to persist."""
    state_dict = model.state_dict()

    if _FIRST_LAYER_WEIGHT_KEY not in state_dict:
        raise InvalidCheckpointError(
            f"State dict has no {_FIRST_LAYER_WEIGHT_KEY!r}; architecture is not "
            f"the expected MLPRULNet. Keys: {list(state_dict)[:5]}"
        )
    true_input_dim = int(state_dict[_FIRST_LAYER_WEIGHT_KEY].shape[1])

    if true_input_dim != INPUT_DIM:
        raise InvalidCheckpointError(
            f"Model input dim {true_input_dim} != feature contract {INPUT_DIM}. "
            "Refusing to persist an artifact that cannot be served."
        )

    if not _scaler_is_fitted(scaler):
        raise InvalidCheckpointError(
            "Refusing to persist a checkpoint without a FITTED scaler. Serving "
            "applies the scaler, so an unscaled or unfitted artifact would feed "
            "the network a distribution it never saw during training."
        )

    for attr in ("mean_", "scale_"):
        width = len(getattr(scaler, attr))
        if width != INPUT_DIM:
            raise InvalidCheckpointError(
                f"scaler.{attr} has width {width}, expected {INPUT_DIM}."
            )

    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": state_dict,
        "model_config": config,
        "input_dim": true_input_dim,
        "hidden_sizes": list(hidden_sizes),
        "dropout": float(dropout),
        "scaler": scaler,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "feature_names": list(FEATURE_NAMES),
        "dataset_id": dataset_id,
        "training_run_id": training_run_id,
        "training_seed": training_seed,
        "training_metrics": dict(training_metrics or {}),
        "evaluation_protocol": evaluation_protocol,
        "history": history or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit(),
    }


def save_canonical_checkpoint(payload: Dict[str, Any], filepath: str | Path) -> str:
    """
    Write a checkpoint atomically and return its SHA-256.

    temp file in the destination directory -> flush -> fsync -> atomic rename.
    A reader therefore never observes a partially written checkpoint, and a
    crash mid-write cannot corrupt an existing good artifact.
    """
    import torch

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(filepath.parent), prefix=f".{filepath.name}.", suffix=".tmp"
    )
    os.close(fd)
    try:
        with open(tmp_path, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, filepath)  # atomic within a filesystem
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    digest = file_sha256(filepath)
    logger.info("Canonical checkpoint written: %s (sha256=%s)", filepath, digest[:16])
    return digest


def load_canonical_checkpoint(
    filepath: str | Path,
    *,
    strict: bool = True,
    device: Optional[str] = None,
    expected_sha256: Optional[str] = None,
) -> CanonicalCheckpoint:
    """
    Load and fully validate a canonical checkpoint.

    Validates, in order: file integrity, format version, architecture, true
    input dim from the state dict, feature contract, fitted scaler, and scaler
    width. Any failure raises -- there is no degraded "load anyway" path when
    strict=True.
    """
    import torch

    MLPRULNet = _load_mlp_class()

    filepath = Path(filepath)
    if not filepath.exists():
        raise InvalidCheckpointError(f"Checkpoint not found: {filepath}")

    digest = file_sha256(filepath)
    if expected_sha256 is not None and digest != expected_sha256:
        raise InvalidCheckpointError(
            f"Checkpoint hash mismatch for {filepath}: expected {expected_sha256}, "
            f"got {digest}. Artifact is corrupt or was substituted."
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(filepath, map_location=device, weights_only=False)

    fmt = ckpt.get("checkpoint_format_version")
    if fmt is None:
        raise InvalidCheckpointError(
            f"{filepath} is not a canonical checkpoint (no "
            "'checkpoint_format_version'). It predates the canonical format and "
            "carries no feature contract, so it cannot be verified safe to serve."
        )
    if fmt != CHECKPOINT_FORMAT_VERSION:
        raise InvalidCheckpointError(
            f"Unsupported checkpoint_format_version {fmt!r} "
            f"(this build reads {CHECKPOINT_FORMAT_VERSION!r})."
        )

    # Feature contract -- compares hash AND names, never dimension alone.
    validate_against_contract(ckpt, strict=strict)

    state_dict = ckpt["model_state_dict"]
    if _FIRST_LAYER_WEIGHT_KEY not in state_dict:
        raise InvalidCheckpointError(
            f"State dict missing {_FIRST_LAYER_WEIGHT_KEY!r} -- not an MLPRULNet."
        )
    true_input_dim = int(state_dict[_FIRST_LAYER_WEIGHT_KEY].shape[1])
    declared_dim = int(ckpt["input_dim"])
    if true_input_dim != declared_dim:
        raise InvalidCheckpointError(
            f"Checkpoint declares input_dim={declared_dim} but its weights expect "
            f"{true_input_dim}. Corrupt artifact."
        )
    if true_input_dim != INPUT_DIM:
        raise InvalidCheckpointError(
            f"Checkpoint weights expect {true_input_dim} features; the serving "
            f"contract produces {INPUT_DIM}. Refusing to pad or truncate."
        )

    scaler = ckpt.get("scaler")
    if not _scaler_is_fitted(scaler):
        raise InvalidCheckpointError(
            f"{filepath} has no fitted scaler. Serving applies the scaler, so "
            "loading this artifact would feed the network unscaled features. "
            "Retrain with the canonical pipeline."
        )
    for attr in ("mean_", "scale_"):
        width = len(getattr(scaler, attr))
        if width != INPUT_DIM:
            raise InvalidCheckpointError(
                f"scaler.{attr} width {width} != contract {INPUT_DIM}."
            )

    hidden_sizes = list(ckpt["hidden_sizes"])
    dropout = float(ckpt["dropout"])
    model = MLPRULNet(
        input_dim=true_input_dim, hidden_sizes=hidden_sizes, dropout=dropout
    ).to(device)
    # strict=True: refuse a state dict that does not match the architecture.
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return CanonicalCheckpoint(
        model=model,
        scaler=scaler,
        input_dim=true_input_dim,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        feature_names=list(ckpt["feature_names"]),
        feature_schema_hash=ckpt["feature_schema_hash"],
        feature_schema_version=ckpt["feature_schema_version"],
        checkpoint_format_version=fmt,
        dataset_id=ckpt.get("dataset_id", "unknown"),
        training_run_id=ckpt.get("training_run_id"),
        training_seed=ckpt.get("training_seed"),
        training_metrics=dict(ckpt.get("training_metrics", {})),
        evaluation_protocol=ckpt.get("evaluation_protocol", "unspecified"),
        created_at=ckpt.get("created_at", "unknown"),
        code_commit=ckpt.get("code_commit", "unknown"),
        checkpoint_path=str(filepath),
        checkpoint_sha256=digest,
        raw=ckpt,
    )


def apply_retention(directory: str | Path, keep: int = 5, pattern: str = "*.pt") -> List[str]:
    """
    Delete all but the ``keep`` most recent checkpoints. Returns removed paths.

    Local disk is treated as a cache, not durable storage -- the durable copy is
    the MLflow artifact. On ECS the container filesystem is ephemeral.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    removed = []
    for stale in files[keep:]:
        stale.unlink()
        removed.append(str(stale))
        logger.info("Retention: removed stale checkpoint %s", stale)
    return removed
