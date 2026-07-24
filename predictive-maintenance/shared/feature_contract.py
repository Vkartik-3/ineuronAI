"""
Canonical, versioned feature contract for the C-MAPSS RUL model.

This module is the SINGLE source of truth for:

  * which sensors are used, and in which order
  * which operating settings are used, and in which order
  * the lag / rolling-window / EMA parameters
  * the rolling standard-deviation ddof
  * the RUL cap
  * the ordered final feature names
  * the engineered feature dimension
  * a deterministic schema hash

Every producer and consumer of the 200-dimensional feature vector -- offline
training (``baseline_comparison``, ``TrainingPipeline``, ``RetrainingPipeline``),
online serving (``OnlineFeatureEngineer``, HTTP / Kafka / batch inference),
benchmarks, and checkpoint loading -- MUST import from here rather than
re-declaring its own constants.

Why this exists
---------------
Before this module, the offline pipeline derived its sensor order from pandas
DataFrame column order (numeric: sensor_2, sensor_3, ..., sensor_20, sensor_21)
while the online engineer used ``sorted()`` (lexical: sensor_11, sensor_12, ...,
sensor_2, sensor_20, ...). Both produced a 200-dimensional vector, so nothing
raised -- but all 14 sensor blocks were permuted at serving time and every
production prediction was computed on mis-assigned columns.

Dimension is DERIVED from the generated feature names, never asserted, so that
changing any parameter below cannot silently desynchronise the contract.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Schema version -- bump on ANY change to the parameters or ordering below.
# ---------------------------------------------------------------------------
FEATURE_SCHEMA_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Sensors
#
# ALL_SENSORS is declared in explicit NUMERIC order. This mirrors the column
# order produced by CMAPSSLoader and therefore the order used by the offline
# feature builder. Never re-derive this with sorted() -- lexical ordering puts
# sensor_11 before sensor_2 and silently permutes the feature vector.
# ---------------------------------------------------------------------------
ALL_SENSORS: Tuple[str, ...] = tuple(f"sensor_{i}" for i in range(1, 22))

# Near-zero variance across C-MAPSS FD001; excluded at training time.
CONSTANT_SENSORS: frozenset = frozenset(
    {
        "sensor_1",
        "sensor_5",
        "sensor_6",
        "sensor_10",
        "sensor_16",
        "sensor_18",
        "sensor_19",
    }
)

# Ordered, non-constant sensors -- numeric order, derived from ALL_SENSORS.
SENSORS: Tuple[str, ...] = tuple(s for s in ALL_SENSORS if s not in CONSTANT_SENSORS)

# ---------------------------------------------------------------------------
# Operating settings and the cycle column
# ---------------------------------------------------------------------------
OP_SETTINGS: Tuple[str, ...] = ("op_setting_1", "op_setting_2", "op_setting_3")
CYCLE_KEY: str = "time_cycle"

# ---------------------------------------------------------------------------
# Temporal feature-engineering parameters
# ---------------------------------------------------------------------------
LAGS: Tuple[int, ...] = (1, 3, 5, 10)
WINDOWS: Tuple[int, ...] = (5, 10, 20)
EMA_ALPHAS: Tuple[float, ...] = (0.1, 0.3, 0.5)

# pandas ``rolling().std()`` defaults to ddof=1 (Bessel's correction). The
# online engineer must pass ddof=1 explicitly to numpy, whose default is 0.
ROLLING_DDOF: int = 1

# Standard piecewise-linear RUL cap from the PHM literature.
RUL_CAP: int = 125

# Canonical online history length. Readings older than this are discarded
# before feature engineering, so HTTP, Kafka and batch paths all compute
# features over an identical window. The largest rolling window is 20 and the
# largest lag is 10, so 50 readings fully saturate every temporal feature.
MAX_ONLINE_SEQUENCE_LENGTH: int = 50


def _ema_suffix(alpha: float) -> str:
    """Match the offline builder's naming: 0.1 -> '01', 0.3 -> '03'."""
    return str(alpha).replace(".", "")


def build_feature_names(
    sensors: Sequence[str] = SENSORS,
    op_settings: Sequence[str] = OP_SETTINGS,
    lags: Sequence[int] = LAGS,
    windows: Sequence[int] = WINDOWS,
    ema_alphas: Sequence[float] = EMA_ALPHAS,
) -> List[str]:
    """
    Generate the ordered final feature names.

    The order below reproduces, exactly, the dict-insertion order of the
    offline builder (``baseline_comparison._engineer_features``), which is what
    ``DataFrame(cols)`` turns into column order:

      1. raw sensor values      (all sensors)
      2. raw operating settings (all op settings)
      3. lag differences        (outer = sensor, inner = lag)
      4. rolling mean + std     (outer = sensor, inner = window)
      5. EMA values             (outer = sensor, inner = alpha)
      6. time_cycle

    The ``rul`` column is dropped by the offline builder and is not a feature.
    """
    names: List[str] = []

    # 1 + 2 -- raw block
    names.extend(sensors)
    names.extend(op_settings)

    # 3 -- lag differences
    for col in sensors:
        for lag in lags:
            names.append(f"{col}_lag{lag}")

    # 4 -- rolling statistics
    for col in sensors:
        for w in windows:
            names.append(f"{col}_roll{w}_mean")
            names.append(f"{col}_roll{w}_std")

    # 5 -- exponential moving averages
    for col in sensors:
        for alpha in ema_alphas:
            names.append(f"{col}_ema{_ema_suffix(alpha)}")

    # 6 -- operational age
    names.append(CYCLE_KEY)

    return names


#: Ordered final feature names for the current contract.
FEATURE_NAMES: Tuple[str, ...] = tuple(build_feature_names())

#: Engineered feature dimension -- DERIVED from the names, never asserted.
#: For FD001: 14 sensors x (1 raw + 4 lag + 6 rolling + 3 EMA) + 3 op + 1 cycle = 200
INPUT_DIM: int = len(FEATURE_NAMES)

FEATURES_PER_SENSOR: int = 1 + len(LAGS) + 2 * len(WINDOWS) + len(EMA_ALPHAS)


def contract_payload() -> Dict:
    """The canonical, JSON-serialisable description of the contract."""
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "sensors": list(SENSORS),
        "constant_sensors": sorted(CONSTANT_SENSORS),
        "op_settings": list(OP_SETTINGS),
        "cycle_key": CYCLE_KEY,
        "lags": list(LAGS),
        "windows": list(WINDOWS),
        "ema_alphas": list(EMA_ALPHAS),
        "rolling_ddof": ROLLING_DDOF,
        "rul_cap": RUL_CAP,
        "max_online_sequence_length": MAX_ONLINE_SEQUENCE_LENGTH,
        "feature_names": list(FEATURE_NAMES),
        "input_dim": INPUT_DIM,
    }


def schema_hash() -> str:
    """
    Deterministic SHA-256 over the full contract payload.

    Changing a lag, window, EMA alpha, sensor list, ddof, RUL cap, or the
    feature order changes this hash. Checkpoints embed it; model loading
    compares it against the serving contract and refuses a mismatch.
    """
    blob = json.dumps(contract_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


#: Hash of the current contract.
FEATURE_SCHEMA_HASH: str = schema_hash()


class FeatureContractMismatch(ValueError):
    """Raised when an artifact's feature contract disagrees with the serving contract."""


def validate_against_contract(
    checkpoint_meta: Dict,
    *,
    strict: bool = True,
) -> None:
    """
    Compare a checkpoint's embedded feature contract with the serving contract.

    Args:
        checkpoint_meta: mapping that should carry ``feature_schema_hash``,
            ``feature_schema_version``, ``input_dim`` and ``feature_names``.
        strict: when True (the production default) any disagreement raises.
            When False the checkpoint predates the contract and only the
            dimension is checked -- callers must surface this as degraded.

    Raises:
        FeatureContractMismatch: on any semantic disagreement.
    """
    ckpt_hash = checkpoint_meta.get("feature_schema_hash")
    ckpt_dim = checkpoint_meta.get("input_dim")
    ckpt_names = checkpoint_meta.get("feature_names")

    if ckpt_hash is None:
        if strict:
            raise FeatureContractMismatch(
                "Checkpoint carries no 'feature_schema_hash'. It predates the "
                "canonical feature contract and cannot be verified as safe to "
                "serve. Retrain with the current pipeline, or load with "
                "strict=False and treat the model as degraded."
            )
        if ckpt_dim is not None and int(ckpt_dim) != INPUT_DIM:
            raise FeatureContractMismatch(
                f"Legacy checkpoint input_dim={ckpt_dim} != serving "
                f"input_dim={INPUT_DIM}; refusing to pad or truncate."
            )
        return

    if ckpt_hash != FEATURE_SCHEMA_HASH:
        detail = ""
        if ckpt_names is not None and list(ckpt_names) != list(FEATURE_NAMES):
            diff = [
                (i, a, b)
                for i, (a, b) in enumerate(zip(ckpt_names, FEATURE_NAMES))
                if a != b
            ]
            detail = (
                f" First differing position: {diff[0]}." if diff else
                f" Length differs: checkpoint={len(ckpt_names)} serving={len(FEATURE_NAMES)}."
            )
        raise FeatureContractMismatch(
            f"Feature schema hash mismatch: checkpoint={ckpt_hash} "
            f"serving={FEATURE_SCHEMA_HASH} "
            f"(serving version {FEATURE_SCHEMA_VERSION}).{detail} "
            "Refusing to serve -- predictions would be computed on "
            "mis-assigned feature columns."
        )

    if ckpt_dim is not None and int(ckpt_dim) != INPUT_DIM:
        raise FeatureContractMismatch(
            f"Checkpoint input_dim={ckpt_dim} disagrees with contract "
            f"input_dim={INPUT_DIM} despite matching hash -- corrupt artifact."
        )
