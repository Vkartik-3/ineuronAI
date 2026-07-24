"""
Online Feature Engineering for MLP RUL Inference.

Reproduces the offline ``_engineer_features()`` builder in
``ml_pipeline/evaluate/baseline_comparison.py`` exactly, so that serving
features match the features the model was trained on.

Both sides derive every parameter -- sensor list, sensor ORDER, op settings,
lags, windows, EMA alphas, rolling ddof, feature order -- from the single
canonical contract in ``shared.feature_contract``. Nothing is re-declared here.

Historical defect this replaces
-------------------------------
This module previously derived its sensor order with ``sorted()``, which is
lexical: sensor_11, sensor_12, ..., sensor_2, sensor_20, sensor_3, ...
The offline builder used pandas column order, which is numeric: sensor_2,
sensor_3, ..., sensor_20, sensor_21. All 14 sensor positions were permuted.
Both paths produced a 200-wide vector, so no dimension check could catch it and
every served prediction used mis-assigned columns.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np

from shared.feature_contract import (
    CONSTANT_SENSORS,
    CYCLE_KEY,
    EMA_ALPHAS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION,
    INPUT_DIM,
    LAGS,
    MAX_ONLINE_SEQUENCE_LENGTH,
    OP_SETTINGS,
    ROLLING_DDOF,
    SENSORS,
    WINDOWS,
)

logger = logging.getLogger(__name__)

# Re-exported for backwards compatibility with existing importers. These are
# now views onto the canonical contract, not independent declarations.
ENGINEERED_FEATURE_DIM = INPUT_DIM

__all__ = [
    "CONSTANT_SENSORS",
    "ENGINEERED_FEATURE_DIM",
    "FEATURE_NAMES",
    "INPUT_DIM",
    "MissingSensorError",
    "OnlineFeatureEngineer",
    "SENSORS",
    "online_feature_engineer",
]


class MissingSensorError(ValueError):
    """A reading is missing sensors the model requires.

    Never silently defaulted to 0.0: a zero in a standardised sensor channel is
    a legitimate mid-range value, so padding a missing sensor produces a
    confidently wrong prediction rather than a visible failure.
    """


class OnlineFeatureEngineer:
    """
    Computes, at serving time, the same lag / rolling / EMA feature vector the
    offline pipeline computes at training time.

    Accepts the full sensor sequence (oldest reading first). Sequences longer
    than ``MAX_ONLINE_SEQUENCE_LENGTH`` are trimmed to their most recent
    ``MAX_ONLINE_SEQUENCE_LENGTH`` readings so that HTTP, Kafka and batch paths
    all compute features over an identical window.

    Returns a ``(1, INPUT_DIM)`` float32 array ready for the MLP forward pass.
    """

    feature_names: Tuple[str, ...] = FEATURE_NAMES
    input_dim: int = INPUT_DIM
    schema_hash: str = FEATURE_SCHEMA_HASH
    schema_version: str = FEATURE_SCHEMA_VERSION

    def engineer(
        self,
        sequence: List[Dict[str, float]],
        equipment_id: str = "",
        *,
        require_full_history: bool = False,
    ) -> np.ndarray:
        """
        Args:
            sequence: sensor-reading dicts, oldest first. Each must contain
                every sensor in the contract; op settings and ``time_cycle``
                are optional and default as documented below.
            equipment_id: used only for logging.
            require_full_history: when True, sequences too short to saturate
                the largest lag/window raise instead of degrading.

        Returns:
            np.ndarray of shape ``(1, INPUT_DIM)``, dtype float32, whose column
            order is exactly ``FEATURE_NAMES``.

        Raises:
            ValueError: empty sequence.
            MissingSensorError: a required sensor is absent from a reading.
        """
        if not sequence:
            raise ValueError(
                f"Empty sensor sequence for equipment {equipment_id!r} -- cannot "
                "engineer features. The caller must supply at least one reading."
            )

        if len(sequence) > MAX_ONLINE_SEQUENCE_LENGTH:
            sequence = sequence[-MAX_ONLINE_SEQUENCE_LENGTH:]

        min_required = max(max(LAGS) + 1, max(WINDOWS))
        if len(sequence) < min_required:
            if require_full_history:
                raise ValueError(
                    f"Sequence for {equipment_id!r} has {len(sequence)} readings; "
                    f"{min_required} are required to saturate every temporal "
                    "feature (max lag 10, max window 20)."
                )
            logger.warning(
                "Short sequence for %s: %d readings (< %d). Lag features beyond "
                "the available history are 0 and rolling windows are partial, "
                "matching the offline min_periods=1 / diff().fillna(0) semantics.",
                equipment_id,
                len(sequence),
                min_required,
            )

        return self._build_feature_vector(sequence, equipment_id)

    # ------------------------------------------------------------------ #

    def _build_feature_vector(
        self, sequence: List[Dict[str, float]], equipment_id: str
    ) -> np.ndarray:
        n = len(sequence)

        sensor_arrays = self._extract_series(sequence, SENSORS, equipment_id, required=True)
        op_arrays = self._extract_series(sequence, OP_SETTINGS, equipment_id, required=False)

        # time_cycle defaults to a 1-based positional index when absent, which
        # matches how the offline loader numbers cycles for a fresh engine.
        cycle_array = np.array(
            [float(r.get(CYCLE_KEY, i + 1)) for i, r in enumerate(sequence)],
            dtype=np.float64,
        )

        features: List[float] = []

        # Block 1 -- raw sensor values, contract order.
        for k in SENSORS:
            features.append(float(sensor_arrays[k][-1]))

        # Block 2 -- raw operating settings, contract order.
        for k in OP_SETTINGS:
            features.append(float(op_arrays[k][-1]))

        # Block 3 -- lag differences (outer = sensor, inner = lag).
        # Offline: grp[col].diff(lag).fillna(0) evaluated at the final row.
        for k in SENSORS:
            s = sensor_arrays[k]
            for lag in LAGS:
                features.append(float(s[-1] - s[-(lag + 1)]) if n > lag else 0.0)

        # Block 4 -- rolling mean/std (outer = sensor, inner = window).
        # Offline: rolling(w, min_periods=1).mean() / .std().fillna(0) at the
        # final row. pandas' .std() uses ddof=1; numpy defaults to ddof=0, so
        # ROLLING_DDOF is passed explicitly.
        for k in SENSORS:
            s = sensor_arrays[k]
            for w in WINDOWS:
                window = s[max(0, n - w):]
                features.append(float(np.mean(window)))
                features.append(
                    float(np.std(window, ddof=ROLLING_DDOF)) if len(window) > ROLLING_DDOF else 0.0
                )

        # Block 5 -- EMA (outer = sensor, inner = alpha).
        for k in SENSORS:
            s = sensor_arrays[k]
            for alpha in EMA_ALPHAS:
                features.append(_ema_last(s, alpha))

        # Block 6 -- operational age.
        features.append(float(cycle_array[-1]))

        vec = np.asarray(features, dtype=np.float32).reshape(1, -1)

        if vec.shape[1] != INPUT_DIM:
            raise AssertionError(
                f"Built {vec.shape[1]} features but the contract declares "
                f"{INPUT_DIM}. feature_engineering.py and shared.feature_contract "
                "have diverged."
            )
        if not np.all(np.isfinite(vec)):
            bad = [FEATURE_NAMES[i] for i in np.where(~np.isfinite(vec[0]))[0]]
            raise ValueError(
                f"Non-finite engineered features for {equipment_id!r}: {bad[:10]}"
            )
        return vec

    @staticmethod
    def _extract_series(
        sequence: Sequence[Dict[str, float]],
        keys: Sequence[str],
        equipment_id: str,
        *,
        required: bool,
    ) -> Dict[str, np.ndarray]:
        """Build per-key float64 time series, validating presence and finiteness."""
        if required:
            missing = sorted(set(keys) - set(sequence[-1]))
            if missing:
                raise MissingSensorError(
                    f"Reading for equipment {equipment_id!r} is missing required "
                    f"sensors {missing}. Refusing to substitute 0.0 -- a zero is a "
                    "valid standardised sensor value and would yield a silently "
                    "wrong prediction."
                )

        out: Dict[str, np.ndarray] = {}
        for k in keys:
            values = []
            for r in sequence:
                v = r.get(k)
                if v is None:
                    if required:
                        raise MissingSensorError(
                            f"Sensor {k!r} missing from a reading for "
                            f"equipment {equipment_id!r}."
                        )
                    v = 0.0
                values.append(float(v))
            arr = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"Non-finite values in {k!r} for equipment {equipment_id!r}."
                )
            out[k] = arr
        return out


def _ema_last(series: np.ndarray, alpha: float) -> float:
    """
    EMA(alpha, adjust=False) over a 1-D array, returning the final value.

    Matches ``s.ewm(alpha=alpha, adjust=False).mean().iloc[-1]``:
        ema[0] = series[0]
        ema[t] = (1 - alpha) * ema[t-1] + alpha * series[t]

    Note: because the recursion is seeded at ``series[0]``, trimming a sequence
    to MAX_ONLINE_SEQUENCE_LENGTH leaves a residual weight of (1-alpha)^n on the
    initial condition relative to a full-history offline EMA. For n=50 and the
    smallest contract alpha (0.1) that residual is (0.9)^50 = 0.5%. Tests
    quantify this bound rather than assuming it away.
    """
    if series.size == 0:
        return 0.0
    ema = float(series[0])
    one_minus = 1.0 - alpha
    for v in series[1:]:
        ema = one_minus * ema + alpha * float(v)
    return ema


# Module-level singleton -- stateless, created once per process.
online_feature_engineer = OnlineFeatureEngineer()
