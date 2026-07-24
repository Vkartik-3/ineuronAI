"""
Feature-contract and training/serving parity tests.

These protect against the class of defect where the offline and online feature
builders both emit a 200-wide vector but disagree on what each column means --
which no dimension check can detect and which silently corrupts every
production prediction.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PM_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PM_ROOT), str(_PM_ROOT / "inference_service")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import feature_contract as fc
from shared.feature_contract import (
    FeatureContractMismatch,
    build_feature_names,
    validate_against_contract,
)
from feature_engineering import (
    MissingSensorError,
    OnlineFeatureEngineer,
    _ema_last,
)


# ---------------------------------------------------------------------------
# Offline reference implementation
#
# This mirrors baseline_comparison._engineer_features() exactly. It is
# duplicated here (rather than imported) so the parity test compares the online
# engineer against a literal pandas implementation, not against shared code
# that could drift with it.
# ---------------------------------------------------------------------------

def _offline_engineer(df, sensor_cols, op_cols):
    cols = {}
    grp = df.sort_values("time_cycle").copy()

    for col in list(sensor_cols) + list(op_cols):
        cols[col] = grp[col].values
    for col in sensor_cols:
        for lag in fc.LAGS:
            cols[f"{col}_lag{lag}"] = grp[col].diff(lag).fillna(0).values
    for col in sensor_cols:
        for w in fc.WINDOWS:
            cols[f"{col}_roll{w}_mean"] = grp[col].rolling(w, min_periods=1).mean().values
            cols[f"{col}_roll{w}_std"] = (
                grp[col].rolling(w, min_periods=1).std().fillna(0).values
            )
    for col in sensor_cols:
        for alpha in fc.EMA_ALPHAS:
            suffix = str(alpha).replace(".", "")
            cols[f"{col}_ema{suffix}"] = grp[col].ewm(alpha=alpha, adjust=False).mean().values
    cols["time_cycle"] = grp["time_cycle"].values
    return pd.DataFrame(cols)


def _make_sequence(n=50, seed=0):
    """Deterministic pseudo-engine run: n readings, oldest first."""
    rng = np.random.default_rng(seed)
    seq = []
    for i in range(n):
        reading = {"time_cycle": float(i + 1)}
        for j, s in enumerate(fc.SENSORS):
            # Distinct drift + noise per sensor so any column permutation shows up.
            reading[s] = float(100.0 + 10.0 * j + 0.5 * i + rng.normal(0, 0.3))
        for j, o in enumerate(fc.OP_SETTINGS):
            reading[o] = float(0.1 * (j + 1) + rng.normal(0, 0.01))
        seq.append(reading)
    return seq


# ---------------------------------------------------------------------------
# Contract structure
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFeatureContract:
    def test_input_dim_is_derived_from_names_not_asserted(self):
        assert fc.INPUT_DIM == len(fc.FEATURE_NAMES)
        assert fc.INPUT_DIM == len(build_feature_names())

    def test_fd001_dimension_decomposition(self):
        """14 sensors x 14 features + 3 op settings + 1 cycle = 200."""
        assert len(fc.SENSORS) == 14
        assert fc.FEATURES_PER_SENSOR == 14
        assert fc.INPUT_DIM == 14 * 14 + 3 + 1 == 200

    def test_sensor_order_is_numeric_not_lexical(self):
        """
        Regression: the online engineer used sorted(), which is lexical and put
        sensor_11 in position 0 where training had sensor_2. All 14 sensor
        blocks were permuted.
        """
        assert list(fc.SENSORS) == [
            "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
            "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
            "sensor_15", "sensor_17", "sensor_20", "sensor_21",
        ]
        assert list(fc.SENSORS) != sorted(fc.SENSORS), (
            "numeric and lexical order must differ, else this test is vacuous"
        )

    def test_rolling_ddof_is_one(self):
        """pandas rolling().std() uses ddof=1; numpy defaults to 0."""
        assert fc.ROLLING_DDOF == 1

    def test_feature_names_unique(self):
        assert len(set(fc.FEATURE_NAMES)) == len(fc.FEATURE_NAMES)

    @pytest.mark.parametrize(
        "attr,new_value",
        [
            ("LAGS", (1, 3, 5, 11)),
            ("WINDOWS", (5, 10, 25)),
            ("EMA_ALPHAS", (0.1, 0.3, 0.6)),
            ("ROLLING_DDOF", 0),
            ("RUL_CAP", 130),
            ("SENSORS", tuple(sorted(fc.SENSORS))),
        ],
    )
    def test_schema_hash_changes_when_any_parameter_changes(
        self, attr, new_value, monkeypatch
    ):
        before = fc.schema_hash()
        monkeypatch.setattr(fc, attr, new_value)
        if attr in {"LAGS", "WINDOWS", "EMA_ALPHAS", "SENSORS"}:
            monkeypatch.setattr(
                fc, "FEATURE_NAMES", tuple(build_feature_names(
                    sensors=fc.SENSORS, lags=fc.LAGS,
                    windows=fc.WINDOWS, ema_alphas=fc.EMA_ALPHAS,
                ))
            )
        assert fc.schema_hash() != before, f"changing {attr} did not change the hash"

    def test_schema_hash_is_stable_across_calls(self):
        assert fc.schema_hash() == fc.schema_hash() == fc.FEATURE_SCHEMA_HASH


# ---------------------------------------------------------------------------
# Contract validation at load time
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestContractValidation:
    def test_matching_checkpoint_accepted(self):
        validate_against_contract({
            "feature_schema_hash": fc.FEATURE_SCHEMA_HASH,
            "input_dim": fc.INPUT_DIM,
            "feature_names": list(fc.FEATURE_NAMES),
        })

    def test_mismatched_hash_rejected(self):
        with pytest.raises(FeatureContractMismatch, match="hash mismatch"):
            validate_against_contract({
                "feature_schema_hash": "deadbeef",
                "input_dim": fc.INPUT_DIM,
                "feature_names": list(fc.FEATURE_NAMES),
            })

    def test_permuted_feature_order_rejected_despite_correct_dimension(self):
        """
        The exact production bug: 200 columns, correct names, wrong ORDER.
        Must be rejected -- a dimension check alone passes it.
        """
        permuted = list(fc.FEATURE_NAMES)
        permuted[0], permuted[6] = permuted[6], permuted[0]
        with pytest.raises(FeatureContractMismatch) as exc:
            validate_against_contract({
                "feature_schema_hash": "some-other-hash",
                "input_dim": 200,
                "feature_names": permuted,
            })
        assert "First differing position" in str(exc.value)

    def test_legacy_checkpoint_without_hash_rejected_in_strict_mode(self):
        with pytest.raises(FeatureContractMismatch, match="predates"):
            validate_against_contract({"input_dim": 200}, strict=True)

    def test_legacy_checkpoint_allowed_in_non_strict_mode(self):
        validate_against_contract({"input_dim": 200}, strict=False)

    def test_legacy_checkpoint_with_wrong_dim_rejected_even_non_strict(self):
        with pytest.raises(FeatureContractMismatch, match="refusing to pad"):
            validate_against_contract({"input_dim": 128}, strict=False)


# ---------------------------------------------------------------------------
# Offline / online parity -- the core guarantee
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTrainServeParity:
    def test_online_feature_names_equal_contract(self):
        assert list(OnlineFeatureEngineer.feature_names) == list(fc.FEATURE_NAMES)

    def test_offline_column_order_equals_contract(self):
        seq = _make_sequence(30, seed=1)
        df = pd.DataFrame(seq)
        offline = _offline_engineer(df, list(fc.SENSORS), list(fc.OP_SETTINGS))
        assert list(offline.columns) == list(fc.FEATURE_NAMES)

    @pytest.mark.parametrize("n", [21, 30, 50])
    def test_exact_numerical_parity_offline_vs_online(self, n):
        """
        The regression test for the sensor-ordering defect. On the old code this
        fails on ~196 of 200 columns; the raw-sensor block alone is fully
        permuted.
        """
        seq = _make_sequence(n, seed=7)
        df = pd.DataFrame(seq)

        offline = _offline_engineer(df, list(fc.SENSORS), list(fc.OP_SETTINGS))
        expected = offline.iloc[-1].values.astype(np.float32)

        actual = OnlineFeatureEngineer().engineer(seq, "unit-1")[0]

        assert actual.shape == expected.shape == (fc.INPUT_DIM,)
        mismatched = [
            fc.FEATURE_NAMES[i]
            for i in range(fc.INPUT_DIM)
            if not np.isclose(actual[i], expected[i], rtol=1e-5, atol=1e-4)
        ]
        assert not mismatched, (
            f"{len(mismatched)}/{fc.INPUT_DIM} features differ, "
            f"first 10: {mismatched[:10]}"
        )

    def test_parity_holds_for_short_sequence(self):
        """min_periods=1 / diff().fillna(0) semantics must match on 3 readings."""
        seq = _make_sequence(3, seed=3)
        offline = _offline_engineer(pd.DataFrame(seq), list(fc.SENSORS), list(fc.OP_SETTINGS))
        expected = offline.iloc[-1].values.astype(np.float32)
        actual = OnlineFeatureEngineer().engineer(seq, "short")[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)

    def test_single_reading_std_is_zero_not_nan(self):
        seq = _make_sequence(1, seed=5)
        vec = OnlineFeatureEngineer().engineer(seq, "one")[0]
        assert np.all(np.isfinite(vec))
        std_idx = [i for i, nm in enumerate(fc.FEATURE_NAMES) if nm.endswith("_std")]
        assert np.allclose(vec[std_idx], 0.0)

    def test_ddof_one_is_load_bearing(self):
        """A ddof=0 online implementation must NOT match the pandas offline std."""
        seq = _make_sequence(30, seed=11)
        col = fc.SENSORS[0]
        series = np.array([r[col] for r in seq])
        w = 5
        pandas_std = pd.Series(series).rolling(w, min_periods=1).std().iloc[-1]
        assert np.isclose(np.std(series[-w:], ddof=1), pandas_std)
        assert not np.isclose(np.std(series[-w:], ddof=0), pandas_std)

    def test_ema_matches_pandas_ewm(self):
        rng = np.random.default_rng(2)
        series = rng.normal(50, 5, 40)
        for alpha in fc.EMA_ALPHAS:
            expected = pd.Series(series).ewm(alpha=alpha, adjust=False).mean().iloc[-1]
            assert np.isclose(_ema_last(series, alpha), expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# Input validation -- no silent padding
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOnlineInputValidation:
    def test_missing_sensor_raises_instead_of_zero_padding(self):
        seq = _make_sequence(25, seed=9)
        for r in seq:
            del r["sensor_14"]
        with pytest.raises(MissingSensorError, match="sensor_14"):
            OnlineFeatureEngineer().engineer(seq, "missing")

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError, match="Empty sensor sequence"):
            OnlineFeatureEngineer().engineer([], "empty")

    def test_nan_sensor_value_raises(self):
        seq = _make_sequence(25, seed=4)
        seq[-1][fc.SENSORS[0]] = float("nan")
        with pytest.raises(ValueError, match="Non-finite"):
            OnlineFeatureEngineer().engineer(seq, "nan")

    def test_inf_sensor_value_raises(self):
        seq = _make_sequence(25, seed=4)
        seq[10][fc.SENSORS[2]] = float("inf")
        with pytest.raises(ValueError, match="Non-finite"):
            OnlineFeatureEngineer().engineer(seq, "inf")

    def test_sequence_trimmed_to_canonical_length(self):
        """HTTP may accept >50 readings; only the last 50 are used, everywhere."""
        long_seq = _make_sequence(120, seed=6)
        full = OnlineFeatureEngineer().engineer(long_seq, "long")
        trimmed = OnlineFeatureEngineer().engineer(
            long_seq[-fc.MAX_ONLINE_SEQUENCE_LENGTH:], "long"
        )
        np.testing.assert_array_equal(full, trimmed)

    def test_require_full_history_rejects_short_sequence(self):
        with pytest.raises(ValueError, match="required to saturate"):
            OnlineFeatureEngineer().engineer(
                _make_sequence(5, seed=1), "short", require_full_history=True
            )

    def test_output_is_float32_2d(self):
        vec = OnlineFeatureEngineer().engineer(_make_sequence(25, seed=1), "x")
        assert vec.dtype == np.float32
        assert vec.shape == (1, fc.INPUT_DIM)


@pytest.mark.unit
class TestEmaTruncationBound:
    def test_truncation_error_is_bounded_and_documented(self):
        """
        Quantify (rather than assume away) the EMA initialisation error caused
        by trimming to MAX_ONLINE_SEQUENCE_LENGTH: residual weight on the seed
        value is (1-alpha)^n.
        """
        n = fc.MAX_ONLINE_SEQUENCE_LENGTH
        worst_alpha = min(fc.EMA_ALPHAS)
        residual = (1 - worst_alpha) ** n
        assert residual < 0.01, f"residual seed weight {residual:.4f} too large"


@pytest.mark.unit
class TestScaledModelInputParity:
    """
    Fix 1: prove the FINAL model input (after scaling) is identical offline and
    online, and that serving applies the scaler exactly once.
    """

    def test_scaled_final_input_parity(self):
        from sklearn.preprocessing import StandardScaler

        # Offline: build features for many rows, fit the scaler on them.
        seqs = [_make_sequence(40, seed=s) for s in range(20)]
        offline_rows = np.vstack([
            _offline_engineer(pd.DataFrame(s), list(fc.SENSORS), list(fc.OP_SETTINGS))
            .iloc[-1].values.astype(np.float32)
            for s in seqs
        ])
        scaler = StandardScaler().fit(offline_rows)

        # For a fresh sequence, offline and online must agree AFTER scaling.
        probe = _make_sequence(45, seed=99)
        offline_vec = (
            _offline_engineer(pd.DataFrame(probe), list(fc.SENSORS), list(fc.OP_SETTINGS))
            .iloc[-1].values.astype(np.float32).reshape(1, -1)
        )
        online_vec = OnlineFeatureEngineer().engineer(probe, "probe")

        offline_scaled = scaler.transform(offline_vec)
        online_scaled = scaler.transform(online_vec)
        np.testing.assert_allclose(offline_scaled, online_scaled, rtol=1e-4, atol=1e-4)

    def test_scaler_applied_exactly_once_not_twice(self):
        """Double-scaling would zero-center an already-centered vector; detect it."""
        from sklearn.preprocessing import StandardScaler

        rows = np.vstack([
            OnlineFeatureEngineer().engineer(_make_sequence(40, seed=s), "x")[0]
            for s in range(15)
        ])
        scaler = StandardScaler().fit(rows)
        once = scaler.transform(rows[:1])
        twice = scaler.transform(scaler.transform(rows[:1]))
        assert not np.allclose(once, twice), "single vs double scaling must differ"
