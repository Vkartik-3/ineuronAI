"""Unit tests for LabelGenerator — RUL computation and health classification."""

import pytest
import numpy as np
import pandas as pd

from features.label_generator import LabelGenerator


def _make_engine_df(equipment_id: str, max_cycle: int) -> pd.DataFrame:
    """Helper: build a simple single-engine DataFrame."""
    return pd.DataFrame(
        {
            "equipment_id": [equipment_id] * max_cycle,
            "cycle": list(range(1, max_cycle + 1)),
        }
    )


def _make_multi_engine_df() -> pd.DataFrame:
    """Helper: two engines with different lifespans."""
    df1 = _make_engine_df("engine_1", 200)
    df2 = _make_engine_df("engine_2", 150)
    return pd.concat([df1, df2], ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# RUL label generation
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestRULLabels:
    def test_rul_decreases_over_time(self, label_config):
        lg = LabelGenerator(label_config)
        df = _make_engine_df("e1", 100)
        result = lg.generate_labels(df)
        assert "rul" in result.columns
        ruls = result["rul"].values
        # RUL must be monotonically decreasing for a single engine
        for i in range(1, len(ruls)):
            assert ruls[i] <= ruls[i - 1]

    def test_rul_ends_at_zero(self, label_config):
        lg = LabelGenerator(label_config)
        df = _make_engine_df("e1", 100)
        result = lg.generate_labels(df)
        assert result["rul"].iloc[-1] == 0

    def test_piecewise_linear_clipping(self, label_config):
        """With piecewise linear enabled, RUL should be capped at early_life_rul."""
        lg = LabelGenerator(label_config)
        df = _make_engine_df("e1", 500)
        result = lg.generate_labels(df)
        max_rul = label_config["label_generation"]["rul"]["piecewise_linear"][
            "early_life_rul"
        ]
        assert result["rul"].max() <= max_rul

    def test_rul_normalized_range(self, label_config):
        lg = LabelGenerator(label_config)
        df = _make_engine_df("e1", 100)
        result = lg.generate_labels(df)
        if "rul_normalized" in result.columns:
            assert result["rul_normalized"].min() >= 0.0
            assert result["rul_normalized"].max() <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Binary failure labels
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestBinaryLabels:
    def test_failure_imminent_flag(self, label_config):
        lg = LabelGenerator(label_config)
        df = _make_engine_df("e1", 100)
        result = lg.generate_labels(df)
        if "failure_imminent" in result.columns:
            window = label_config["label_generation"]["binary_failure"][
                "window_before_failure"
            ]
            # Last `window` cycles should be 1
            tail = result["failure_imminent"].iloc[-window:]
            assert tail.sum() == window
            # Earlier cycles should be 0
            assert result["failure_imminent"].iloc[0] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Health status classification
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestHealthStatus:
    def test_health_status_column_exists(self, label_config):
        lg = LabelGenerator(label_config)
        df = _make_engine_df("e1", 200)
        result = lg.generate_labels(df)
        assert "health_status" in result.columns

    def test_health_status_categories(self, label_config):
        lg = LabelGenerator(label_config)
        df = _make_engine_df("e1", 300)
        result = lg.generate_labels(df)
        expected = set(
            label_config["label_generation"]["health_status"]["categories"].keys()
        )
        actual = set(result["health_status"].dropna().unique())
        # At least some configured categories should appear
        assert len(actual & expected) > 0, (
            f"Got statuses {actual}, expected from {expected}"
        )

    def test_health_status_code_numeric(self, label_config):
        lg = LabelGenerator(label_config)
        df = _make_engine_df("e1", 200)
        result = lg.generate_labels(df)
        if "health_status_code" in result.columns:
            codes = result["health_status_code"].dropna().unique()
            for c in codes:
                assert isinstance(int(c), int)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-engine
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestMultiEngine:
    def test_per_engine_rul_independent(self, label_config):
        lg = LabelGenerator(label_config)
        df = _make_multi_engine_df()
        result = lg.generate_labels(df)
        for eid in result["equipment_id"].unique():
            sub = result[result["equipment_id"] == eid]
            assert sub["rul"].iloc[-1] == 0
