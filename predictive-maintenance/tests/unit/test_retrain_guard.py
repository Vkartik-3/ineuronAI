"""Retraining trigger-safety and comparator-defense regression tests."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

_PM_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_PM_ROOT), str(_PM_ROOT / "ml_pipeline" / "retrain")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from retrain_guard import RetrainGuard, RetrainState  # noqa: E402


def _guard(cooldown=3600, min_samples=50):
    return RetrainGuard(cooldown_seconds=cooldown, min_samples=min_samples)


@pytest.mark.unit
class TestRetrainThresholdSemantics:
    def test_79_percent_triggers(self):
        d = _guard().decide(accuracy=0.79, threshold=0.80, n_samples=60)
        assert d.state is RetrainState.DEGRADED and d.should_retrain

    def test_exactly_80_percent_does_not_trigger(self):
        d = _guard().decide(accuracy=0.80, threshold=0.80, n_samples=60)
        assert d.state is RetrainState.HEALTHY and not d.should_retrain

    def test_81_percent_does_not_trigger(self):
        d = _guard().decide(accuracy=0.81, threshold=0.80, n_samples=60)
        assert d.state is RetrainState.HEALTHY and not d.should_retrain

    def test_insufficient_samples_does_not_trigger(self):
        d = _guard(min_samples=50).decide(accuracy=0.10, threshold=0.80, n_samples=49)
        assert d.state is RetrainState.INSUFFICIENT_DATA and not d.should_retrain

    def test_evaluation_unavailable_is_not_healthy(self):
        d = _guard().decide(
            accuracy=None, threshold=0.80, n_samples=60, evaluation_available=False
        )
        assert d.state is RetrainState.EVALUATION_UNAVAILABLE
        assert not d.should_retrain
        # critically: NOT reported as healthy
        assert d.state is not RetrainState.HEALTHY


@pytest.mark.unit
class TestRetrainGuardConcurrencyAndCooldown:
    def test_active_job_prevents_duplicate(self):
        g = _guard()
        assert g.begin() is True
        # second begin while active is refused
        assert g.begin() is False
        # decide() reports ALREADY_ACTIVE for a fresh below-threshold signal
        d = g.decide(accuracy=0.5, threshold=0.80, n_samples=60)
        assert d.state is RetrainState.ALREADY_ACTIVE and not d.should_retrain
        g.end()

    def test_cooldown_prevents_immediate_retrigger(self):
        g = _guard(cooldown=3600)
        now = datetime.now(timezone.utc)
        assert g.begin(now=now) is True
        g.end()
        # 10 minutes later, still within the 1h cooldown
        d = g.decide(
            accuracy=0.5, threshold=0.80, n_samples=60, now=now + timedelta(minutes=10)
        )
        assert d.state is RetrainState.COOLDOWN_ACTIVE and not d.should_retrain

    def test_cooldown_expires(self):
        g = _guard(cooldown=3600)
        now = datetime.now(timezone.utc)
        g.begin(now=now)
        g.end()
        d = g.decide(
            accuracy=0.5, threshold=0.80, n_samples=60, now=now + timedelta(hours=2)
        )
        assert d.state is RetrainState.DEGRADED and d.should_retrain

    def test_begin_is_atomic_under_threads(self):
        import threading

        g = _guard(cooldown=0)
        wins = []

        def worker():
            if g.begin():
                wins.append(1)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(wins) == 1, "exactly one thread may claim the active-job slot"


@pytest.mark.unit
class TestComparatorDefense:
    def _cmp(self):
        from model_comparator import ModelComparator

        c = ModelComparator.__new__(ModelComparator)
        c.min_improvement = 5.0
        c.primary_metric = "mae"
        return c

    def test_significant_improvement_promotes(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(20, 120, 400)
        champ = y + rng.normal(0, 15, 400)
        chall = y + rng.normal(0, 6, 400)
        r = self._cmp().compare_predictions(y, champ, chall)
        assert r.should_promote and r.winner == "challenger"
        assert r.details["bootstrap_ci"]["challenger_significantly_better"]

    def test_tie_does_not_promote(self):
        rng = np.random.default_rng(1)
        y = rng.uniform(20, 120, 200)
        pred = y + rng.normal(0, 10, 200)
        r = self._cmp().compare_predictions(y, pred, pred.copy())
        assert not r.should_promote

    def test_below_5_percent_does_not_promote(self):
        rng = np.random.default_rng(2)
        y = rng.uniform(20, 120, 500)
        champ = y + rng.normal(0, 10.0, 500)
        chall = y + rng.normal(0, 9.7, 500)  # ~3% better, under the 5% gate
        r = self._cmp().compare_predictions(y, champ, chall)
        assert not r.should_promote

    def test_unequal_lengths_error(self):
        r = self._cmp().compare_predictions(np.zeros(10), np.zeros(10), np.zeros(9))
        assert not r.should_promote and "ERROR" in r.recommendation

    def test_nan_rejected(self):
        y = np.array([1.0, 2.0, 3.0])
        r = self._cmp().compare_predictions(y, np.array([1.0, np.nan, 3.0]), y)
        assert not r.should_promote and "ERROR" in r.recommendation

    def test_zero_champion_score_guarded(self):
        y = np.array([10.0, 20.0, 30.0])
        r = self._cmp().compare_predictions(y, y.copy(), y + 1.0)  # champion perfect
        assert not r.should_promote
        assert r.details.get("reason") == "zero_champion_score"

    def test_empty_error(self):
        r = self._cmp().compare_predictions(np.array([]), np.array([]), np.array([]))
        assert not r.should_promote and "ERROR" in r.recommendation
