"""
Retraining trigger-safety guard.

The historical best model sits at 79.9% within-15 accuracy against an 80%
threshold, so a naive "accuracy < threshold -> retrain" loop would retrain on
every scheduled check, forever, on the same static C-MAPSS data. This guard
makes the trigger decision explicit and safe:

  * a cooldown prevents immediate repeated retraining
  * an active-job lock prevents two concurrent retrains
  * a last-trigger timestamp is persisted in the decision record
  * every outcome is one explicit, named state -- never a silent "healthy"

States (RetrainDecision.state):
    HEALTHY               accuracy >= threshold
    DEGRADED              accuracy < threshold and all guards clear -> retrain
    INSUFFICIENT_DATA     not enough samples to decide
    EVALUATION_UNAVAILABLE could not evaluate -> NOT treated as healthy
    COOLDOWN_ACTIVE       below threshold but within the cooldown window
    ALREADY_ACTIVE        a retraining job is already running

Thread-safety: the active-job flag is guarded by an RLock so concurrent
scheduler ticks cannot both enter retraining.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional


class RetrainState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    INSUFFICIENT_DATA = "insufficient_data"
    EVALUATION_UNAVAILABLE = "evaluation_unavailable"
    COOLDOWN_ACTIVE = "cooldown_active"
    ALREADY_ACTIVE = "already_active"


@dataclass
class RetrainDecision:
    state: RetrainState
    should_retrain: bool
    reason: str
    accuracy: Optional[float] = None
    threshold: Optional[float] = None
    n_samples: Optional[int] = None
    last_trigger_at: Optional[str] = None
    cooldown_seconds: Optional[int] = None
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class RetrainGuard:
    """Gate that decides whether a below-threshold signal may start retraining."""

    def __init__(self, cooldown_seconds: int = 3600, min_samples: int = 50):
        self.cooldown_seconds = cooldown_seconds
        self.min_samples = min_samples
        self._lock = threading.RLock()
        self._active = False
        self._last_trigger_at: Optional[datetime] = None

    # -- introspection -------------------------------------------------------
    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def last_trigger_at(self) -> Optional[datetime]:
        with self._lock:
            return self._last_trigger_at

    def _in_cooldown(self, now: datetime) -> bool:
        if self._last_trigger_at is None:
            return False
        return now - self._last_trigger_at < timedelta(seconds=self.cooldown_seconds)

    def decide(
        self,
        *,
        accuracy: Optional[float],
        threshold: float,
        n_samples: int,
        evaluation_available: bool = True,
        now: Optional[datetime] = None,
    ) -> RetrainDecision:
        """
        Pure decision -- does NOT mark a job active. Call ``begin()`` only after
        acting on a DEGRADED decision.
        """
        now = now or datetime.now(timezone.utc)
        last = self._last_trigger_at.isoformat() if self._last_trigger_at else None

        def mk(state, should, reason):
            return RetrainDecision(
                state=state, should_retrain=should, reason=reason,
                accuracy=accuracy, threshold=threshold, n_samples=n_samples,
                last_trigger_at=last, cooldown_seconds=self.cooldown_seconds,
            )

        if not evaluation_available or accuracy is None:
            # NEVER a silent "healthy": an evaluation we could not run is
            # inconclusive, and must not suppress a real degradation.
            return mk(RetrainState.EVALUATION_UNAVAILABLE, False,
                      "model/data evaluation unavailable — inconclusive")

        if n_samples < self.min_samples:
            return mk(RetrainState.INSUFFICIENT_DATA, False,
                      f"only {n_samples} samples (< {self.min_samples})")

        if accuracy >= threshold:
            return mk(RetrainState.HEALTHY, False,
                      f"accuracy {accuracy:.3f} >= threshold {threshold:.3f}")

        with self._lock:
            if self._active:
                return mk(RetrainState.ALREADY_ACTIVE, False,
                          "a retraining job is already running")
            if self._in_cooldown(now):
                return mk(RetrainState.COOLDOWN_ACTIVE, False,
                          f"within {self.cooldown_seconds}s cooldown of last trigger")

        return mk(RetrainState.DEGRADED, True,
                  f"accuracy {accuracy:.3f} < threshold {threshold:.3f}")

    def begin(self, now: Optional[datetime] = None) -> bool:
        """
        Atomically claim the active-job slot and stamp last_trigger_at.
        Returns False if a job is already active or cooldown is in force.
        """
        now = now or datetime.now(timezone.utc)
        with self._lock:
            if self._active or self._in_cooldown(now):
                return False
            self._active = True
            self._last_trigger_at = now
            return True

    def end(self) -> None:
        with self._lock:
            self._active = False
