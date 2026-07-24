"""
Performance Monitor — Accuracy-Based Retraining Trigger

Monitors live model accuracy on a rolling window of recent predictions.
Triggers retraining when the within-tolerance prediction accuracy drops
below the configured threshold (default 80%).

Accuracy definition:
    fraction of predictions within ±tolerance cycles of the true RUL
    (standard metric in PHM prognostics literature, tolerance=15 cycles)

This is the mechanism described in the resume:
    "triggered retraining when accuracy dropped below 80%"

Persistence:
    Ground-truth/prediction records are written to a shared Redis sorted set
    (score = UTC timestamp) so that accuracy is tracked **system-wide** —
    durable across process restarts and shared by every Uvicorn worker / ECS
    task, not just the worker that happened to receive the /feedback call.
    If Redis is unreachable, the monitor falls back to a per-process
    in-memory list (with a loud warning) so the service keeps functioning,
    just without cross-worker durability.
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import yaml

logger = logging.getLogger(__name__)

try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _redis_lib = None
    _REDIS_AVAILABLE = False
    logger.warning(
        "redis-py not installed — PerformanceMonitor will store prediction "
        "records in-memory only (not shared across workers/ECS tasks)"
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord:
    """Single ground-truth / prediction pair used for accuracy monitoring."""
    timestamp: datetime
    equipment_id: str
    true_rul: float
    predicted_rul: float
    within_tolerance: bool = field(init=False)

    def __post_init__(self) -> None:
        self.within_tolerance = False  # set externally once tolerance is known


@dataclass
class AccuracyReport:
    """Result of an accuracy check against the configured threshold."""
    timestamp: datetime
    window_days: int
    n_samples: int
    accuracy: float                   # fraction within tolerance
    threshold: float                  # the configured 80% threshold
    tolerance_cycles: int             # ±15 cycles by default
    below_threshold: bool             # True → retraining should fire
    mae: float
    rmse: float
    details: Dict


# ---------------------------------------------------------------------------
# Monitor class
# ---------------------------------------------------------------------------

class PerformanceMonitor:
    """
    Tracks rolling prediction accuracy and emits a retraining signal when
    accuracy drops below the configured threshold.

    **Shared, durable state via Redis — survives restarts and is visible to
    every Uvicorn worker / ECS task.**

    Records are written to a Redis sorted set (``ZADD``, score = UTC
    timestamp) keyed by ``<key_prefix>records``. ``check_accuracy_threshold()``
    reads the rolling window straight out of Redis with ``ZRANGEBYSCORE``, so
    every worker / task observes the same system-wide accuracy — not just the
    fraction of /feedback calls that happened to land on it. Stale records
    (older than the retention window) are pruned with ``ZREMRANGEBYSCORE`` on
    every write.

    If Redis is unreachable at startup (or a write/read fails at runtime), the
    monitor logs a clear warning and falls back to a per-process in-memory
    list — the service keeps functioning, but accuracy tracking degrades to a
    single-worker approximation until Redis is restored. Use
    ``is_persistent`` to check which mode is active.

    Ground-truth must be provided externally via the ``POST /feedback`` API
    endpoint or ``record_batch()``. Without ground-truth records the monitor
    never triggers retraining.

    Usage:
        monitor = PerformanceMonitor()

        # Feed in predictions as they arrive (via POST /feedback):
        monitor.record_prediction("EQ-01", true_rul=87.0, predicted_rul=80.2)

        # Check during scheduled runs:
        report = monitor.check_accuracy_threshold()
        if report.below_threshold:
            pipeline.trigger_retraining(reason="accuracy_degradation")
    """

    def __init__(self, config_path: str = "config/retrain_config.yaml"):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            # Config file not present (e.g. Docker inference image which only
            # copies inference_service/, not the full retrain package).
            # All config keys have sensible defaults below — log and continue.
            logger.warning(
                "PerformanceMonitor config not found at '%s' — using built-in "
                "defaults (threshold=80%%, tolerance=±15 cycles, window=7d)",
                config_path,
            )
            cfg = {}

        pm_cfg = cfg.get("performance_monitoring", {})
        self.accuracy_threshold: float = pm_cfg.get("accuracy_threshold", 0.80)
        self.tolerance_cycles: int = pm_cfg.get("accuracy_tolerance_cycles", 15)
        self.window_days: int = pm_cfg.get("window_days", 7)
        self.min_samples: int = pm_cfg.get("min_samples_for_check", 50)

        # In-memory fallback store — used only when Redis is unavailable.
        self._records: List[PredictionRecord] = []

        # Shared, durable store — every worker/ECS task reads & writes the
        # same Redis sorted set, so accuracy tracking is system-wide.
        redis_cfg = pm_cfg.get("redis", {})
        self._redis_key_prefix: str = redis_cfg.get("key_prefix", "perf_monitor:")
        self._redis_key: str = f"{self._redis_key_prefix}records"
        # Keep a bit more than the rolling window so get_accuracy_trend()
        # (which looks across all stored records) still has history to plot.
        self._record_retention_days: int = redis_cfg.get(
            "record_retention_days", max(self.window_days * 2, 14)
        )
        self._redis_client = self._connect_redis(redis_cfg)

        logger.info(
            "PerformanceMonitor initialised — threshold=%.0f%%  tolerance=±%d cycles  "
            "window=%d days  min_samples=%d  persistence=%s",
            self.accuracy_threshold * 100,
            self.tolerance_cycles,
            self.window_days,
            self.min_samples,
            "redis (shared/durable)" if self._redis_client is not None else "in-memory (per-process)",
        )

    # ------------------------------------------------------------------
    # Redis connection setup — mirrors PredictionCache's pattern
    # ------------------------------------------------------------------

    def _connect_redis(self, redis_cfg: Dict) -> Optional["_redis_lib.Redis"]:
        """
        Connect to Redis for durable, cross-worker record storage.

        Returns the connected client, or None (with a warning already logged)
        if redis-py isn't installed or the server is unreachable — callers
        must then use the in-memory ``self._records`` fallback.
        """
        if not _REDIS_AVAILABLE:
            logger.warning(
                "redis-py not installed — PerformanceMonitor will track "
                "accuracy in-memory only. Records will NOT survive restarts "
                "and will NOT be shared across Uvicorn workers / ECS tasks. "
                "Install redis-py for durable, system-wide accuracy tracking."
            )
            return None

        host = os.environ.get("REDIS_HOST", redis_cfg.get("host", "redis"))
        port = int(os.environ.get("REDIS_PORT", redis_cfg.get("port", 6379)))
        db = int(redis_cfg.get("db", 1))
        password = os.environ.get("REDIS_PASSWORD", redis_cfg.get("password"))

        try:
            client = _redis_lib.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_timeout=2,
                socket_connect_timeout=2,
                socket_keepalive=True,
            )
            client.ping()
            logger.info(
                "PerformanceMonitor connected to Redis at %s:%d (db=%d, key=%s) "
                "— prediction records are now durable and shared across workers",
                host, port, db, self._redis_key,
            )
            return client
        except Exception as exc:
            logger.warning(
                "Redis not reachable at %s:%d (db=%d) — PerformanceMonitor "
                "falling back to in-memory records. This is NOT durable across "
                "restarts and NOT shared across multiple workers/ECS tasks "
                "(each will only see its own predictions): %s",
                host, port, db, exc,
            )
            return None

    @property
    def is_persistent(self) -> bool:
        """True when records are durably stored in Redis (shared across workers)."""
        return self._redis_client is not None

    # ------------------------------------------------------------------
    # Record (de)serialisation for Redis storage
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_record(rec: PredictionRecord) -> str:
        """
        JSON-encode a record for storage as a Redis sorted-set member.

        A random ``_id`` is embedded so that two records with identical
        field values still produce distinct members (ZSET members must be
        unique strings — without this, simultaneous identical predictions
        would silently overwrite one another).
        """
        return json.dumps(
            {
                "_id": uuid.uuid4().hex,
                "timestamp": rec.timestamp.isoformat(),
                "equipment_id": rec.equipment_id,
                "true_rul": rec.true_rul,
                "predicted_rul": rec.predicted_rul,
                "within_tolerance": rec.within_tolerance,
            }
        )

    @staticmethod
    def _deserialize_record(raw: str) -> Optional[PredictionRecord]:
        """Decode a Redis sorted-set member back into a PredictionRecord."""
        try:
            d = json.loads(raw)
            rec = PredictionRecord(
                timestamp=datetime.fromisoformat(d["timestamp"]),
                equipment_id=d["equipment_id"],
                true_rul=float(d["true_rul"]),
                predicted_rul=float(d["predicted_rul"]),
            )
            rec.within_tolerance = bool(d["within_tolerance"])
            return rec
        except Exception as exc:
            logger.warning("Skipping malformed PredictionRecord from Redis: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Recording predictions
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        equipment_id: str,
        true_rul: float,
        predicted_rul: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Store a prediction/ground-truth pair for accuracy tracking.

        Writes to the shared Redis sorted set (durable, cross-worker) when
        available; otherwise appends to the in-memory fallback list.

        Args:
            equipment_id:  Engine / equipment identifier
            true_rul:      Ground-truth remaining useful life (cycles)
            predicted_rul: Model's RUL prediction
            timestamp:     Observation time (default: now)
        """
        ts = timestamp or datetime.utcnow()
        rec = PredictionRecord(
            timestamp=ts,
            equipment_id=equipment_id,
            true_rul=true_rul,
            predicted_rul=predicted_rul,
        )
        rec.within_tolerance = abs(true_rul - predicted_rul) <= self.tolerance_cycles

        if self._redis_client is not None:
            try:
                payload = self._serialize_record(rec)
                self._redis_client.zadd(self._redis_key, {payload: ts.timestamp()})
                # Prune anything past the retention window on every write —
                # keeps the sorted set bounded without a separate cron job.
                cutoff = ts - timedelta(days=self._record_retention_days)
                self._redis_client.zremrangebyscore(self._redis_key, "-inf", cutoff.timestamp())
                return
            except Exception as exc:
                logger.warning(
                    "Redis write failed for prediction record (equipment=%s) — "
                    "storing in-memory for this process only: %s",
                    equipment_id, exc,
                )
                # fall through to in-memory append below

        self._records.append(rec)

    def record_batch(
        self,
        equipment_ids: List[str],
        true_ruls: np.ndarray,
        predicted_ruls: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Convenience wrapper for recording a batch of predictions."""
        ts = timestamp or datetime.utcnow()
        for eid, t, p in zip(equipment_ids, true_ruls, predicted_ruls):
            self.record_prediction(eid, float(t), float(p), ts)

    # ------------------------------------------------------------------
    # Accuracy computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_within_tolerance_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tolerance: int = 15,
    ) -> float:
        """
        Fraction of predictions within ±tolerance cycles of the true RUL.

        This is the accuracy definition used for the 80% retraining threshold:
            accuracy = |{i : |y_true_i - y_pred_i| <= tolerance}| / N

        Args:
            y_true:    True RUL values (n_samples,)
            y_pred:    Predicted RUL values (n_samples,)
            tolerance: Cycle tolerance window (default 15 cycles)

        Returns:
            Float in [0, 1]
        """
        return float(np.mean(np.abs(y_true - y_pred) <= tolerance))

    def _get_window_records(self) -> List[PredictionRecord]:
        """
        Records from the rolling accuracy window (now - window_days .. now),
        read from the shared Redis store when available so every worker /
        ECS task evaluates the exact same system-wide population.
        """
        cutoff = datetime.utcnow() - timedelta(days=self.window_days)

        if self._redis_client is not None:
            try:
                raw_members = self._redis_client.zrangebyscore(
                    self._redis_key, cutoff.timestamp(), "+inf"
                )
                records = [self._deserialize_record(m) for m in raw_members]
                return [r for r in records if r is not None]
            except Exception as exc:
                logger.warning(
                    "Redis read failed while fetching accuracy window — "
                    "falling back to in-memory records for this check: %s", exc,
                )

        return [r for r in self._records if r.timestamp >= cutoff]

    def _get_all_records(self) -> List[PredictionRecord]:
        """All retained records (used for trend analysis / record counts)."""
        if self._redis_client is not None:
            try:
                raw_members = self._redis_client.zrange(self._redis_key, 0, -1)
                records = [self._deserialize_record(m) for m in raw_members]
                return [r for r in records if r is not None]
            except Exception as exc:
                logger.warning(
                    "Redis read failed while fetching all records — "
                    "falling back to in-memory records: %s", exc,
                )

        return list(self._records)

    def check_accuracy_threshold(
        self,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
    ) -> AccuracyReport:
        """
        Evaluate current model accuracy against the 80% threshold.

        Can be called in two modes:
          1. Pass y_true / y_pred arrays directly (ad-hoc evaluation)
          2. No arguments — uses the rolling window of stored records

        Returns:
            AccuracyReport (check .below_threshold to decide on retraining)
        """
        now = datetime.utcnow()

        if y_true is not None and y_pred is not None:
            # Ad-hoc mode
            n_samples = len(y_true)
            accuracy = self.compute_within_tolerance_accuracy(
                y_true, y_pred, self.tolerance_cycles
            )
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        else:
            # Rolling window mode
            records = self._get_window_records()
            n_samples = len(records)

            if n_samples < self.min_samples:
                logger.info(
                    "Not enough samples for accuracy check (%d < %d required)",
                    n_samples, self.min_samples,
                )
                return AccuracyReport(
                    timestamp=now,
                    window_days=self.window_days,
                    n_samples=n_samples,
                    accuracy=1.0,           # treat as OK when insufficient data
                    threshold=self.accuracy_threshold,
                    tolerance_cycles=self.tolerance_cycles,
                    below_threshold=False,
                    mae=0.0,
                    rmse=0.0,
                    details={"reason": "insufficient_samples"},
                )

            y_true = np.array([r.true_rul for r in records], dtype=np.float32)
            y_pred = np.array([r.predicted_rul for r in records], dtype=np.float32)
            accuracy = float(np.mean([r.within_tolerance for r in records]))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        below = accuracy < self.accuracy_threshold

        report = AccuracyReport(
            timestamp=now,
            window_days=self.window_days,
            n_samples=n_samples,
            accuracy=accuracy,
            threshold=self.accuracy_threshold,
            tolerance_cycles=self.tolerance_cycles,
            below_threshold=below,
            mae=mae,
            rmse=rmse,
            details={
                "accuracy_pct": round(accuracy * 100, 2),
                "threshold_pct": round(self.accuracy_threshold * 100, 2),
                "gap_pct": round((accuracy - self.accuracy_threshold) * 100, 2),
                "tolerance_cycles": self.tolerance_cycles,
                "window_days": self.window_days,
            },
        )

        if below:
            logger.warning(
                "ACCURACY BELOW THRESHOLD — %.1f%% < %.1f%% (n=%d, MAE=%.2f) "
                "→ retraining required",
                accuracy * 100, self.accuracy_threshold * 100, n_samples, mae,
            )
        else:
            logger.info(
                "Accuracy OK — %.1f%% >= %.1f%% (n=%d, MAE=%.2f)",
                accuracy * 100, self.accuracy_threshold * 100, n_samples, mae,
            )

        return report

    # ------------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------------

    def log_accuracy_report(self, report: AccuracyReport) -> None:
        """Log accuracy report metrics to the active MLflow run."""
        try:
            mlflow.log_metrics(
                {
                    "current_accuracy": report.accuracy,
                    "accuracy_threshold": report.threshold,
                    "current_mae": report.mae,
                    "current_rmse": report.rmse,
                    "n_eval_samples": report.n_samples,
                    "below_threshold": int(report.below_threshold),
                }
            )
            mlflow.set_tag("accuracy_check_timestamp", report.timestamp.isoformat())
            mlflow.set_tag(
                "retraining_required", "yes" if report.below_threshold else "no"
            )
        except Exception as exc:
            logger.warning("MLflow logging for accuracy report failed: %s", exc)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear_old_records(self) -> int:
        """
        Prune records older than the retention window.

        Returns the count removed. Operates on Redis (``ZREMRANGEBYSCORE``)
        when persistent storage is active — note that ``record_prediction``
        already prunes opportunistically on every write, so this is mainly
        useful for an idle monitor or the in-memory fallback.
        """
        cutoff = datetime.utcnow() - timedelta(days=self._record_retention_days)

        if self._redis_client is not None:
            try:
                removed = int(
                    self._redis_client.zremrangebyscore(self._redis_key, "-inf", cutoff.timestamp())
                )
                if removed:
                    logger.debug("Cleared %d stale prediction records from Redis", removed)
                return removed
            except Exception as exc:
                logger.warning("Redis prune failed — leaving records in place: %s", exc)
                return 0

        before = len(self._records)
        self._records = [r for r in self._records if r.timestamp >= cutoff]
        removed = before - len(self._records)
        if removed:
            logger.debug("Cleared %d stale prediction records", removed)
        return removed

    def get_accuracy_trend(self, n_points: int = 10) -> List[Tuple[datetime, float]]:
        """
        Return the last n_points daily accuracy values for trend analysis.

        Useful for detecting gradual drift before the threshold is crossed.
        """
        all_records = self._get_all_records()
        if not all_records:
            return []

        sorted_recs = sorted(all_records, key=lambda r: r.timestamp)
        earliest = sorted_recs[0].timestamp
        latest = sorted_recs[-1].timestamp

        trend: List[Tuple[datetime, float]] = []
        day = earliest.replace(hour=0, minute=0, second=0, microsecond=0)
        while day <= latest:
            next_day = day + timedelta(days=1)
            day_recs = [r for r in sorted_recs if day <= r.timestamp < next_day]
            if len(day_recs) >= 5:
                day_acc = float(np.mean([r.within_tolerance for r in day_recs]))
                trend.append((day, day_acc))
            day = next_day

        return trend[-n_points:]

    @property
    def n_records(self) -> int:
        if self._redis_client is not None:
            try:
                return int(self._redis_client.zcard(self._redis_key))
            except Exception as exc:
                logger.warning("Redis ZCARD failed — falling back to in-memory count: %s", exc)
        return len(self._records)
