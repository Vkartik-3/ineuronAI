"""
Database Notifier

Persists alerts to the TimescaleDB `alerts` table.
Uses psycopg2 connection pooling with retry logic.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

_INSERT_ALERT = """
INSERT INTO alerts (
    alert_id, equipment_id, rule_id, severity, message,
    status, data, notifications_sent, triggered_at,
    acknowledged_by, acknowledged_at, resolved_at,
    suppressed, suppress_reason
) VALUES (
    %(alert_id)s, %(equipment_id)s, %(rule_id)s, %(severity)s, %(message)s,
    %(status)s, %(data)s, %(notifications_sent)s, %(triggered_at)s,
    %(acknowledged_by)s, %(acknowledged_at)s, %(resolved_at)s,
    %(suppressed)s, %(suppress_reason)s
)
ON CONFLICT (alert_id) DO UPDATE SET
    status            = EXCLUDED.status,
    data              = EXCLUDED.data,
    notifications_sent = EXCLUDED.notifications_sent,
    acknowledged_by   = EXCLUDED.acknowledged_by,
    acknowledged_at   = EXCLUDED.acknowledged_at,
    resolved_at       = EXCLUDED.resolved_at,
    suppressed        = EXCLUDED.suppressed,
    suppress_reason   = EXCLUDED.suppress_reason
"""

_UPDATE_ALERT_STATUS = """
UPDATE alerts
   SET status          = %(status)s,
       acknowledged_by = %(acknowledged_by)s,
       acknowledged_at = %(acknowledged_at)s,
       resolved_at     = %(resolved_at)s
 WHERE alert_id = %(alert_id)s
"""

_QUERY_HISTORY_BASE = """
SELECT alert_id, equipment_id, rule_id, severity, message,
       status, data, notifications_sent, triggered_at,
       acknowledged_by, acknowledged_at, resolved_at,
       suppressed, suppress_reason
  FROM alerts
 WHERE 1=1
"""

_DELETE_OLD_ALERTS = """
DELETE FROM alerts WHERE triggered_at < %s
"""

_STATS_QUERY = """
SELECT
    COUNT(*)                                                     AS total_alerts,
    COUNT(*) FILTER (WHERE status IN ('triggered','acknowledged')) AS active_alerts,
    COUNT(*) FILTER (WHERE severity = 'critical')                 AS critical_count,
    COUNT(*) FILTER (WHERE severity = 'warning')                  AS warning_count,
    COUNT(*) FILTER (WHERE severity = 'info')                     AS info_count,
    MIN(triggered_at)                                             AS earliest_alert,
    MAX(triggered_at)                                             AS latest_alert
  FROM alerts
"""


class DatabaseNotifier:
    """
    Persists alerts in the TimescaleDB ``alerts`` table.

    Uses ``psycopg2.pool.SimpleConnectionPool`` for connection management
    and includes retry logic for transient connection failures.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.retention_days = config.get("retention_days", 90)
        self._pool: Optional[pg_pool.SimpleConnectionPool] = None

        if self.enabled:
            self._init_pool()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _dsn_kwargs(self) -> Dict[str, Any]:
        """Build keyword args for psycopg2 from config."""
        if "connection_string" in self.config:
            return {"dsn": self.config["connection_string"]}
        return {
            "host": self.config.get("host", "timescaledb"),
            "port": int(self.config.get("port", 5432)),
            "dbname": self.config.get("database", "predictive_maintenance"),
            "user": self.config.get("user", "pmuser"),
            "password": self.config.get("password", "pmpassword"),
        }

    def _init_pool(self) -> None:
        """Create the connection pool with retry."""
        min_conn = int(self.config.get("pool_min_connections", 1))
        max_conn = int(self.config.get("pool_max_connections", 5))
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                self._pool = pg_pool.SimpleConnectionPool(
                    min_conn, max_conn, **self._dsn_kwargs()
                )
                logger.info("Database notifier pool initialised")
                return
            except psycopg2.OperationalError as exc:
                logger.warning(
                    "DB pool init attempt %d/%d failed: %s",
                    attempt,
                    self.MAX_RETRIES,
                    exc,
                )
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY * attempt)
        logger.error("Could not initialise database notifier pool – disabling")
        self.enabled = False

    def _get_conn(self):
        """Get a connection from the pool, reconnecting if necessary."""
        if self._pool is None or self._pool.closed:
            self._init_pool()
        if self._pool is None:
            raise psycopg2.OperationalError("Connection pool unavailable")
        return self._pool.getconn()

    def _put_conn(self, conn) -> None:
        if self._pool and not self._pool.closed:
            self._pool.putconn(conn)

    def _execute(self, sql: str, params=None, *, fetch: bool = False):
        """Execute *sql* with automatic retry on connection errors."""
        last_exc = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            conn = None
            try:
                conn = self._get_conn()
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, params)
                    result = cur.fetchall() if fetch else cur.rowcount
                conn.commit()
                return result
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
                last_exc = exc
                logger.warning("DB execute attempt %d failed: %s", attempt, exc)
                if conn:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    try:
                        self._put_conn(conn)
                    except Exception:
                        pass
                    conn = None
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY * attempt)
            except Exception as exc:
                last_exc = exc
                if conn:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                raise
            finally:
                if conn:
                    self._put_conn(conn)
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_alert(self, alert: Any) -> bool:
        """
        Persist an alert to the ``alerts`` table.

        Args:
            alert: An ``Alert`` instance from ``alert_rules.py``.

        Returns:
            ``True`` if the row was written successfully.
        """
        if not self.enabled:
            logger.debug("Database notifications disabled")
            return False

        try:
            record = self._alert_to_record(alert)
            self._execute(_INSERT_ALERT, record)
            logger.info("Alert %s persisted to database", record["alert_id"])
            return True
        except Exception as exc:
            logger.error("Failed to persist alert: %s", exc)
            return False

    def update_alert(
        self,
        alert_id: str,
        status: str,
        acknowledged_by: Optional[str] = None,
        acknowledged_at: Optional[datetime] = None,
        resolved_at: Optional[datetime] = None,
    ) -> bool:
        """
        Update the status of an existing alert.

        Args:
            alert_id: The alert to update.
            status: New status value (``acknowledged``, ``resolved``, etc.).
            acknowledged_by: User who acknowledged the alert.
            acknowledged_at: Timestamp of acknowledgement.
            resolved_at: Timestamp of resolution.

        Returns:
            ``True`` if at least one row was updated.
        """
        if not self.enabled:
            return False

        try:
            rows = self._execute(
                _UPDATE_ALERT_STATUS,
                {
                    "alert_id": alert_id,
                    "status": status,
                    "acknowledged_by": acknowledged_by,
                    "acknowledged_at": acknowledged_at,
                    "resolved_at": resolved_at,
                },
            )
            if rows and rows > 0:
                logger.info("Alert %s updated to status '%s'", alert_id, status)
                return True
            logger.warning("Alert %s not found for update", alert_id)
            return False
        except Exception as exc:
            logger.error("Failed to update alert %s: %s", alert_id, exc)
            return False

    def get_alert_history(
        self,
        equipment_id: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query alert history with optional filters.

        Returns:
            List of alert records as dictionaries.
        """
        if not self.enabled:
            return []

        try:
            clauses: List[str] = []
            params: List[Any] = []

            if equipment_id:
                clauses.append("equipment_id = %s")
                params.append(equipment_id)
            if severity:
                clauses.append("severity = %s")
                params.append(severity)
            if start_time:
                clauses.append("triggered_at >= %s")
                params.append(start_time)
            if end_time:
                clauses.append("triggered_at <= %s")
                params.append(end_time)

            sql = _QUERY_HISTORY_BASE
            if clauses:
                sql += " AND " + " AND ".join(clauses)
            sql += " ORDER BY triggered_at DESC LIMIT %s"
            params.append(limit)

            rows = self._execute(sql, params, fetch=True)
            # Convert RealDictRow → plain dict and serialise datetimes
            return [self._serialise_row(dict(r)) for r in rows]
        except Exception as exc:
            logger.error("Failed to query alert history: %s", exc)
            return []

    def cleanup_old_alerts(self) -> int:
        """Delete alerts older than the configured retention period."""
        if not self.enabled:
            return 0

        try:
            cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
            deleted = self._execute(_DELETE_OLD_ALERTS, (cutoff,))
            logger.info(
                "Cleaned up %d alerts older than %d days", deleted, self.retention_days
            )
            return deleted or 0
        except Exception as exc:
            logger.error("Failed to cleanup old alerts: %s", exc)
            return 0

    def get_statistics(self) -> Dict[str, Any]:
        """Aggregate statistics from the alerts table."""
        if not self.enabled:
            return {}

        try:
            rows = self._execute(_STATS_QUERY, fetch=True)
            if not rows:
                return {}
            row = dict(rows[0])
            return {
                "total_alerts": row.get("total_alerts", 0),
                "active_alerts": row.get("active_alerts", 0),
                "by_severity": {
                    "critical": row.get("critical_count", 0),
                    "warning": row.get("warning_count", 0),
                    "info": row.get("info_count", 0),
                },
                "earliest_alert": (
                    row["earliest_alert"].isoformat()
                    if row.get("earliest_alert")
                    else None
                ),
                "latest_alert": (
                    row["latest_alert"].isoformat() if row.get("latest_alert") else None
                ),
            }
        except Exception as exc:
            logger.error("Failed to get statistics: %s", exc)
            return {}

    def send_bulk_alerts(self, alerts: List[Any]) -> Dict[str, int]:
        """Persist multiple alerts, returning success/failure counts."""
        results = {"success": 0, "failed": 0}
        for alert in alerts:
            if self.send_alert(alert):
                results["success"] += 1
            else:
                results["failed"] += 1
        return results

    def close(self) -> None:
        """Close the connection pool."""
        if self._pool and not self._pool.closed:
            self._pool.closeall()
            logger.info("Database notifier pool closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _alert_to_record(alert: Any) -> Dict[str, Any]:
        """Map an ``Alert`` object to INSERT parameters."""
        suppressed = False
        suppress_reason = None
        try:
            from rules.alert_rules import AlertStatus

            suppressed = alert.status == AlertStatus.SUPPRESSED
        except Exception:
            pass

        return {
            "alert_id": alert.alert_id,
            "equipment_id": alert.equipment_id,
            "rule_id": alert.rule_id,
            "severity": alert.severity.value
            if hasattr(alert.severity, "value")
            else str(alert.severity),
            "message": alert.message,
            "status": alert.status.value
            if hasattr(alert.status, "value")
            else str(alert.status),
            "data": json.dumps(alert.data) if alert.data else None,
            "notifications_sent": None,
            "triggered_at": alert.timestamp,
            "acknowledged_by": getattr(alert, "acknowledged_by", None),
            "acknowledged_at": getattr(alert, "acknowledged_at", None),
            "resolved_at": getattr(alert, "resolved_at", None),
            "suppressed": suppressed,
            "suppress_reason": suppress_reason,
        }

    @staticmethod
    def _serialise_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """JSON-safe serialisation of a database row."""
        for key in ("triggered_at", "acknowledged_at", "resolved_at"):
            val = row.get(key)
            if val and hasattr(val, "isoformat"):
                row[key] = val.isoformat()
        return row
