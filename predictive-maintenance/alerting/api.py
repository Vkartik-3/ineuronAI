"""
Alert Service REST API (port 8001).

Endpoints:
    POST /alerts/evaluate — accept prediction data, evaluate rules
    GET  /alerts          — list active alerts (from DB)
    POST /alerts/{id}/acknowledge
    POST /alerts/{id}/resolve
    GET  /alerts/statistics
    GET  /health
"""

import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse
from pydantic import BaseModel, Field

from metrics import (
    router as metrics_router,
    ALERTS_TRIGGERED_TOTAL,
    NOTIFICATIONS_SENT_TOTAL,
    ALERTS_ACKNOWLEDGED_TOTAL,
    ALERTS_RESOLVED_TOTAL,
    KAFKA_MESSAGES_CONSUMED_TOTAL,
    ALERT_EVALUATION_SECONDS,
    ACTIVE_ALERTS,
    KAFKA_CONSUMER_RUNNING,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class EvaluateRequest(BaseModel):
    equipment_id: str
    rul: Optional[float] = None
    anomaly_score: Optional[float] = None
    health_status: Optional[str] = None
    temperature: Optional[float] = None
    vibration: Optional[float] = None
    timestamp: Optional[str] = None


class AlertSummary(BaseModel):
    alert_id: str
    rule_id: str
    severity: str
    message: str
    notifications_sent: Optional[List[str]] = None


class EvaluateResponse(BaseModel):
    alerts_triggered: int
    alerts: List[AlertSummary]


class AlertDetail(BaseModel):
    alert_id: str
    rule_id: str
    equipment_id: str
    severity: str
    message: str
    timestamp: str
    status: str
    data: Optional[Dict[str, Any]] = None


class AlertListResponse(BaseModel):
    total: int
    alerts: List[AlertDetail]
    next_cursor: Optional[str] = None
    has_more: bool = False


class AcknowledgeRequest(BaseModel):
    user: str


class AcknowledgeResponse(BaseModel):
    alert_id: str
    status: str = "acknowledged"
    acknowledged_by: str
    acknowledged_at: str


class ResolveResponse(BaseModel):
    alert_id: str
    status: str = "resolved"
    resolved_at: str


class AlertStatsResponse(BaseModel):
    total_alerts: int = 0
    active_alerts: int = 0
    severity_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}


class HealthResponse(BaseModel):
    status: str
    service: str = "alert-engine"
    kafka_consumer: str = "unknown"
    timestamp: str
    dependencies: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# App factory — called from run_alert_engine.py
# ---------------------------------------------------------------------------


def create_alert_api(alert_manager, kafka_consumer=None) -> FastAPI:
    """
    Build and return the FastAPI application.

    ``alert_manager`` is the ``AlertManager`` instance.
    ``kafka_consumer`` is the ``KafkaAlertConsumer`` (may be None at init time).
    """

    app = FastAPI(
        title="Alert Service API",
        version="1.0.0",
        docs_url="/docs",
    )

    # CORS — restrict to known origins
    cors_origins = os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:8501"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in cors_origins],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Security headers middleware
    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next):
            response: StarletteResponse = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
            response.headers["Cache-Control"] = "no-store"
            return response

    app.add_middleware(SecurityHeadersMiddleware)

    # Mount Prometheus metrics endpoint
    app.include_router(metrics_router)

    # -- POST /alerts/evaluate ---------------------------------------------

    @app.post("/alerts/evaluate", response_model=EvaluateResponse)
    async def evaluate_prediction(req: EvaluateRequest):
        import time as _time

        _eval_start = _time.time()
        prediction = req.dict(exclude_none=True)
        triggered = alert_manager.process_prediction(prediction)
        ALERT_EVALUATION_SECONDS.observe(_time.time() - _eval_start)
        summaries = []
        for a in triggered:
            sev = a.severity.value if hasattr(a.severity, "value") else str(a.severity)
            ALERTS_TRIGGERED_TOTAL.labels(severity=sev, rule_id=a.rule_id).inc()
            summaries.append(
                AlertSummary(
                    alert_id=a.alert_id,
                    rule_id=a.rule_id,
                    severity=sev,
                    message=a.message,
                    notifications_sent=list(alert_manager.notifiers.keys()),
                )
            )
        ACTIVE_ALERTS.set(len(alert_manager.get_active_alerts()))
        return EvaluateResponse(alerts_triggered=len(summaries), alerts=summaries)

    # -- GET /alerts -------------------------------------------------------

    @app.get("/alerts", response_model=AlertListResponse)
    async def list_alerts(
        equipment_id: Optional[str] = Query(None),
        severity: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=500),
        cursor: Optional[str] = Query(
            None, description="Cursor for pagination (alert_id of last item)"
        ),
    ):
        def _to_detail(r, ts_key="triggered_at"):
            return AlertDetail(
                alert_id=r.get("alert_id", ""),
                rule_id=r.get("rule_id", ""),
                equipment_id=r.get("equipment_id", ""),
                severity=r.get("severity", ""),
                message=r.get("message", ""),
                timestamp=str(r.get(ts_key, r.get("timestamp", ""))),
                status=r.get("status", ""),
                data=r.get("data"),
            )

        # fetch limit + 1 to detect has_more
        fetch_limit = limit + 1

        # Try DB first via database notifier
        db_notifier = alert_manager.notifiers.get("database")
        if db_notifier and db_notifier.enabled:
            rows = db_notifier.get_alert_history(
                equipment_id=equipment_id,
                severity=severity,
                limit=fetch_limit,
            )
            if status:
                rows = [r for r in rows if r.get("status") == status]

            # Cursor-based pagination: skip everything up to and including cursor
            if cursor:
                skip = True
                filtered = []
                for r in rows:
                    if skip:
                        if r.get("alert_id") == cursor:
                            skip = False
                        continue
                    filtered.append(r)
                rows = filtered

            has_more = len(rows) > limit
            rows = rows[:limit]
            alerts = [_to_detail(r) for r in rows]
            next_cursor = alerts[-1].alert_id if has_more and alerts else None
            return AlertListResponse(
                total=len(alerts),
                alerts=alerts,
                next_cursor=next_cursor,
                has_more=has_more,
            )

        # Fallback to in-memory
        in_mem = alert_manager.get_active_alerts(
            equipment_id=equipment_id, severity=severity
        )

        if cursor:
            skip = True
            filtered = []
            for a in in_mem:
                if skip:
                    if a.get("alert_id") == cursor:
                        skip = False
                    continue
                filtered.append(a)
            in_mem = filtered

        has_more = len(in_mem) > limit
        in_mem = in_mem[:limit]
        alerts = [_to_detail(a, ts_key="timestamp") for a in in_mem]
        next_cursor = alerts[-1].alert_id if has_more and alerts else None
        return AlertListResponse(
            total=len(alerts),
            alerts=alerts,
            next_cursor=next_cursor,
            has_more=has_more,
        )

    # -- POST /alerts/{alert_id}/acknowledge --------------------------------

    @app.post("/alerts/{alert_id}/acknowledge", response_model=AcknowledgeResponse)
    async def acknowledge_alert(alert_id: str, body: AcknowledgeRequest):
        now = datetime.now(timezone.utc)
        success = alert_manager.acknowledge_alert(alert_id, body.user)

        # Also update DB
        db_notifier = alert_manager.notifiers.get("database")
        if db_notifier and db_notifier.enabled:
            db_notifier.update_alert(
                alert_id=alert_id,
                status="acknowledged",
                acknowledged_by=body.user,
                acknowledged_at=now,
            )

        if not success:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        ALERTS_ACKNOWLEDGED_TOTAL.inc()
        ACTIVE_ALERTS.set(len(alert_manager.get_active_alerts()))
        return AcknowledgeResponse(
            alert_id=alert_id,
            acknowledged_by=body.user,
            acknowledged_at=now.isoformat(),
        )

    # -- POST /alerts/{alert_id}/resolve ------------------------------------

    @app.post("/alerts/{alert_id}/resolve", response_model=ResolveResponse)
    async def resolve_alert(alert_id: str):
        now = datetime.now(timezone.utc)
        success = alert_manager.resolve_alert(alert_id)

        db_notifier = alert_manager.notifiers.get("database")
        if db_notifier and db_notifier.enabled:
            db_notifier.update_alert(
                alert_id=alert_id,
                status="resolved",
                resolved_at=now,
            )

        if not success:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

        ALERTS_RESOLVED_TOTAL.inc()
        ACTIVE_ALERTS.set(len(alert_manager.get_active_alerts()))
        return ResolveResponse(alert_id=alert_id, resolved_at=now.isoformat())

    # -- GET /alerts/statistics ---------------------------------------------

    @app.get("/alerts/statistics", response_model=AlertStatsResponse)
    async def alert_statistics():
        db_notifier = alert_manager.notifiers.get("database")
        if db_notifier and db_notifier.enabled:
            stats = db_notifier.get_statistics()
            if stats:
                return AlertStatsResponse(
                    total_alerts=stats.get("total_alerts", 0),
                    active_alerts=stats.get("active_alerts", 0),
                    severity_counts=stats.get("by_severity", {}),
                )

        # In-memory fallback
        mem_stats = alert_manager.get_statistics()
        return AlertStatsResponse(
            total_alerts=mem_stats.get("total_processed", 0),
            active_alerts=mem_stats.get("active_alerts", 0),
            severity_counts=mem_stats.get("by_severity", {}),
        )

    # -- GET /health --------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health():
        import time as _time

        kafka_status = "unknown"
        if kafka_consumer:
            kafka_status = "running" if kafka_consumer._running else "stopped"
            KAFKA_CONSUMER_RUNNING.set(1 if kafka_consumer._running else 0)

        deps: Dict[str, Any] = {}
        overall = "healthy"

        # TimescaleDB check
        try:
            import psycopg2

            _t0 = _time.time()
            db_cfg = alert_manager.config.get("timescaledb", {})
            conn = psycopg2.connect(
                host=os.environ.get("DB_HOST", db_cfg.get("host", "timescaledb")),
                port=int(os.environ.get("DB_PORT", db_cfg.get("port", 5432))),
                dbname=os.environ.get(
                    "DB_NAME", db_cfg.get("database", "predictive_maintenance")
                ),
                user=os.environ.get("DB_USER", db_cfg.get("user", "pmuser")),
                password=os.environ.get(
                    "DB_PASSWORD", db_cfg.get("password", "pmpassword")
                ),
                connect_timeout=3,
            )
            conn.close()
            deps["timescaledb"] = {
                "status": "healthy",
                "latency_ms": round((_time.time() - _t0) * 1000, 2),
            }
        except Exception as exc:
            deps["timescaledb"] = {"status": "unhealthy", "details": str(exc)[:200]}
            overall = "degraded"

        # Kafka check
        try:
            from kafka import KafkaConsumer as _KC

            _t0 = _time.time()
            bootstrap = os.environ.get(
                "KAFKA_BOOTSTRAP_SERVERS",
                alert_manager.config.get("kafka", {}).get(
                    "bootstrap_servers", "kafka:29092"
                ),
            )
            _kc = _KC(bootstrap_servers=bootstrap, request_timeout_ms=3000)
            _kc.topics()
            _kc.close()
            deps["kafka"] = {
                "status": "healthy",
                "latency_ms": round((_time.time() - _t0) * 1000, 2),
            }
        except Exception as exc:
            deps["kafka"] = {"status": "unhealthy", "details": str(exc)[:200]}
            overall = "degraded"

        deps["kafka_consumer"] = {"status": kafka_status}

        return HealthResponse(
            status=overall,
            kafka_consumer=kafka_status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            dependencies=deps,
        )

    return app
