"""
Shared structured logging configuration.

Provides a JSON-formatted logger that every service can use for consistent,
machine-parseable log output.

Usage in any service module::

    from shared.logging_config import get_logger
    logger = get_logger("my_service")
    logger.info("Processing started", extra={"equipment_id": "EQ-001", "request_id": "abc-123"})

Environment variables:
    LOG_LEVEL   — DEBUG | INFO | WARNING | ERROR (default: INFO)
    LOG_FORMAT  — json | text (default: json)
    SERVICE_NAME — service identifier included in every log line
"""

import logging
import os
import json
import sys
from datetime import datetime, timezone
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def __init__(self, service: str = "unknown"):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Standard extra fields used across services
        for field in (
            "equipment_id",
            "request_id",
            "correlation_id",
            "model",
            "rule_id",
            "alert_id",
            "duration_ms",
            "error",
        ):
            value = getattr(record, field, None)
            if value is not None:
                log_entry[field] = value

        # Exception info
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Any other extra fields passed via extra={...}
        standard_attrs = set(
            logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
        )
        for key, value in record.__dict__.items():
            if key not in standard_attrs and key not in log_entry:
                try:
                    json.dumps(value)  # only include JSON-serialisable values
                    log_entry[key] = value
                except (TypeError, ValueError):
                    pass

        return json.dumps(log_entry, default=str)


def configure_logging(
    service_name: Optional[str] = None,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
):
    """
    Configure the root logger for structured output.

    Call this once at service startup.  Subsequent ``logging.getLogger(...)``
    calls will inherit the configuration.
    """
    service = service_name or os.environ.get("SERVICE_NAME", "unknown")
    level_name = (log_level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    fmt = (log_format or os.environ.get("LOG_FORMAT", "json")).lower()

    level = getattr(logging, level_name, logging.INFO)

    # Remove any previously-attached handlers on the root logger
    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if fmt == "json":
        handler.setFormatter(JSONFormatter(service=service))
    else:
        handler.setFormatter(
            logging.Formatter(
                f"%(asctime)s [{service}] %(name)s %(levelname)s — %(message)s"
            )
        )

    root.setLevel(level)
    root.addHandler(handler)


def get_logger(name: str, service_name: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger.

    If ``configure_logging`` has not been called yet it will be called
    automatically with defaults.
    """
    root = logging.getLogger()
    if not root.handlers:
        configure_logging(service_name=service_name)
    return logging.getLogger(name)
