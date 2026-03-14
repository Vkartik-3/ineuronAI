"""
API Key authentication for the Inference API.

Keys are read from the ``API_KEYS`` and ``ADMIN_API_KEYS`` environment
variables (comma-separated).  The ``/health`` and ``/docs`` endpoints are
exempt from authentication.
"""

import os
import logging
from typing import Optional

from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Comma-separated lists from environment
_VALID_KEYS: Optional[set] = None
_ADMIN_KEYS: Optional[set] = None

# Paths that never require authentication
PUBLIC_PATHS = frozenset(
    {
        "/",
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)

# Paths that require admin-level keys
ADMIN_PATHS = frozenset(
    {
        "/models/reload",
        "/train",
    }
)


def _load_keys() -> None:
    global _VALID_KEYS, _ADMIN_KEYS
    raw = os.environ.get("API_KEYS", "")
    _VALID_KEYS = {k.strip() for k in raw.split(",") if k.strip()} if raw else set()

    raw_admin = os.environ.get("ADMIN_API_KEYS", "")
    _ADMIN_KEYS = (
        {k.strip() for k in raw_admin.split(",") if k.strip()} if raw_admin else set()
    )

    # Admin keys are also valid regular keys
    _VALID_KEYS |= _ADMIN_KEYS

    if _VALID_KEYS:
        logger.info(
            "API authentication enabled (%d key(s), %d admin key(s))",
            len(_VALID_KEYS),
            len(_ADMIN_KEYS),
        )
    else:
        logger.warning(
            "API_KEYS not set – authentication is DISABLED. "
            "Set API_KEYS env var to require authentication."
        )


def _keys() -> set:
    if _VALID_KEYS is None:
        _load_keys()
    return _VALID_KEYS  # type: ignore[return-value]


def _admin_keys() -> set:
    if _ADMIN_KEYS is None:
        _load_keys()
    return _ADMIN_KEYS  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Security(API_KEY_HEADER),
) -> Optional[str]:
    """
    FastAPI dependency that validates the ``X-API-Key`` header.

    * Public paths (``/health``, ``/docs``, …) are always allowed.
    * If no keys are configured the check is skipped (dev mode).
    * Admin paths require a key from ``ADMIN_API_KEYS``.
    """
    path = request.url.path.rstrip("/") or "/"

    # Public endpoints are exempt
    if path in PUBLIC_PATHS:
        return api_key

    valid = _keys()

    # If no keys configured, auth is disabled (development mode)
    if not valid:
        return api_key

    if not api_key or api_key not in valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Admin check
    if path in ADMIN_PATHS:
        if api_key not in _admin_keys():
            raise HTTPException(
                status_code=403,
                detail="Admin API key required for this endpoint",
            )

    return api_key
