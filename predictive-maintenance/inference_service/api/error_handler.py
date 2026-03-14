"""
Standardized error handling for the Inference API.

All errors return:
{
    "error": "error_code",
    "message": "Human-readable message",
    "details": { ... },
    "timestamp": "ISO-8601",
    "request_id": "uuid-v4"
}
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Request-ID header name
REQUEST_ID_HEADER = "X-Request-ID"


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class APIError(Exception):
    """
    Application-level API error.

    Raise this in endpoint handlers to produce a consistent JSON body.
    """

    def __init__(
        self,
        status_code: int = 500,
        error: str = "internal_error",
        message: str = "An internal error occurred",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error = error
        self.message = message
        self.details = details or {}


# ---------------------------------------------------------------------------
# Request-ID middleware
# ---------------------------------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Attach a unique ``X-Request-ID`` to every request/response cycle.

    If the caller already supplies the header the value is kept,
    otherwise a new UUID-v4 is generated.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER, str(uuid.uuid4()))
        # Stash on request state so handlers can read it
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _request_id(request: Request) -> str:
    """Retrieve request_id from state, falling back to a new UUID."""
    return getattr(getattr(request, "state", None), "request_id", str(uuid.uuid4()))


def _error_body(
    error: str,
    message: str,
    details: Any,
    request_id: str,
) -> Dict[str, Any]:
    return {
        "error": error,
        "message": message,
        "details": details if details else {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
    }


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


async def _handle_api_error(request: Request, exc: APIError) -> JSONResponse:
    rid = _request_id(request)
    logger.warning(
        "APIError [%s] %s – %s (request_id=%s)",
        exc.status_code,
        exc.error,
        exc.message,
        rid,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(exc.error, exc.message, exc.details, rid),
        headers={REQUEST_ID_HEADER: rid},
    )


async def _handle_http_exception(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    rid = _request_id(request)

    error_map = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        422: "unprocessable_entity",
        429: "rate_limit_exceeded",
        503: "service_unavailable",
    }
    error_code = error_map.get(exc.status_code, "http_error")

    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(error_code, str(exc.detail), {}, rid),
        headers={REQUEST_ID_HEADER: rid},
    )


async def _handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    rid = _request_id(request)
    details = exc.errors() if hasattr(exc, "errors") else str(exc)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_error_body(
            "validation_error",
            "Request validation failed",
            details,
            rid,
        ),
        headers={REQUEST_ID_HEADER: rid},
    )


async def _handle_unhandled(request: Request, exc: Exception) -> JSONResponse:
    rid = _request_id(request)
    logger.error("Unhandled exception (request_id=%s): %s", rid, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_body("internal_error", "An unexpected error occurred", {}, rid),
        headers={REQUEST_ID_HEADER: rid},
    )


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_error_handlers(app: FastAPI) -> None:
    """
    Register all exception handlers and middleware on *app*.

    Call this **after** app creation, **before** adding routes.
    """
    # Middleware (runs first in the stack)
    app.add_middleware(RequestIDMiddleware)

    # Exception handlers
    app.add_exception_handler(APIError, _handle_api_error)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _handle_validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _handle_unhandled)  # type: ignore[arg-type]
