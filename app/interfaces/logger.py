"""Request/response logging utilities for FastAPI."""

from __future__ import annotations

import logging
import os
import time
from contextvars import ContextVar
from typing import Optional
from uuid import uuid4

from fastapi import Request, Response

REQUEST_ID_CTX: ContextVar[Optional[str]] = ContextVar(
    "request_id", default=None)
TRACE_ID_CTX: ContextVar[Optional[str]] = ContextVar(
    "trace_id", default=None)


class RequestIdFilter(logging.Filter):
    """Inject request id into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQUEST_ID_CTX.get() or "-"
        record.trace_id = TRACE_ID_CTX.get() or "-"
        return True


class ColorFormatter(logging.Formatter):
    """Colorize entire log lines by level."""

    LEVEL_COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m",
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str, *, use_color: bool = True, datefmt: str | None = None):
        super().__init__(fmt, datefmt=datefmt)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        if not self._use_color:
            return base
        color = self.LEVEL_COLORS.get(record.levelno)
        if not color:
            return base
        return f"{color}{base}{self.RESET}"


def configure_logger(name: str = "text_extractor_api") -> logging.Logger:
    """Create or reuse a configured logger for the API."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    use_color = os.getenv("NO_COLOR") is None
    formatter = ColorFormatter(
        fmt=(
            "[%(levelname)s] - %(asctime)s.%(msecs)03d - "
            "[trace=%(trace_id)s] - %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        use_color=use_color,
    )
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def request_logging_middleware(logger: logging.Logger):
    """Return a middleware that logs request/response lifecycle."""

    async def middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or uuid4().hex
        trace_id = uuid4().hex
        request_token = REQUEST_ID_CTX.set(request_id)
        trace_token = TRACE_ID_CTX.set(trace_id)
        start = time.monotonic()
        response: Optional[Response] = None
        path = request.url.path
        if request.url.query:
            path = f"{path}?{request.url.query}"
        client = request.client.host if request.client else "-"
        try:
            logger.info(
                "HTTP %s %s start client=%s",
                request.method,
                path,
                client,
            )
            response = await call_next(request)
            response.headers["X-Request-Id"] = request_id
            return response
        except Exception:
            logger.exception("HTTP %s %s error client=%s",
                             request.method, path, client)
            raise
        finally:
            duration_ms = (time.monotonic() - start) * 1000.0
            status_code = response.status_code if response else 500
            message = "HTTP %s %s status=%s duration_ms=%.2f client=%s"
            if status_code >= 500:
                logger.error(
                    message,
                    request.method,
                    path,
                    status_code,
                    duration_ms,
                    client,
                )
            elif status_code >= 400:
                logger.warning(
                    message,
                    request.method,
                    path,
                    status_code,
                    duration_ms,
                    client,
                )
            else:
                logger.info(
                    message,
                    request.method,
                    path,
                    status_code,
                    duration_ms,
                    client,
                )
            REQUEST_ID_CTX.reset(request_token)
            TRACE_ID_CTX.reset(trace_token)

    return middleware
