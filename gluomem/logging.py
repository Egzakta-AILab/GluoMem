"""Structured JSON logging for GluoMem HTTP service and background workers."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from gluomem.storage.scope import MemoryScope

_STANDARD_LOG_ATTRS: frozenset[str] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


def _json_default(value: Any) -> Any:
    if isinstance(value, MemoryScope):
        return value.as_dict()
    return str(value)


class JsonLogFormatter(logging.Formatter):
    """Format stdlib ``LogRecord`` instances as single-line structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC).isoformat()
        payload: dict[str, Any] = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_ATTRS or key.startswith("_"):
                continue
            if isinstance(value, MemoryScope):
                payload[key] = value.as_dict()
            else:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=_json_default)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream: Any = None,
    logger_name: str = "gluomem",
) -> logging.Logger:
    """Configure ``gluomem`` logger with ``JsonLogFormatter``."""
    logger = logging.getLogger(logger_name)
    resolved_level = (
        getattr(logging, level.upper(), logging.INFO) if isinstance(level, str) else int(level)
    )
    logger.setLevel(resolved_level)
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonLogFormatter())
    logger.handlers = [handler]
    logger.propagate = False
    return logger


__all__ = ["JsonLogFormatter", "configure_logging"]
