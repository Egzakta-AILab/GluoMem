"""Transient extraction retries with bounded backoff and non-echoing telemetry."""

from __future__ import annotations

import logging
import math
import time
from asyncio import CancelledError
from typing import TYPE_CHECKING, TypeVar
from uuid import uuid4

from dspy.utils.exceptions import (
    AdapterParseError,
    LMAuthError,
    LMConfigurationError,
    LMInvalidRequestError,
    LMRateLimitError,
    LMServerError,
    LMTimeoutError,
    LMTransportError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gluomem.config.app_config import DspyRetryConfig

_LOG = logging.getLogger(__name__)
_T = TypeVar("_T")
_ERRORS = (
    (CancelledError, "cancelled", False),
    (LMRateLimitError, "rate_limit", True),
    (LMServerError, "server", True),
    (LMTimeoutError, "timeout", True),
    (LMTransportError, "transport", True),
    (LMAuthError, "auth", False),
    (LMConfigurationError, "configuration", False),
    (LMInvalidRequestError, "invalid_request", False),
    (AdapterParseError, "output_validation", False),
)


def run_with_retry(operation: Callable[[], _T], *, task: str, policy: DspyRetryConfig) -> _T:
    """Retry only trusted transient LM error types, never arbitrary error strings.

    The operation owns its concurrency slot: it is released before any backoff.
    No prompt, response, endpoint, credential or arbitrary exception text is logged.
    """
    operation_id = uuid4().hex
    delay = policy.initial_delay_seconds
    for attempt in range(1, policy.max_attempts + 1):
        started = time.monotonic()
        fields = {
            "event": "llm_attempt",
            "operation_id": operation_id,
            "task": task,
            "attempt": attempt,
            "max_attempts": policy.max_attempts,
        }
        try:
            result = operation()
        except BaseException as exc:
            category, retryable = next(
                ((label, transient) for cls, label, transient in _ERRORS if isinstance(exc, cls)),
                ("other", False),
            )
            status = exc.__dict__.get("status")
            retrying = retryable and attempt < policy.max_attempts
            scheduled_delay = delay
            blocked_reason = None
            hint = exc.__dict__.get("retry_after")
            if (
                retrying
                and (type(hint) is int or (type(hint) is float and math.isfinite(hint)))
                and hint >= 0
            ):
                if hint > policy.max_delay_seconds:
                    retrying = False
                    blocked_reason = "retry_after_exceeds_cap"
                else:
                    scheduled_delay = max(delay, float(hint))
            outcome = "retrying" if retrying else "failed"
            _LOG.warning(
                "LLM attempt %s/%s %s",
                attempt,
                policy.max_attempts,
                outcome,
                extra={
                    **fields,
                    "outcome": outcome,
                    "error_category": category,
                    "http_status": status if type(status) is int and 100 <= status <= 599 else None,
                    "elapsed_seconds": time.monotonic() - started,
                    "retry_delay_seconds": scheduled_delay if retrying else 0.0,
                    "retry_blocked_reason": blocked_reason,
                },
            )
            if not retrying:
                raise
            time.sleep(scheduled_delay)
            delay = min(policy.max_delay_seconds, delay * 2)
        else:
            _LOG.info(
                "LLM attempt %s/%s succeeded",
                attempt,
                policy.max_attempts,
                extra={
                    **fields,
                    "outcome": "succeeded",
                    "error_category": None,
                    "http_status": None,
                    "elapsed_seconds": time.monotonic() - started,
                    "retry_delay_seconds": 0.0,
                },
            )
            return result
    raise AssertionError("retry policy requires at least one attempt")
