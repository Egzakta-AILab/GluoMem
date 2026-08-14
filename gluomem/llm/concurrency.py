"""Process-wide concurrency throttle for extraction LLM calls.

Batch ingest fans out across conversations (one thread pool) *and* within each
conversation across lanes/observations (a second pool), so the raw thread count
can reach ``conv_parallelism * parallelism``. What actually needs bounding is not
threads but *simultaneous API calls* — OpenRouter rate-limits concurrency, not
worker count. :class:`ConcurrencyGate` is the single choke: every real extraction
call acquires a slot, so however many threads exist, at most ``limit`` calls are in
flight. Decoupling "how many pools" from "how many concurrent calls" lets
cross-conversation fan-out be aggressive without a rate-limit storm.

The gate is a thin wrapper over ``threading.BoundedSemaphore`` (a plain acquire /
release, no nesting → no deadlock) kept separate from ``modules.py`` so it is
importable and unit-testable without pulling in ``dspy``.
"""

from __future__ import annotations

import threading
from types import TracebackType


class ConcurrencyGate:
    """A reusable ``limit``-slot gate; use as a context manager per call.

    ``with gate:`` blocks until a slot is free, holds it for the body, and always
    releases on exit. Re-entrant use within a single thread would exhaust slots, so
    callers wrap exactly one LLM call per ``with`` and never nest.
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError(f"ConcurrencyGate limit must be >= 1, got {limit}")
        self.limit = limit
        self._sem = threading.BoundedSemaphore(limit)

    def __enter__(self) -> ConcurrencyGate:
        self._sem.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._sem.release()


__all__ = ["ConcurrencyGate"]
