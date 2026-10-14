"""Tests for per-scope job queue backpressure, deduplication, and per-job LLM/token caps."""

from __future__ import annotations

from pathlib import Path

import pytest

from gluomem.service.jobs import (
    BackpressureError,
    JobBudgetExceededError,
    JobQueue,
)
from gluomem.storage.db import MindDB
from gluomem.storage.scope import MemoryScope


def test_per_scope_backpressure_and_idempotent_enqueue(tmp_path: Path) -> None:
    db = MindDB(tmp_path / "backpressure.mind", vector_dim=32)
    queue = JobQueue(db.backend, max_pending_per_scope=3)

    scope_a = MemoryScope("egzakta", "user_a", "personal")
    scope_b = MemoryScope("egzakta", "user_b", "personal")

    j1 = queue.enqueue(scope_a, "ingest_event", "k1", {"content": "one"})
    j2 = queue.enqueue(scope_a, "ingest_event", "k2", {"content": "two"})
    j3 = queue.enqueue(scope_a, "ingest_event", "k3", {"content": "three"})
    assert {j1.id, j2.id, j3.id} == {1, 2, 3}

    # Re-enqueueing an existing key is idempotent and does NOT trip backpressure
    j1_dup = queue.enqueue(scope_a, "ingest_event", "k1", {"content": "one"})
    assert j1_dup.id == j1.id

    # A 4th distinct pending job in scope_a raises BackpressureError
    with pytest.raises(BackpressureError) as exc_info:
        queue.enqueue(scope_a, "ingest_event", "k4", {"content": "four"})
    assert exc_info.value.scope == scope_a
    assert exc_info.value.pending == 3
    assert exc_info.value.limit == 3

    # Another scope (scope_b) is unaffected and can enqueue freely
    jb = queue.enqueue(scope_b, "ingest_event", "k1", {"content": "b-one"})
    assert jb.scope == scope_b

    # Per-job LLM call and token caps are enforced
    updated = queue.record_usage(jb.id, llm_calls=2, tokens=1500)
    assert updated.llm_calls_used == 2
    assert updated.tokens_used == 1500
    with pytest.raises(JobBudgetExceededError):
        queue.record_usage(jb.id, llm_calls=5, tokens=100)

    db.close()
