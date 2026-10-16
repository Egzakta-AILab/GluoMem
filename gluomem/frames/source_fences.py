"""Durable scoped source removal. No ingest flag grants restoration authority."""

from __future__ import annotations

import json

from gluomem.frames.generation import lock_scope_generation
from gluomem.storage.backend import DatabaseBackend
from gluomem.storage.scope import MemoryScope, scope_sql


class SourceRetractedError(ValueError):
    """The source was removed; retrying or force_reassert cannot restore it."""


def lock_source(backend: DatabaseBackend, scope: MemoryScope, source_ref: str) -> None:
    """Serialize commit/enqueue and retraction inside the caller's transaction.

    PostgreSQL uses a transaction advisory lock even before a fence row exists.
    SQLite reserves the writer lock before reading, avoiding a read/write race.
    """
    # Share the lifecycle/worker order even before a scope or source row exists.
    lock_scope_generation(backend, scope)
    if backend.dialect == "postgres":
        key = json.dumps([*scope.as_tuple(), source_ref], ensure_ascii=True)
        backend.scalar("SELECT pg_advisory_xact_lock(hashtextextended(?, 0))", (key,))


def assert_source_not_retracted(
    backend: DatabaseBackend, scope: MemoryScope, source_ref: str, *, lock: bool = False
) -> None:
    """Reject a removed source before compute, or under a transactional write lock.

    ``lock=True`` requires the caller to own a transaction through its enqueue or
    commit. There is intentionally no public restore operation: a future trusted
    administrator policy must explicitly authorize that separate lifecycle action.
    """
    if lock:
        lock_source(backend, scope, source_ref)
    if (
        backend.fetchone(
            f"SELECT 1 FROM source_retractions WHERE {scope_sql()} AND source_ref = ?",
            (*scope.as_tuple(), source_ref),
        )
        is not None
    ):
        raise SourceRetractedError("Source is retracted; trusted restoration policy required")
