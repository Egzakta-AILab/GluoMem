"""Transactional scope generation counters."""

from gluomem.storage.backend import DatabaseBackend
from gluomem.storage.scope import DEFAULT_SCOPE, MemoryScope, scope_sql
from gluomem.timeutil import now_utc_iso


def current_scope_generation(backend: DatabaseBackend, scope: MemoryScope = DEFAULT_SCOPE) -> int:
    """Read the scope generation, including writes in the current transaction."""
    return int(
        backend.scalar(
            f"SELECT generation FROM scope_generations WHERE {scope_sql()}", scope.as_tuple()
        )
        or 0
    )


get_scope_generation = current_scope_generation


def lock_scope_generation(backend: DatabaseBackend, scope: MemoryScope = DEFAULT_SCOPE) -> None:
    """Lock the scope without incrementing its counter, inside a write transaction.

    Acquire this before source advisory locks or any sessions, frames, provenance,
    or jobs are written. The upsert locks even a previously unseen scope; SQLite
    also reserves its writer before the first read. Keep the actual bump at the
    operation's existing semantic boundary, not at lock acquisition.
    """
    backend.execute(
        "INSERT INTO scope_generations "
        "(tenant_id, principal_id, namespace, generation, updated_at) "
        "VALUES (?, ?, ?, 0, ?) "
        "ON CONFLICT (tenant_id, principal_id, namespace) DO UPDATE "
        "SET generation = scope_generations.generation",
        (*scope.as_tuple(), now_utc_iso()),
    )


def bump_scope_generation(
    backend: DatabaseBackend, scope: MemoryScope = DEFAULT_SCOPE, *, commit: bool = False
) -> int:
    """Atomically increment and return the scope generation."""
    row = backend.fetchone(
        "INSERT INTO scope_generations (tenant_id, principal_id, namespace, generation, "
        "updated_at) "
        "VALUES (?, ?, ?, 1, ?) ON CONFLICT (tenant_id, principal_id, namespace) DO UPDATE SET "
        "generation = scope_generations.generation + 1, updated_at = EXCLUDED.updated_at "
        "RETURNING generation",
        (*scope.as_tuple(), now_utc_iso()),
    )
    assert row is not None
    if commit:
        backend.commit()
    return int(row["generation"])
