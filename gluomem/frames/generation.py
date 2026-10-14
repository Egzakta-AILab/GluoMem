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
