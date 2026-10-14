"""Shared scoped substring search for relational backends."""

from __future__ import annotations

from typing import Any

from gluomem.search.multilingual import fold_to_ascii
from gluomem.storage.backend import DatabaseBackend
from gluomem.storage.scope import MemoryScope, eligible_sql, resolve_scope, scope_sql
from gluomem.timeutil import now_iso


def search_like(
    backend: DatabaseBackend,
    query: str | list[str],
    limit: int,
    *,
    scope: MemoryScope | None = None,
    now: str | None = None,
) -> list[int]:
    """Find eligible scoped frames containing every token, with LIKE metacharacters escaped."""
    tokens = [query] if isinstance(query, str) else query
    if limit <= 0 or not tokens:
        return []

    def escape(value: str) -> str:
        return "%" + value.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_") + "%"

    clauses = " AND ".join(
        "(content LIKE ? ESCAPE '\\' OR folded_content LIKE ? ESCAPE '\\')" for _ in tokens
    )
    params: list[Any] = [
        pattern
        for token in tokens
        for pattern in (escape(token), escape(fold_to_ascii(token) or token))
    ]
    params.extend([*resolve_scope(scope).as_tuple(), now_iso(now), int(limit)])
    rows = backend.fetchall(
        f"SELECT id FROM memory_frames WHERE {clauses} "
        f"AND {scope_sql()} AND {eligible_sql()} ORDER BY id ASC LIMIT ?",
        params,
    )
    return [int(row["id"]) for row in rows]
