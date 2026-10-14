"""Session (GOP) helpers."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

from gluomem.storage.scope import MemoryScope, resolve_scope, scope_sql

if TYPE_CHECKING:
    from gluomem.storage.backend import DatabaseBackend
    from gluomem.storage.db import MindDB


def get_or_create_session(
    db: MindDB | DatabaseBackend | sqlite3.Connection,
    gop_id: str,
    project_id: str | None = None,
    *,
    scope: MemoryScope | None = None,
    commit: bool = False,
) -> None:
    """Ensure a row exists in ``sessions`` for ``(scope, gop_id)``."""
    eff_scope = resolve_scope(scope)
    params = (
        *eff_scope.as_tuple(),
        gop_id,
        project_id,
    )
    if isinstance(db, sqlite3.Connection):
        db.execute(
            "INSERT OR IGNORE INTO sessions"
            "(tenant_id, principal_id, namespace, gop_id, project_id) "
            "VALUES (?, ?, ?, ?, ?)",
            params,
        )
        if commit:
            db.commit()
        return

    backend: Any = db.backend if hasattr(db, "backend") else db
    if getattr(backend, "dialect", "sqlite") == "postgres":
        sql = (
            "INSERT INTO sessions"
            "(tenant_id, principal_id, namespace, gop_id, project_id) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (tenant_id, principal_id, namespace, gop_id) DO NOTHING"
        )
    else:
        sql = (
            "INSERT OR IGNORE INTO sessions"
            "(tenant_id, principal_id, namespace, gop_id, project_id) "
            "VALUES (?, ?, ?, ?, ?)"
        )
    backend.execute(sql, params)
    if commit:
        backend.commit()


def end_session(
    db: MindDB | DatabaseBackend | sqlite3.Connection,
    gop_id: str,
    summary: str | None = None,
    *,
    scope: MemoryScope | None = None,
    commit: bool = True,
) -> None:
    """Mark a session as ended and optionally store its summary."""
    eff_scope = resolve_scope(scope)
    params = (
        summary,
        *eff_scope.as_tuple(),
        gop_id,
    )
    if isinstance(db, sqlite3.Connection):
        db.execute(
            "UPDATE sessions SET status = 'ended', ended_at = datetime('now'), summary = ? "
            f"WHERE {scope_sql()} AND gop_id = ?",
            params,
        )
        if commit:
            db.commit()
        return

    backend: Any = db.backend if hasattr(db, "backend") else db
    now_expr = (
        " CURRENT_TIMESTAMP::text "
        if getattr(backend, "dialect", "sqlite") == "postgres"
        else " datetime('now') "
    )
    backend.execute(
        f"UPDATE sessions SET status = 'ended', ended_at = {now_expr}, summary = ? "
        f"WHERE {scope_sql()} AND gop_id = ?",
        params,
    )
    if commit:
        backend.commit()
