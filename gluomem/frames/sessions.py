"""Session (GOP) helpers."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gluomem.storage.backend import DatabaseBackend
    from gluomem.storage.db import MindDB


def get_or_create_session(
    db: MindDB | DatabaseBackend | sqlite3.Connection,
    gop_id: str,
    project_id: str | None = None,
    *,
    commit: bool = False,
) -> None:
    """Ensure a row exists in ``sessions`` for ``gop_id``."""
    if isinstance(db, sqlite3.Connection):
        db.execute(
            "INSERT OR IGNORE INTO sessions(gop_id, project_id) VALUES (?, ?)",
            (gop_id, project_id),
        )
        if commit:
            db.commit()
        return

    backend: Any = db.backend if hasattr(db, "backend") else db
    if getattr(backend, "dialect", "sqlite") == "postgres":
        sql = (
            "INSERT INTO sessions(gop_id, project_id) VALUES (?, ?) "
            "ON CONFLICT (gop_id) DO NOTHING"
        )
    else:
        sql = "INSERT OR IGNORE INTO sessions(gop_id, project_id) VALUES (?, ?)"
    backend.execute(sql, (gop_id, project_id))
    if commit:
        backend.commit()


def end_session(
    db: MindDB | DatabaseBackend | sqlite3.Connection,
    gop_id: str,
    summary: str | None = None,
    *,
    commit: bool = True,
) -> None:
    """Mark a session as ended and optionally store its summary."""
    if isinstance(db, sqlite3.Connection):
        db.execute(
            "UPDATE sessions SET status = 'ended', ended_at = datetime('now'), summary = ? "
            "WHERE gop_id = ?",
            (summary, gop_id),
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
        "WHERE gop_id = ?",
        (summary, gop_id),
    )
    if commit:
        backend.commit()
