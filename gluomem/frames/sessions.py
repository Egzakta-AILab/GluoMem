"""gluomem.frames.sessions — GOP session lifecycle.

A *session* is the Group of Pictures (GOP) container that frames belong to.
``memory_frames.gop_id`` has a foreign key to ``sessions(gop_id)``, so a
session row MUST exist before any frame is inserted. This module provides the
``get_or_create_session`` helper used by ``FrameStore`` to ensure the FK target
exists (insert-if-absent, idempotent).
"""

from __future__ import annotations

import sqlite3


def get_or_create_session(
    conn: sqlite3.Connection,
    gop_id: str,
    *,
    project_id: str | None = None,
) -> int:
    """Ensure a session row exists for ``gop_id`` and return its integer id.

    Inserts if absent (``INSERT OR IGNORE``), then reads back the id. Idempotent:
    calling twice with the same ``gop_id`` returns the same session id and does
    NOT create a duplicate (``sessions.gop_id`` is UNIQUE).
    """
    conn.execute(
        "INSERT OR IGNORE INTO sessions(gop_id, project_id) VALUES (?, ?)",
        (gop_id, project_id),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM sessions WHERE gop_id=?", (gop_id,)
    ).fetchone()
    if row is None:  # pragma: no cover — INSERT OR IGNORE guarantees the row
        raise RuntimeError(f"session for gop_id={gop_id!r} vanished after upsert")
    return int(row[0])
