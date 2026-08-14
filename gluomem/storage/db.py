"""gluomem.storage.db — MindDB connection wrapper.

Opens a connection to a ``*.mind`` SQLite file, sets WAL + foreign_keys=ON,
loads the sqlite-vec extension, runs ``schema.sql`` (idempotent — every
statement uses ``CREATE ... IF NOT EXISTS``), and creates the two ``vec0``
virtual tables at the configured embedding dim (never hardcoded in the .sql).

Exposes the live ``sqlite3.Connection`` plus a ``meta`` key/value get/set.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import sqlite_vec

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class MindDB:
    """A connection wrapper over a ``*.mind`` SQLite file.

    Parameters
    ----------
    path:
        Filesystem path to the ``*.mind`` SQLite database. A leading ``~`` is
        expanded (the configured default is ``~/.gluomem/default.mind``). Parent
        directories are NOT created; callers should pass an existing directory
        (e.g. a pytest ``tmp_path``).
    dim:
        The active embedder's native vector dimension. The ``memory_frames_vec``
        and ``memory_frame_chunks_vec`` ``vec0`` tables are created with this
        dim. This MUST equal ``cfg.storage.vector_dim`` / the embedder dim
        (cross-validated upstream in ``AppConfig``).
    """

    def __init__(self, path: str | Path, *, dim: int) -> None:
        self.path: Path = Path(path).expanduser()
        self.dim: int = int(dim)
        # check_same_thread=False lets a connection cross threads; concurrent
        # ACCESS is still forbidden — every multi-threaded caller must hold
        # its own lock or keep all DB work on a single thread.
        self.conn: sqlite3.Connection = sqlite3.connect(
            self.path, check_same_thread=False
        )
        self._init_pragmas()
        self._load_vec_extension()
        self._apply_schema()
        self._create_vec_tables()

    # --- initialization steps ----------------------------------------------

    def _init_pragmas(self) -> None:
        """WAL mode + foreign-key enforcement."""
        # journal_mode is persistent on the file; set it before anything else.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def _load_vec_extension(self) -> None:
        """Load the sqlite-vec loadable extension into the connection."""
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)

    def _apply_schema(self) -> None:
        """Run schema.sql (idempotent — every CREATE uses IF NOT EXISTS)."""
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        self.conn.executescript(sql)

    def _create_vec_tables(self) -> None:
        """Create the two vec0 virtual tables at the configured dim.

        The dim is interpolated as a SQL literal; it comes from the active
        embedder via ``AppConfig`` and is NEVER hardcoded in ``schema.sql``.
        """
        if self.dim <= 0:
            raise ValueError(f"MindDB dim must be positive, got {self.dim}")
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_frames_vec "
            f"USING vec0(embedding float[{self.dim}])"
        )
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_frame_chunks_vec "
            f"USING vec0(embedding float[{self.dim}])"
        )

    # --- meta get/set -------------------------------------------------------

    def meta_get(self, key: str) -> str | None:
        """Return the value stored under ``key`` in ``meta``, or ``None``."""
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row is not None else None

    def meta_set(self, key: str, value: str) -> None:
        """Upsert ``key`` → ``value`` in the ``meta`` table."""
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    # --- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying connection."""
        self.conn.close()

    def __enter__(self) -> MindDB:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
