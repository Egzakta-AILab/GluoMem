"""SQLite ``DatabaseBackend`` and ``SqliteFts5Store`` implementations."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import sqlite_vec

_SCHEMA_PATH: Path = Path(__file__).with_name("schema.sql")


class SqliteBackend:
    """SQLite database backend with WAL, foreign keys, and ``sqlite-vec`` loaded."""

    dialect: Literal["sqlite", "postgres"] = "sqlite"

    def __init__(self, path: Path | str, *, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError(f"vector dim must be positive, got {dim}")
        raw_str = str(path)
        if raw_str == ":memory:" or raw_str.startswith("file:"):
            self.path: Path = Path(raw_str)
            connect_target = raw_str
        else:
            resolved = Path(path).expanduser()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self.path = resolved
            connect_target = str(resolved)

        self.dim: int = dim
        self._tx_depth: int = 0
        self.conn: sqlite3.Connection = sqlite3.connect(
            connect_target,
            uri=raw_str.startswith("file:"),
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        self._configure_connection()
        self._apply_schema()

    def _configure_connection(self) -> None:
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

    def _apply_schema(self) -> None:
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        self.conn.executescript(sql)
        # Ensure canonical_name exists on pre-existing files opened from disk.
        cols = {
            str(r[1])
            for r in self.conn.execute("PRAGMA table_info(knowledge_entities)").fetchall()
        }
        if "canonical_name" not in cols:
            self.conn.execute(
                "ALTER TABLE knowledge_entities ADD COLUMN canonical_name TEXT"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_entities_canonical "
                "ON knowledge_entities(canonical_name)"
            )
        self.conn.execute(
            "UPDATE knowledge_entities SET canonical_name = LOWER(name) "
            "WHERE canonical_name IS NULL AND name IS NOT NULL"
        )
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_frames_vec "
            f"USING vec0(embedding float[{self.dim}])"
        )
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_frame_chunks_vec "
            f"USING vec0(embedding float[{self.dim}])"
        )
        self.conn.commit()

    @property
    def in_transaction(self) -> bool:
        return self._tx_depth > 0

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, tuple(params))

    def insert_returning_id(
        self, sql: str, params: Sequence[Any] = (), *, id_col: str = "id"
    ) -> int:
        del id_col
        cur = self.conn.execute(sql, tuple(params))
        if cur.lastrowid is None:
            raise RuntimeError("INSERT did not return a lastrowid")
        return int(cur.lastrowid)

    def executemany(
        self, sql: str, seq_of_params: Iterable[Sequence[Any]]
    ) -> None:
        self.conn.executemany(sql, [tuple(p) for p in seq_of_params])

    def fetchone(
        self, sql: str, params: Sequence[Any] = ()
    ) -> Any:
        row = self.conn.execute(sql, tuple(params)).fetchone()
        return row if row is not None else None

    def fetchall(
        self, sql: str, params: Sequence[Any] = ()
    ) -> list[Any]:
        rows = self.conn.execute(sql, tuple(params)).fetchall()
        return list(rows)

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self.conn.execute(sql, tuple(params)).fetchone()
        return row[0] if row is not None else None

    @contextmanager
    def transaction(self) -> Iterator[None]:
        if self._tx_depth == 0:
            self._tx_depth += 1
            try:
                yield
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
            finally:
                self._tx_depth -= 1
        else:
            self._tx_depth += 1
            try:
                yield
            finally:
                self._tx_depth -= 1

    def commit(self) -> None:
        if self._tx_depth == 0:
            self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

    def close(self) -> None:
        self.conn.close()


class SqliteFts5Store:
    """Full-text search store backed by SQLite FTS5 (`memory_frames_fts`)."""

    def __init__(self, backend: SqliteBackend) -> None:
        self._backend: SqliteBackend = backend

    def upsert(
        self, rowid: int, content: str, *, commit: bool = False
    ) -> None:
        rid = int(rowid)
        self._backend.execute("DELETE FROM memory_frames_fts WHERE rowid = ?", (rid,))
        self._backend.execute(
            "INSERT INTO memory_frames_fts(rowid, content) VALUES (?, ?)",
            (rid, content),
        )
        if commit:
            self._backend.commit()

    def delete(self, rowid: int, *, commit: bool = False) -> None:
        self._backend.execute(
            "DELETE FROM memory_frames_fts WHERE rowid = ?", (int(rowid),)
        )
        if commit:
            self._backend.commit()

    def search(self, sanitized_query: str, limit: int) -> list[int]:
        if not sanitized_query or limit <= 0:
            return []
        try:
            rows = self._backend.fetchall(
                "SELECT f.rowid AS rowid FROM memory_frames_fts f "
                "JOIN memory_frames m ON m.id = f.rowid "
                "WHERE memory_frames_fts MATCH ? AND m.trust != 'quarantined' "
                "ORDER BY f.rank, f.rowid ASC LIMIT ?",
                (sanitized_query, int(limit)),
            )
        except sqlite3.OperationalError:
            return []
        return [int(r["rowid"]) for r in rows]

    def search_like(self, query: str | list[str], limit: int) -> list[int]:
        if limit <= 0:
            return []
        tokens = [query] if isinstance(query, str) else query
        if not tokens:
            return []
        clauses = " AND ".join(
            "content LIKE ? ESCAPE '\\'" for _ in tokens
        )
        params: list[Any] = [
            "%" + t.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_") + "%"
            for t in tokens
        ]
        params.append(int(limit))
        rows = self._backend.fetchall(
            f"SELECT id FROM memory_frames "
            f"WHERE {clauses} AND trust != 'quarantined' "
            f"ORDER BY id ASC LIMIT ?",
            params,
        )
        return [int(r["id"]) for r in rows]

    def get_content(self, rowid: int) -> str | None:
        val = self._backend.scalar(
            "SELECT content FROM memory_frames_fts WHERE rowid = ?",
            (int(rowid),),
        )
        return str(val) if val is not None else None

    def get_text(self, rowid: int) -> str | None:
        return self.get_content(rowid)

    def id_to_content(self) -> dict[int, str]:
        rows = self._backend.fetchall(
            "SELECT rowid, content FROM memory_frames_fts"
        )
        return {int(r["rowid"]): str(r["content"]) for r in rows}

    def clear(self, *, commit: bool = False) -> None:
        self._backend.execute("DELETE FROM memory_frames_fts")
        if commit:
            self._backend.commit()


__all__ = ["SqliteBackend", "SqliteFts5Store"]
