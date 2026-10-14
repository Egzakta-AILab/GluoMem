"""SQLite ``DatabaseBackend`` and ``SqliteFts5Store`` implementations."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import sqlite_vec

from gluomem.search.multilingual import (
    expand_fts5_query,
    fold_multilingual_text,
)
from gluomem.storage.backend import IncompatibleSchemaError
from gluomem.storage.like import search_like
from gluomem.storage.scope import (
    MemoryScope,
    eligible_sql,
    resolve_scope,
    scope_sql,
)
from gluomem.timeutil import now_iso as format_now_iso

SQLITE_SCHEMA_VERSION = 1

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
        try:
            self._configure_connection()
            self._apply_schema()
        except Exception:
            self.conn.close()
            raise

    def _configure_connection(self) -> None:
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

    def _apply_schema(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if version == SQLITE_SCHEMA_VERSION:
            for table in ("sessions", "memory_frames", "knowledge_entities", "knowledge_relations"):
                columns = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
                if not {"tenant_id", "principal_id", "namespace"} <= columns:
                    raise IncompatibleSchemaError(
                        "Missing SQLite scope columns; recreate the database"
                    )
            return
        if version != 0 or tables:
            raise IncompatibleSchemaError(
                f"Unsupported SQLite schema version {version}; recreate the database"
            )
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        self.conn.executescript(
            "BEGIN IMMEDIATE;\n"
            + sql
            + f"CREATE VIRTUAL TABLE memory_frames_vec USING vec0(embedding float[{self.dim}]);"
            f"CREATE VIRTUAL TABLE memory_frame_chunks_vec USING vec0(embedding float[{self.dim}]);"
            f"PRAGMA user_version={SQLITE_SCHEMA_VERSION}; COMMIT;"
        )

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

    def executemany(self, sql: str, seq_of_params: Iterable[Sequence[Any]]) -> None:
        self.conn.executemany(sql, [tuple(p) for p in seq_of_params])

    def fetchone(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self.conn.execute(sql, tuple(params)).fetchone()
        return row if row is not None else None

    def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[Any]:
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

    def upsert(self, rowid: int, content: str, *, commit: bool = False) -> None:
        rid = int(rowid)
        folded = fold_multilingual_text(content)
        self._backend.execute("DELETE FROM memory_frames_fts WHERE rowid = ?", (rid,))
        self._backend.execute(
            "INSERT INTO memory_frames_fts(rowid, content, folded_content) VALUES (?, ?, ?)",
            (rid, content, folded),
        )
        self._backend.execute(
            "UPDATE memory_frames SET folded_content = ? WHERE id = ?",
            (folded, rid),
        )
        if commit:
            self._backend.commit()

    def delete(self, rowid: int, *, commit: bool = False) -> None:
        self._backend.execute("DELETE FROM memory_frames_fts WHERE rowid = ?", (int(rowid),))
        if commit:
            self._backend.commit()

    def search(
        self,
        sanitized_query: str,
        limit: int,
        *,
        scope: MemoryScope | None = None,
        now_iso: str | None = None,
        match_any: bool = False,
    ) -> list[int]:
        if not sanitized_query or limit <= 0:
            return []
        eff_scope = resolve_scope(scope)
        eff_now = format_now_iso(now_iso)
        match_expr = expand_fts5_query(sanitized_query, match_any=match_any) or sanitized_query
        try:
            rows = self._backend.fetchall(
                "SELECT f.rowid AS rowid FROM memory_frames_fts f "
                "JOIN memory_frames m ON m.id = f.rowid "
                f"WHERE memory_frames_fts MATCH ? AND {scope_sql('m')} "
                f"AND {eligible_sql('m')} "
                "ORDER BY f.rank, f.rowid ASC LIMIT ?",
                (match_expr, *eff_scope.as_tuple(), eff_now, int(limit)),
            )
        except sqlite3.OperationalError:
            return []
        return [int(r["rowid"]) for r in rows]

    def search_like(
        self,
        query: str | list[str],
        limit: int,
        *,
        scope: MemoryScope | None = None,
        now_iso: str | None = None,
    ) -> list[int]:
        return search_like(self._backend, query, limit, scope=scope, now=now_iso)

    def get_text(self, rowid: int) -> str | None:
        val = self._backend.scalar(
            "SELECT content FROM memory_frames_fts WHERE rowid = ?", (int(rowid),)
        )
        return str(val) if val is not None else None

    def clear(self, *, commit: bool = False) -> None:
        self._backend.execute("DELETE FROM memory_frames_fts")
        if commit:
            self._backend.commit()


__all__ = ["SqliteBackend", "SqliteFts5Store"]
