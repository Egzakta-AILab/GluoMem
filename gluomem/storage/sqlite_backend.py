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
    fold_to_ascii,
)
from gluomem.storage.scope import (
    MemoryScope,
    eligible_sql,
    resolve_scope,
    scope_sql,
)
from gluomem.timeutil import now_utc_iso, utc_key

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

    def _ensure_legacy_columns(self) -> None:
        existing_tables = {
            str(r[0])
            for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        table_migrations: dict[str, list[tuple[str, str]]] = {
            "sessions": [
                ("tenant_id", "TEXT NOT NULL DEFAULT 'local'"),
                ("principal_id", "TEXT NOT NULL DEFAULT 'default'"),
                ("namespace", "TEXT NOT NULL DEFAULT 'personal'"),
            ],
            "memory_frames": [
                ("tenant_id", "TEXT NOT NULL DEFAULT 'local'"),
                ("principal_id", "TEXT NOT NULL DEFAULT 'default'"),
                ("namespace", "TEXT NOT NULL DEFAULT 'personal'"),
                ("folded_content", "TEXT NOT NULL DEFAULT ''"),
                ("kind", "TEXT NOT NULL DEFAULT 'fact'"),
                ("status", "TEXT NOT NULL DEFAULT 'active'"),
                ("confidence", "REAL NOT NULL DEFAULT 1.0"),
                ("superseded_by", "INTEGER REFERENCES memory_frames(id)"),
                ("supersedes", "INTEGER REFERENCES memory_frames(id)"),
                ("expires_at", "TEXT"),
                ("pinned", "INTEGER NOT NULL DEFAULT 0"),
                ("verified_by_user", "INTEGER NOT NULL DEFAULT 0"),
            ],
            "knowledge_entities": [
                ("tenant_id", "TEXT NOT NULL DEFAULT 'local'"),
                ("principal_id", "TEXT NOT NULL DEFAULT 'default'"),
                ("namespace", "TEXT NOT NULL DEFAULT 'personal'"),
                ("canonical_name", "TEXT"),
            ],
            "knowledge_relations": [
                ("tenant_id", "TEXT NOT NULL DEFAULT 'local'"),
                ("principal_id", "TEXT NOT NULL DEFAULT 'default'"),
                ("namespace", "TEXT NOT NULL DEFAULT 'personal'"),
            ],
        }
        for table, col_defs in table_migrations.items():
            if table not in existing_tables:
                continue
            cols = {
                str(r[1])
                for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for col_name, col_ddl in col_defs:
                if col_name not in cols:
                    self.conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col_name} {col_ddl}"
                    )

        fts_rows_to_restore: list[tuple[int, str]] = []
        if "memory_frames_fts" in existing_tables:
            fts_cols = {
                str(r[1])
                for r in self.conn.execute(
                    "PRAGMA table_info(memory_frames_fts)"
                ).fetchall()
            }
            if "folded_content" not in fts_cols:
                fts_rows_to_restore = [
                    (int(r[0]), str(r[1]))
                    for r in self.conn.execute(
                        "SELECT rowid, content FROM memory_frames_fts"
                    ).fetchall()
                ]
                self.conn.execute("DROP TABLE memory_frames_fts")
        self._fts_rows_to_restore: list[tuple[int, str]] = fts_rows_to_restore

    def _apply_schema(self) -> None:
        self._ensure_legacy_columns()
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        self.conn.executescript(sql)
        if getattr(self, "_fts_rows_to_restore", None):
            for rid, content in self._fts_rows_to_restore:
                folded = fold_multilingual_text(content)
                self.conn.execute(
                    "INSERT INTO memory_frames_fts(rowid, content, folded_content) "
                    "VALUES (?, ?, ?)",
                    (rid, content, folded),
                )
                self.conn.execute(
                    "UPDATE memory_frames SET folded_content = ? WHERE id = ?",
                    (folded, rid),
                )
            self._fts_rows_to_restore = []
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
        folded = fold_multilingual_text(content)
        self._backend.execute("DELETE FROM memory_frames_fts WHERE rowid = ?", (rid,))
        self._backend.execute(
            "INSERT INTO memory_frames_fts(rowid, content, folded_content) "
            "VALUES (?, ?, ?)",
            (rid, content, folded),
        )
        self._backend.execute(
            "UPDATE memory_frames SET folded_content = ? WHERE id = ?",
            (folded, rid),
        )
        if commit:
            self._backend.commit()

    def delete(self, rowid: int, *, commit: bool = False) -> None:
        self._backend.execute(
            "DELETE FROM memory_frames_fts WHERE rowid = ?", (int(rowid),)
        )
        if commit:
            self._backend.commit()

    def search(
        self,
        sanitized_query: str,
        limit: int,
        *,
        scope: MemoryScope | None = None,
        now_iso: str | None = None,
    ) -> list[int]:
        if not sanitized_query or limit <= 0:
            return []
        eff_scope = resolve_scope(scope)
        eff_now = utc_key(now_iso) if now_iso is not None else now_utc_iso()
        match_expr = expand_fts5_query(sanitized_query) or sanitized_query
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
        if limit <= 0:
            return []
        tokens = [query] if isinstance(query, str) else query
        if not tokens:
            return []
        eff_scope = resolve_scope(scope)
        eff_now = utc_key(now_iso) if now_iso is not None else now_utc_iso()

        def _esc(s: str) -> str:
            return (
                "%"
                + s.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
                + "%"
            )

        clause_parts: list[str] = []
        params: list[Any] = []
        for t in tokens:
            raw_pat = _esc(t)
            folded_pat = _esc(fold_to_ascii(t) or t)
            clause_parts.append(
                "(content LIKE ? ESCAPE '\\' OR folded_content LIKE ? ESCAPE '\\')"
            )
            params.extend([raw_pat, folded_pat])

        clauses = " AND ".join(clause_parts)
        params.extend([*eff_scope.as_tuple(), eff_now, int(limit)])
        rows = self._backend.fetchall(
            f"SELECT id FROM memory_frames WHERE {clauses} "
            f"AND {scope_sql()} AND {eligible_sql()} ORDER BY id ASC LIMIT ?",
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
