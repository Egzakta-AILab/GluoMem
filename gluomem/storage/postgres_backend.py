"""PostgreSQL + ``pgvector`` implementations of storage protocols."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np

from gluomem.extras import require
from gluomem.search.multilingual import (
    fold_multilingual_text,
    fold_query_text,
    fold_to_ascii,
)
from gluomem.storage.backend import assert_dim
from gluomem.storage.scope import (
    MemoryScope,
    eligible_sql,
    resolve_scope,
    scope_sql,
)
from gluomem.timeutil import now_utc_iso, utc_key

_PG_SCHEMA_PATH: Path = Path(__file__).with_name("schema_postgres.sql")
_ALLOWED_VEC_TABLES: frozenset[str] = frozenset(
    {"memory_frames_vec", "memory_frame_chunks_vec"}
)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _adapt_qmark_to_pyformat(sql: str) -> str:
    """Convert SQLite ``?`` parameter placeholders to psycopg ``%s`` outside string literals."""
    if "?" not in sql and "%" not in sql:
        return sql
    out: list[str] = []
    in_quote = False
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch == "'":
            out.append(ch)
            if in_quote and i + 1 < n and sql[i + 1] == "'":
                out.append("'")
                i += 2
                continue
            in_quote = not in_quote
        elif ch == "?" and not in_quote:
            out.append("%s")
        elif ch == "%" and not in_quote:
            out.append("%%")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


class _PgRow:
    """``sqlite3.Row``-compatible row supporting both positional and key access."""

    __slots__ = ("_cols", "_vals", "_map")

    def __init__(self, row: Mapping[str, Any]) -> None:
        self._cols = tuple(row.keys())
        self._vals = tuple(row.values())
        self._map = dict(row)

    def __getitem__(self, key: int | str | slice) -> Any:
        if isinstance(key, str):
            return self._map[key]
        return self._vals[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._vals)

    def __len__(self) -> int:
        return len(self._vals)

    def keys(self) -> list[str]:
        return list(self._cols)

    def values(self) -> list[Any]:
        return list(self._vals)


class PostgresBackend:
    """PostgreSQL database backend with ``pgvector`` and schema isolation."""

    dialect: Literal["sqlite", "postgres"] = "postgres"

    def __init__(
        self,
        dsn: str,
        *,
        dim: int = 384,
        postgres_schema: str = "gluomem",
    ) -> None:
        if dim <= 0:
            raise ValueError(f"vector dim must be positive, got {dim}")
        if not dsn or not dsn.strip():
            raise ValueError("PostgresBackend requires a non-empty dsn")
        if not _IDENT_RE.match(postgres_schema):
            raise ValueError(f"Invalid postgres_schema identifier: {postgres_schema!r}")

        require("postgres", "PostgresBackend")
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg.rows import dict_row

        self.dsn: str = dsn
        self.dim: int = dim
        self.postgres_schema: str = postgres_schema
        self.path: Path = Path(f"postgres://{postgres_schema}")
        self._tx_depth: int = 0

        self.conn: Any = psycopg.connect(dsn, row_factory=dict_row)
        self._apply_schema(register_vector)

    def _apply_schema(self, register_vector_fn: Any) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.postgres_schema}"')
            cur.execute(f'SET search_path TO "{self.postgres_schema}", public')
            ddl = _PG_SCHEMA_PATH.read_text(encoding="utf-8").replace(
                "__VECTOR_DIM__", str(self.dim)
            )
            cur.execute(ddl)
        self.conn.commit()
        register_vector_fn(self.conn)

    @property
    def in_transaction(self) -> bool:
        return self._tx_depth > 0

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        adapted = _adapt_qmark_to_pyformat(sql)
        return self.conn.execute(adapted, tuple(params))

    def insert_returning_id(
        self, sql: str, params: Sequence[Any] = (), *, id_col: str = "id"
    ) -> int:
        query = sql.rstrip().rstrip(";")
        if "RETURNING" not in query.upper():
            query = f"{query} RETURNING {id_col}"
        row = self.fetchone(query, params)
        if row is None:
            raise RuntimeError("INSERT RETURNING did not return a row")
        return int(row[0])

    def executemany(
        self, sql: str, seq_of_params: Iterable[Sequence[Any]]
    ) -> None:
        adapted = _adapt_qmark_to_pyformat(sql)
        with self.conn.cursor() as cur:
            cur.executemany(adapted, [tuple(p) for p in seq_of_params])

    def fetchone(
        self, sql: str, params: Sequence[Any] = ()
    ) -> Any:
        row = self.execute(sql, params).fetchone()
        return _PgRow(row) if row is not None else None

    def fetchall(
        self, sql: str, params: Sequence[Any] = ()
    ) -> list[Any]:
        rows = self.execute(sql, params).fetchall()
        return [_PgRow(r) for r in rows]

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self.fetchone(sql, params)
        if row is None:
            return None
        return row[0]

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self._tx_depth += 1
        try:
            with self.conn.transaction():
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


class PgVectorStore:
    """``VectorStore`` backed by PostgreSQL + ``pgvector``."""

    def __init__(self, backend: PostgresBackend, table: str, dim: int) -> None:
        if table not in _ALLOWED_VEC_TABLES:
            raise ValueError(
                f"Invalid vector table {table!r}; must be one of {sorted(_ALLOWED_VEC_TABLES)}"
            )
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._backend: PostgresBackend = backend
        self.table: str = table
        self.dim: int = dim

    def add(self, rowid: int, vec: np.ndarray, *, commit: bool = True) -> None:
        rid = int(rowid)
        f32 = assert_dim(vec, self.dim)
        self._backend.execute(
            f"INSERT INTO {self.table}(rowid, embedding) VALUES (?, ?) "
            f"ON CONFLICT (rowid) DO UPDATE SET embedding = EXCLUDED.embedding",
            (rid, f32),
        )
        if commit:
            self._backend.commit()

    def add_batch(
        self, items: list[tuple[int, np.ndarray]], *, commit: bool = True
    ) -> None:
        if not items:
            return
        prepared = [(int(rowid), assert_dim(vec, self.dim)) for rowid, vec in items]
        self._backend.executemany(
            f"INSERT INTO {self.table}(rowid, embedding) VALUES (?, ?) "
            f"ON CONFLICT (rowid) DO UPDATE SET embedding = EXCLUDED.embedding",
            prepared,
        )
        if commit:
            self._backend.commit()

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        scope: MemoryScope | None = None,
        now_iso: str | None = None,
    ) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        f32 = assert_dim(query, self.dim)
        eff_scope = resolve_scope(scope)
        eff_now = utc_key(now_iso) if now_iso is not None else now_utc_iso()
        if self.table == "memory_frames_vec":
            join = "FROM memory_frames_vec v JOIN memory_frames m ON m.id = v.rowid"
        else:
            join = (
                "FROM memory_frame_chunks_vec v "
                "JOIN memory_frame_chunks c ON c.id = v.rowid "
                "JOIN memory_frames m ON m.id = c.frame_id"
            )
        sql = (
            f"SELECT v.rowid AS rowid, (v.embedding <-> ?) AS distance {join} "
            f"WHERE {scope_sql('m')} AND {eligible_sql('m')} "
            "ORDER BY v.embedding <-> ? ASC, v.rowid ASC LIMIT ?"
        )
        rows = self._backend.fetchall(
            sql, (f32, *eff_scope.as_tuple(), eff_now, f32, int(k))
        )
        return [(int(r["rowid"]), float(r["distance"])) for r in rows]

    def delete(self, rowid: int, *, commit: bool = True) -> None:
        self._backend.execute(
            f"DELETE FROM {self.table} WHERE rowid = ?", (int(rowid),)
        )
        if commit:
            self._backend.commit()

    def clear(self, *, commit: bool = True) -> None:
        self._backend.execute(f"DELETE FROM {self.table}")
        if commit:
            self._backend.commit()

    def count(self) -> int:
        val = self._backend.scalar(f"SELECT COUNT(*) FROM {self.table}")
        return int(val) if val is not None else 0

    def has(self, rowid: int) -> bool:
        row = self._backend.fetchone(
            f"SELECT 1 FROM {self.table} WHERE rowid = ? LIMIT 1", (int(rowid),)
        )
        return row is not None

    def count_by_ids(self, rowids: Sequence[int]) -> int:
        if not rowids:
            return 0
        placeholders = ",".join("?" for _ in rowids)
        val = self._backend.scalar(
            f"SELECT COUNT(*) FROM {self.table} WHERE rowid IN ({placeholders})",
            [int(rid) for rid in rowids],
        )
        return int(val) if val is not None else 0

    def ids(self) -> set[int]:
        rows = self._backend.fetchall(f"SELECT rowid FROM {self.table}")
        return {int(r["rowid"]) for r in rows}


class PgTsVectorStore:
    """Full-text search store backed by PostgreSQL ``tsvector`` + GIN index."""

    def __init__(self, backend: PostgresBackend) -> None:
        self._backend: PostgresBackend = backend

    def upsert(
        self, rowid: int, content: str, *, commit: bool = False
    ) -> None:
        folded = fold_multilingual_text(content)
        self._backend.execute(
            "UPDATE memory_frames SET folded_content = ? WHERE id = ?",
            (folded, int(rowid)),
        )
        self._backend.execute(
            "INSERT INTO memory_frames_fts(rowid, content, folded_content) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT (rowid) DO UPDATE SET "
            "content = EXCLUDED.content, folded_content = EXCLUDED.folded_content",
            (int(rowid), content, folded),
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
        plain_terms = sanitized_query.replace('"', " ").strip()
        if not plain_terms:
            return []
        folded_terms = fold_query_text(plain_terms) or plain_terms
        eff_scope = resolve_scope(scope)
        eff_now = utc_key(now_iso) if now_iso is not None else now_utc_iso()
        tsq_expr = (
            "(websearch_to_tsquery('simple', ?) || websearch_to_tsquery('english', ?))"
        )
        rows = self._backend.fetchall(
            f"SELECT f.rowid AS rowid FROM memory_frames_fts f "
            f"JOIN memory_frames m ON m.id = f.rowid "
            f"WHERE f.tsv @@ {tsq_expr} "
            f"AND {scope_sql('m')} AND {eligible_sql('m')} "
            f"ORDER BY ts_rank_cd(f.tsv, {tsq_expr}) DESC, f.rowid ASC "
            f"LIMIT ?",
            (
                folded_terms,
                plain_terms,
                *eff_scope.as_tuple(),
                eff_now,
                folded_terms,
                plain_terms,
                int(limit),
            ),
        )
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


__all__ = [
    "PgTsVectorStore",
    "PgVectorStore",
    "PostgresBackend",
    "_adapt_qmark_to_pyformat",
]
