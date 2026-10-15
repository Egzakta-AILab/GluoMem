"""PostgreSQL + ``pgvector`` implementations of storage protocols."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np

from gluomem.extras import require
from gluomem.migrations.runner import get_heads, validate_schema_name
from gluomem.search.multilingual import (
    fold_multilingual_text,
    fold_query_text,
)
from gluomem.storage.backend import IncompatibleSchemaError, assert_dim
from gluomem.storage.like import search_like
from gluomem.storage.scope import (
    MemoryScope,
    eligible_sql,
    resolve_scope,
    scope_sql,
)
from gluomem.timeutil import now_iso as format_now_iso

_ALLOWED_VEC_TABLES: frozenset[str] = frozenset({"memory_frames_vec", "memory_frame_chunks_vec"})


def _adapt_qmark_to_pyformat(sql: str) -> str:
    """Adapt qmarks outside SQL literals/comments and escape all psycopg percent signs."""
    out: list[str] = []
    i = 0
    while i < len(sql):
        start = i
        if sql.startswith("--", i):
            end = sql.find("\n", i)
            i = len(sql) if end < 0 else end
        elif sql.startswith("/*", i):
            depth = 1
            i += 2
            while i < len(sql) and depth:
                if sql.startswith("/*", i):
                    depth += 1
                    i += 2
                elif sql.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
        elif sql[i] in ("'", '"'):
            quote = sql[i]
            escaped = (
                quote == "'"
                and i > 0
                and sql[i - 1] in "eE"
                and (i < 2 or not (sql[i - 2].isalnum() or sql[i - 2] == "_"))
            )
            i += 1
            while i < len(sql):
                if escaped and sql[i] == "\\":
                    i += 2
                elif sql[i] == quote:
                    i += 1
                    if i >= len(sql) or sql[i] != quote:
                        break
                    i += 1
                else:
                    i += 1
        elif sql[i] == "$" and (match := re.match(r"\$(?:[a-zA-Z_]\w*)?\$", sql[i:])):
            delimiter = match.group()
            end = sql.find(delimiter, i + len(delimiter))
            i = len(sql) if end < 0 else end + len(delimiter)
        else:
            out.append("%s" if sql[i] == "?" else "%%" if sql[i] == "%" else sql[i])
            i += 1
            continue
        out.append(sql[start:i].replace("%", "%%"))
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


def _configure_connection(conn: Any, postgres_schema: str) -> None:
    """Configure session parameters and register pgvector on a raw psycopg connection."""
    validate_schema_name(postgres_schema)
    from pgvector.psycopg import register_vector

    conn.execute(f"SET search_path TO {postgres_schema}, public")
    conn.execute("SET hnsw.iterative_scan = strict_order")
    conn.execute("SET hnsw.max_scan_tuples = 20000")
    conn.execute("SET hnsw.scan_mem_multiplier = 4")
    register_vector(conn)


def _verify_schema_conn(conn: Any, postgres_schema: str, dim: int) -> None:
    """Verify Alembic revision and pgvector column dimensions on an open connection."""
    version_table = f"{postgres_schema}.alembic_version"
    exists = conn.execute("SELECT to_regclass(%s)", (version_table,)).fetchone()
    if exists is None or exists["to_regclass"] is None:
        raise IncompatibleSchemaError(
            "PostgreSQL schema is not versioned; recreate the database if it is legacy, "
            "then run gluomem migrate upgrade --dsn ... --schema " + postgres_schema
        )
    revisions = conn.execute(f"SELECT version_num FROM {version_table}").fetchall()
    if {row["version_num"] for row in revisions} != set(get_heads()):
        raise IncompatibleSchemaError(
            "Unsupported PostgreSQL schema revision; recreate legacy databases, "
            "or run gluomem migrate upgrade --dsn ... --schema " + postgres_schema
        )
    for table in _ALLOWED_VEC_TABLES:
        row = conn.execute(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass "
            "AND attname = 'embedding'",
            (f"{postgres_schema}.{table}",),
        ).fetchone()
        if row is None or row["atttypmod"] != dim:
            raise IncompatibleSchemaError("Vector dimension differs; recreate the database")


class PostgresConnectionPool:
    """Process-wide PostgreSQL connection pool with startup-only schema verification."""

    def __init__(
        self,
        dsn: str,
        *,
        dim: int = 384,
        postgres_schema: str = "gluomem",
        min_size: int = 2,
        max_size: int = 16,
        open_pool: bool = True,
        verify_schema: bool = True,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"vector dim must be positive, got {dim}")
        if not dsn or not dsn.strip():
            raise ValueError("PostgresConnectionPool requires a non-empty dsn")
        if min_size < 1 or max_size < min_size:
            raise ValueError(f"Invalid pool bounds: min_size={min_size}, max_size={max_size}")
        validate_schema_name(postgres_schema)

        require("postgres", "PostgresConnectionPool")
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        self.dsn: str = dsn
        self.dim: int = dim
        self.postgres_schema: str = postgres_schema
        self.min_size: int = min_size
        self.max_size: int = max_size
        self._verify_on_open: bool = verify_schema
        self._verified: bool = False

        self._pool: Any = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            kwargs={"autocommit": True, "row_factory": dict_row},
            configure=lambda conn: _configure_connection(conn, postgres_schema),
            open=False,
        )
        if open_pool:
            self.open()

    def open(self, *, wait: bool = True) -> None:
        self._pool.open(wait=wait)
        if self._verify_on_open and not self._verified:
            try:
                with self._pool.connection() as conn:
                    _verify_schema_conn(conn, self.postgres_schema, self.dim)
                self._verified = True
            except Exception:
                self._pool.close()
                raise

    @contextmanager
    def connection(self) -> Iterator[Any]:
        with self._pool.connection() as conn:
            yield conn

    @contextmanager
    def borrow_backend(self) -> Iterator[PostgresBackend]:
        with self.connection() as conn:
            yield PostgresBackend(
                self.dsn,
                dim=self.dim,
                postgres_schema=self.postgres_schema,
                conn=conn,
                verify_schema=False,
                owns_connection=False,
            )

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> PostgresConnectionPool:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class PostgresBackend:
    """Schema-isolated PostgreSQL backend; each thread must own its connection."""

    dialect: Literal["sqlite", "postgres"] = "postgres"

    def __init__(
        self,
        dsn: str,
        *,
        dim: int = 384,
        postgres_schema: str = "gluomem",
        conn: Any | None = None,
        verify_schema: bool = True,
        owns_connection: bool | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"vector dim must be positive, got {dim}")
        if not dsn or not dsn.strip():
            raise ValueError("PostgresBackend requires a non-empty dsn")
        validate_schema_name(postgres_schema)

        require("postgres", "PostgresBackend")

        self.dsn: str = dsn
        self.dim: int = dim
        self.postgres_schema: str = postgres_schema
        self.path: Path = Path(f"postgres://{postgres_schema}")
        self._tx_depth: int = 0
        self._owns_connection: bool = (
            (conn is None) if owns_connection is None else bool(owns_connection)
        )

        if conn is None:
            import psycopg
            from psycopg.rows import dict_row

            self.conn: Any = psycopg.connect(dsn, row_factory=dict_row, autocommit=True)
            try:
                _configure_connection(self.conn, postgres_schema)
                if verify_schema:
                    self._verify_schema()
            except Exception:
                self.conn.close()
                raise
        else:
            self.conn = conn
            if verify_schema:
                self._verify_schema()

    def _verify_schema(self) -> None:
        _verify_schema_conn(self.conn, self.postgres_schema, self.dim)

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

    def executemany(self, sql: str, seq_of_params: Iterable[Sequence[Any]]) -> None:
        adapted = _adapt_qmark_to_pyformat(sql)
        with self.conn.cursor() as cur:
            cur.executemany(adapted, [tuple(p) for p in seq_of_params])

    def fetchone(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self.execute(sql, params).fetchone()
        return _PgRow(row) if row is not None else None

    def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[Any]:
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
        if self._owns_connection:
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

    def add_batch(self, items: list[tuple[int, np.ndarray]], *, commit: bool = True) -> None:
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
        eff_now = format_now_iso(now_iso)
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
        rows = self._backend.fetchall(sql, (f32, *eff_scope.as_tuple(), eff_now, f32, int(k)))
        return [(int(r["rowid"]), float(r["distance"])) for r in rows]

    def delete(self, rowid: int, *, commit: bool = True) -> None:
        self._backend.execute(f"DELETE FROM {self.table} WHERE rowid = ?", (int(rowid),))
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

    def upsert(self, rowid: int, content: str, *, commit: bool = False) -> None:
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
        plain_terms = sanitized_query.replace('"', " ").strip()
        if not plain_terms:
            return []
        folded_terms = fold_query_text(plain_terms, match_any=match_any)
        if not folded_terms:
            return []
        eff_scope = resolve_scope(scope)
        eff_now = format_now_iso(now_iso)
        # plainto_tsquery always AND-s its terms, so the any-term form uses only the folded query.
        if match_any:
            tsq_expr = "to_tsquery('simple', ?)"
            tsq_params: tuple[str, ...] = (folded_terms,)
        else:
            tsq_expr = "(to_tsquery('simple', ?) || plainto_tsquery('english', ?))"
            tsq_params = (folded_terms, plain_terms)
        rows = self._backend.fetchall(
            f"SELECT f.rowid AS rowid FROM memory_frames_fts f "
            f"JOIN memory_frames m ON m.id = f.rowid "
            f"WHERE f.tsv @@ {tsq_expr} "
            f"AND {scope_sql('m')} AND {eligible_sql('m')} "
            f"ORDER BY ts_rank_cd(f.tsv, {tsq_expr}) DESC, f.rowid ASC "
            f"LIMIT ?",
            (
                *tsq_params,
                *eff_scope.as_tuple(),
                eff_now,
                *tsq_params,
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


__all__ = [
    "PgTsVectorStore",
    "PgVectorStore",
    "PostgresBackend",
    "PostgresConnectionPool",
    "_adapt_qmark_to_pyformat",
    "_configure_connection",
]
