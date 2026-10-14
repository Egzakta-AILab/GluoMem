"""PostgreSQL + ``pgvector`` implementations of storage protocols."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np

from gluomem.extras import require
from gluomem.storage.backend import assert_dim

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

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        f32 = assert_dim(query, self.dim)
        rows = self._backend.fetchall(
            f"SELECT rowid, (embedding <-> ?) AS distance FROM {self.table} "
            f"ORDER BY embedding <-> ? ASC, rowid ASC LIMIT ?",
            (f32, f32, int(k)),
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
        self._backend.execute(
            "INSERT INTO memory_frames_fts(rowid, content) VALUES (?, ?) "
            "ON CONFLICT (rowid) DO UPDATE SET content = EXCLUDED.content",
            (int(rowid), content),
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
        plain_terms = sanitized_query.replace('"', " ").strip()
        if not plain_terms:
            return []
        rows = self._backend.fetchall(
            "SELECT f.rowid AS rowid FROM memory_frames_fts f "
            "JOIN memory_frames m ON m.id = f.rowid "
            "WHERE f.tsv @@ websearch_to_tsquery('english', ?) "
            "AND m.trust != 'quarantined' "
            "ORDER BY ts_rank_cd(f.tsv, websearch_to_tsquery('english', ?)) DESC, f.rowid ASC "
            "LIMIT ?",
            (plain_terms, plain_terms, int(limit)),
        )
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


__all__ = [
    "PgTsVectorStore",
    "PgVectorStore",
    "PostgresBackend",
    "_adapt_qmark_to_pyformat",
]
