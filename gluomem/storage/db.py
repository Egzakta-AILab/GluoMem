"""Database facade (`MindDB`) wiring relational backend, vector stores, and FTS store."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from gluomem.storage.backend import DatabaseBackend, FtsStore, VectorStore
from gluomem.storage.sqlite_backend import SqliteBackend, SqliteFts5Store
from gluomem.storage.sqlite_vec import SqliteVecStore

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig, StorageConfig


class MindDB:
    """Unified database handle for SQLite (`.mind`) or PostgreSQL (`pgvector`) storage."""

    def __init__(
        self,
        path: Path | str = ":memory:",
        dim: int | None = None,
        *,
        vector_dim: int | None = None,
        backend_type: Literal["sqlite", "postgres"] = "sqlite",
        dsn: str | None = None,
        postgres_schema: str = "gluomem",
        backend: DatabaseBackend | None = None,
        frame_vec: VectorStore | None = None,
        chunk_vec: VectorStore | None = None,
        fts: FtsStore | None = None,
    ) -> None:
        resolved_dim = (
            dim if dim is not None else (vector_dim if vector_dim is not None else 384)
        )
        if resolved_dim <= 0:
            raise ValueError(f"vector dim must be positive, got {resolved_dim}")

        self.dim: int = resolved_dim

        if backend is not None:
            self.backend: DatabaseBackend = backend
            self.path: Path = getattr(backend, "path", Path(str(path)))
        elif backend_type == "postgres":
            from gluomem.storage.postgres_backend import (
                PgTsVectorStore,
                PgVectorStore,
                PostgresBackend,
            )

            if not dsn:
                raise ValueError("MindDB(backend_type='postgres') requires dsn")
            pg_backend = PostgresBackend(
                dsn, dim=resolved_dim, postgres_schema=postgres_schema
            )
            self.backend = pg_backend
            self.path = pg_backend.path
            frame_vec = frame_vec or PgVectorStore(
                pg_backend, table="memory_frames_vec", dim=resolved_dim
            )
            chunk_vec = chunk_vec or PgVectorStore(
                pg_backend, table="memory_frame_chunks_vec", dim=resolved_dim
            )
            fts = fts or PgTsVectorStore(pg_backend)
        else:
            sqlite_backend = SqliteBackend(path, dim=resolved_dim)
            self.backend = sqlite_backend
            self.path = sqlite_backend.path
            frame_vec = frame_vec or SqliteVecStore(
                sqlite_backend.conn, table="memory_frames_vec", dim=resolved_dim
            )
            chunk_vec = chunk_vec or SqliteVecStore(
                sqlite_backend.conn, table="memory_frame_chunks_vec", dim=resolved_dim
            )
            fts = fts or SqliteFts5Store(sqlite_backend)

        if frame_vec is None or chunk_vec is None or fts is None:
            raise ValueError(
                "Custom backend passed to MindDB requires frame_vec, chunk_vec, and fts stores"
            )
        self.frame_vec: VectorStore = frame_vec
        self.chunk_vec: VectorStore = chunk_vec
        self.fts: FtsStore = fts

    @classmethod
    def from_config(
        cls,
        cfg: AppConfig | StorageConfig,
        *,
        path_override: Path | str | None = None,
    ) -> MindDB:
        """Open a ``MindDB`` from ``AppConfig`` or ``StorageConfig``."""
        from gluomem.config.app_config import AppConfig

        storage_cfg: StorageConfig = cfg.storage if isinstance(cfg, AppConfig) else cfg
        if storage_cfg.backend == "postgres" and path_override is None:
            return cls(
                dim=storage_cfg.vector_dim,
                backend_type="postgres",
                dsn=storage_cfg.dsn,
                postgres_schema=storage_cfg.postgres_schema,
            )
        target_path = path_override if path_override is not None else storage_cfg.path
        return cls(target_path, dim=storage_cfg.vector_dim)

    @property
    def dialect(self) -> Literal["sqlite", "postgres"]:
        return self.backend.dialect

    @property
    def conn(self) -> Any:
        """Underlying connection handle for low-level inspection."""
        return getattr(self.backend, "conn", None)

    @property
    def in_transaction(self) -> bool:
        return self.backend.in_transaction

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        return self.backend.execute(sql, params)

    def executemany(
        self, sql: str, seq_of_params: Iterable[Sequence[Any]]
    ) -> None:
        self.backend.executemany(sql, seq_of_params)

    def fetchone(
        self, sql: str, params: Sequence[Any] = ()
    ) -> Mapping[str, Any] | None:
        return self.backend.fetchone(sql, params)

    def fetchall(
        self, sql: str, params: Sequence[Any] = ()
    ) -> list[Mapping[str, Any]]:
        return self.backend.fetchall(sql, params)

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        return self.backend.scalar(sql, params)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self.backend.transaction():
            yield

    def commit(self) -> None:
        self.backend.commit()

    def rollback(self) -> None:
        self.backend.rollback()

    def meta_get(self, key: str) -> str | None:
        """Return ``meta.value`` for ``key``, or ``None`` if absent."""
        val = self.scalar("SELECT value FROM meta WHERE key = ?", (key,))
        return str(val) if val is not None else None

    def meta_set(self, key: str, value: str, *, commit: bool = True) -> None:
        """Upsert ``(key, value)`` into ``meta``."""
        self.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value",
            (key, value),
        )
        if commit:
            self.commit()

    def has_any_vectors(self) -> bool:
        """Return True iff either vector store holds at least one vector."""
        return self.frame_vec.count() > 0 or self.chunk_vec.count() > 0

    def close(self) -> None:
        self.backend.close()

    def __enter__(self) -> MindDB:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
