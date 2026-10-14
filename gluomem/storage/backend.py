"""Storage protocols (`DatabaseBackend`, `VectorStore`, `FtsStore`) and vector guards."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from gluomem.storage.scope import MemoryScope


class VectorDimMismatchError(ValueError):
    """Raised when a vector's shape does not match the store's ``dim``."""


class IncompatibleSchemaError(RuntimeError):
    """Raised when a database does not have the supported schema version."""


def assert_dim(vec: np.ndarray, dim: int) -> np.ndarray:
    """Validate that ``vec`` is 1-D of length ``dim`` and return float32 view."""
    arr = np.asarray(vec)
    if arr.shape != (dim,):
        raise VectorDimMismatchError(
            f"Vector dim mismatch: expected shape ({dim},), got {arr.shape} "
            f"(no truncation or zero-padding allowed)"
        )
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


@runtime_checkable
class DatabaseBackend(Protocol):
    """Dialect-isolated relational database backend protocol."""

    dialect: Literal["sqlite", "postgres"]

    @property
    def in_transaction(self) -> bool: ...

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any: ...

    def insert_returning_id(
        self, sql: str, params: Sequence[Any] = (), *, id_col: str = "id"
    ) -> int: ...

    def executemany(self, sql: str, seq_of_params: Iterable[Sequence[Any]]) -> None: ...

    def fetchone(self, sql: str, params: Sequence[Any] = ()) -> Mapping[str, Any] | None: ...

    def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[Mapping[str, Any]]: ...

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any: ...

    def transaction(self) -> AbstractContextManager[None]: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class VectorStore(Protocol):
    """Exact or approximate nearest-neighbor vector store protocol."""

    dim: int

    def add(self, rowid: int, vec: np.ndarray, *, commit: bool = True) -> None: ...

    def add_batch(self, items: list[tuple[int, np.ndarray]], *, commit: bool = True) -> None: ...

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        scope: MemoryScope | None = None,
        now_iso: str | None = None,
    ) -> list[tuple[int, float]]: ...

    def delete(self, rowid: int, *, commit: bool = True) -> None: ...

    def clear(self, *, commit: bool = True) -> None: ...

    def count(self) -> int: ...

    def has(self, rowid: int) -> bool: ...

    def count_by_ids(self, rowids: Sequence[int]) -> int: ...

    def ids(self) -> set[int]: ...


@runtime_checkable
class FtsStore(Protocol):
    """Full-text search index protocol (`FTS5` on SQLite, `tsvector` on Postgres)."""

    def upsert(self, rowid: int, content: str, *, commit: bool = False) -> None: ...

    def delete(self, rowid: int, *, commit: bool = False) -> None: ...

    def search(
        self,
        sanitized_query: str,
        limit: int,
        *,
        scope: MemoryScope | None = None,
        now_iso: str | None = None,
    ) -> list[int]: ...

    def search_like(
        self,
        query: str | list[str],
        limit: int,
        *,
        scope: MemoryScope | None = None,
        now_iso: str | None = None,
    ) -> list[int]: ...

    def get_text(self, rowid: int) -> str | None: ...

    def clear(self, *, commit: bool = False) -> None: ...


__all__ = [
    "DatabaseBackend",
    "FtsStore",
    "VectorDimMismatchError",
    "VectorStore",
    "assert_dim",
]
