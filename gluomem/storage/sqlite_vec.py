"""SQLite ``vec0`` exact-KNN ``VectorStore`` implementation."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import numpy as np

from gluomem.storage.backend import (
    VectorDimMismatchError,
    VectorStore,
    assert_dim,
)

_ALLOWED_TABLES: frozenset[str] = frozenset(
    {"memory_frames_vec", "memory_frame_chunks_vec"}
)


class SqliteVecStore:
    """Exact-KNN vector store backed by a ``sqlite-vec`` ``vec0`` virtual table."""

    def __init__(self, conn: sqlite3.Connection, table: str, dim: int) -> None:
        if table not in _ALLOWED_TABLES:
            raise ValueError(
                f"Invalid vec0 table {table!r}; must be one of {sorted(_ALLOWED_TABLES)}"
            )
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._conn: sqlite3.Connection = conn
        self.table: str = table
        self.dim: int = dim

    def add(self, rowid: int, vec: np.ndarray, *, commit: bool = True) -> None:
        rid = int(rowid)
        f32 = assert_dim(vec, self.dim)
        blob = f32.tobytes()
        if commit:
            with self._conn:
                self._conn.execute(f"DELETE FROM {self.table} WHERE rowid = {rid}")
                self._conn.execute(
                    f"INSERT INTO {self.table}(rowid, embedding) VALUES ({rid}, ?)",
                    (blob,),
                )
        else:
            self._conn.execute(f"DELETE FROM {self.table} WHERE rowid = {rid}")
            self._conn.execute(
                f"INSERT INTO {self.table}(rowid, embedding) VALUES ({rid}, ?)",
                (blob,),
            )

    def add_batch(
        self, items: list[tuple[int, np.ndarray]], *, commit: bool = True
    ) -> None:
        if not items:
            return
        prepared: list[tuple[int, bytes]] = []
        for rowid, vec in items:
            rid = int(rowid)
            f32 = assert_dim(vec, self.dim)
            prepared.append((rid, f32.tobytes()))
        if commit:
            with self._conn:
                for rid, blob in prepared:
                    self._conn.execute(f"DELETE FROM {self.table} WHERE rowid = {rid}")
                    self._conn.execute(
                        f"INSERT INTO {self.table}(rowid, embedding) VALUES ({rid}, ?)",
                        (blob,),
                    )
        else:
            for rid, blob in prepared:
                self._conn.execute(f"DELETE FROM {self.table} WHERE rowid = {rid}")
                self._conn.execute(
                    f"INSERT INTO {self.table}(rowid, embedding) VALUES ({rid}, ?)",
                    (blob,),
                )

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        f32 = assert_dim(query, self.dim)
        blob = f32.tobytes()
        rows = self._conn.execute(
            f"SELECT rowid, distance FROM {self.table} "
            f"WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (blob, int(k)),
        ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    def delete(self, rowid: int, *, commit: bool = True) -> None:
        rid = int(rowid)
        self._conn.execute(f"DELETE FROM {self.table} WHERE rowid = {rid}")
        if commit:
            self._conn.commit()

    def clear(self, *, commit: bool = True) -> None:
        self._conn.execute(f"DELETE FROM {self.table}")
        if commit:
            self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()
        return int(row[0]) if row is not None else 0

    def has(self, rowid: int) -> bool:
        row = self._conn.execute(
            f"SELECT 1 FROM {self.table} WHERE rowid = {int(rowid)} LIMIT 1"
        ).fetchone()
        return row is not None

    def count_by_ids(self, rowids: Sequence[int]) -> int:
        if not rowids:
            return 0
        literals = ",".join(str(int(rid)) for rid in rowids)
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM {self.table} WHERE rowid IN ({literals})"
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def ids(self) -> set[int]:
        rows = self._conn.execute(f"SELECT rowid FROM {self.table}").fetchall()
        return {int(r[0]) for r in rows}


__all__ = ["SqliteVecStore", "VectorDimMismatchError", "VectorStore", "assert_dim"]
