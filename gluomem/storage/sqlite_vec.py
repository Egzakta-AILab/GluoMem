"""gluomem.storage.sqlite_vec — SqliteVecStore over sqlite-vec vec0.

An exact-KNN vector store over a ``vec0`` virtual table created by ``MindDB``
at the active embedder's native dim. No ANN index. Vectors are stored at native
dim with no zero-pad; a fingerprint guard raises on any dim mismatch before a
write.

The ``rowid`` is passed as a SQL literal per the sqlite-vec convention
(vec0 maps the integer primary key onto the rowid); the embedding is supplied
as a packed ``float32`` byte blob. Search uses
``embedding MATCH ? ORDER BY distance LIMIT k`` (exact KNN).
"""

from __future__ import annotations

import sqlite3

import numpy as np

from gluomem.embedding.fingerprint import VectorDimMismatchError, assert_dim

# Re-export the dim-mismatch error so callers can import it from the store
# module (the guard is the public contract surface for add/add_batch failures).
__all__ = ["SqliteVecStore", "VectorDimMismatchError"]


class SqliteVecStore:
    """Exact-KNN vector store over a sqlite-vec ``vec0`` table.

    Parameters
    ----------
    conn:
        An open ``sqlite3.Connection`` with the sqlite-vec extension loaded and
        the target ``vec0`` table already created (typically by ``MindDB``).
    table:
        The ``vec0`` virtual table name (e.g. ``memory_frames_vec``).
    dim:
        The native embedding dim the table was created at. Every ``add`` /
        ``add_batch`` vector MUST have this length — the fingerprint guard
        enforces it and raises ``VectorDimMismatchError`` on mismatch.
    """

    def __init__(self, conn: sqlite3.Connection, *, table: str, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"SqliteVecStore dim must be positive, got {dim}")
        self._conn: sqlite3.Connection = conn
        self._table: str = table
        self._dim: int = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    # --- writes -------------------------------------------------------------

    def add(self, rowid: int, vec: np.ndarray, *, commit: bool = True) -> None:
        """Insert one vector at ``rowid`` (native dim, no pad/truncate).

        Raises ``VectorDimMismatchError`` if ``len(vec) != dim`` and writes no
        row. The rowid is passed as a SQL literal per sqlite-vec; the embedding
        is a packed float32 byte blob. ``commit=False`` leaves the write inside
        the caller's open transaction (the FrameStore index hook groups FTS +
        vec + chunk writes into one atomic commit).
        """
        assert_dim(vec, self._dim)
        blob = self._as_blob(vec)
        self._conn.execute(
            f"INSERT INTO {self._table}(rowid, embedding) VALUES (?, ?)",
            (int(rowid), blob),
        )
        if commit:
            self._conn.commit()

    def add_batch(
        self, items: list[tuple[int, np.ndarray]], *, commit: bool = True
    ) -> None:
        """Insert many ``(rowid, vec)`` pairs in one transaction.

        All vectors are fingerprint-checked up front; on any mismatch the batch
        is aborted (no partial write for the bad rows). ``commit=False`` leaves
        the writes inside the caller's open transaction.
        """
        # Validate every vector first so a mismatch aborts before any write.
        rows: list[tuple[int, bytes]] = []
        for rowid, vec in items:
            assert_dim(vec, self._dim)
            rows.append((int(rowid), self._as_blob(vec)))
        self._conn.executemany(
            f"INSERT INTO {self._table}(rowid, embedding) VALUES (?, ?)",
            rows,
        )
        if commit:
            self._conn.commit()

    # --- reads --------------------------------------------------------------

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Exact-KNN search: return ``[(rowid, distance)]`` for the nearest ``k``.

        Uses ``embedding MATCH ? ORDER BY distance LIMIT k`` (exact KNN, no ANN).
        The query vector is NOT required to match the store dim for a MATCH
        (sqlite-vec validates at query time), but we still fingerprint it so a
        wrong-dim query fails loudly with the same clear error as a bad add.
        """
        assert_dim(query, self._dim)
        blob = self._as_blob(query)
        cur = self._conn.execute(
            f"SELECT rowid, distance FROM {self._table} "
            f"WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (blob, int(k)),
        )
        return [(int(r), float(d)) for r, d in cur.fetchall()]

    # --- deletes ------------------------------------------------------------

    def delete(self, rowid: int, *, commit: bool = True) -> None:
        """Delete the vector stored at ``rowid`` (no-op if absent).

        ``commit=False`` leaves the delete inside the caller's open transaction.
        """
        self._conn.execute(
            f"DELETE FROM {self._table} WHERE rowid=?", (int(rowid),)
        )
        if commit:
            self._conn.commit()

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _as_blob(vec: np.ndarray) -> bytes:
        """Pack a float32 ndarray as a contiguous byte blob for sqlite-vec."""
        arr = np.ascontiguousarray(vec, dtype=np.float32)
        return arr.tobytes()
