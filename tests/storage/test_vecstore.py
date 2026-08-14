"""SqliteVecStore tests.

``SqliteVecStore`` stores native-dim (384) vectors with no zero-pad and
exact-KNN returns the nearest rowid first. The fingerprint guard blocks a
wrong-dim vector (no truncation, no zero-pad) and writes no row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gluomem.storage.db import MindDB
from gluomem.storage.sqlite_vec import SqliteVecStore, VectorDimMismatchError


def _store(tmp_path: Path, *, dim: int = 384) -> tuple[MindDB, SqliteVecStore]:
    db = MindDB(tmp_path / "test.mind", dim=dim)
    return db, SqliteVecStore(db.conn, table="memory_frames_vec", dim=dim)


def _unit_vec(dim: int, idx: int) -> np.ndarray:
    """A deterministic unit vector with a 1.0 at position ``idx``."""
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


def test_exact_knn_returns_nearest_rowid(tmp_path: Path) -> None:
    """Add two distinct 384-dim vectors, search ~v2 k=1 → rowid 2."""
    db, store = _store(tmp_path)
    v1 = _unit_vec(384, 0)
    v2 = _unit_vec(384, 1)
    store.add(1, v1)
    store.add(2, v2)
    # query == v2 → nearest is rowid 2 (distance 0).
    hits = store.search(v2, k=1)
    assert len(hits) == 1
    rowid, distance = hits[0]
    assert rowid == 2
    assert distance == pytest.approx(0.0, abs=1e-6)
    db.close()


def test_exact_knn_not_ann(tmp_path: Path) -> None:
    """EXACT KNN: a query between two stored vectors ranks the closer one first."""
    db, store = _store(tmp_path)
    v1 = _unit_vec(384, 0)
    v2 = _unit_vec(384, 5)
    store.add(1, v1)
    store.add(2, v2)
    # query closer to v2 than v1.
    q = _unit_vec(384, 5)
    hits = store.search(q, k=2)
    assert [r for r, _ in hits] == [2, 1]
    db.close()


def test_persisted_embedding_length_is_native(tmp_path: Path) -> None:
    """Persisted embedding length is 384 (no zero-pad)."""
    db, store = _store(tmp_path)
    v = _unit_vec(384, 0)
    store.add(1, v)
    length = db.conn.execute(
        "SELECT vec_length(embedding) FROM memory_frames_vec WHERE rowid=1"
    ).fetchone()[0]
    assert length == 384
    db.close()


def test_add_batch_roundtrip(tmp_path: Path) -> None:
    """add_batch stores multiple vectors and they are searchable."""
    db, store = _store(tmp_path)
    items = [
        (1, _unit_vec(384, 0)),
        (2, _unit_vec(384, 1)),
        (3, _unit_vec(384, 2)),
    ]
    store.add_batch(items)
    hits = store.search(_unit_vec(384, 2), k=3)
    assert hits[0][0] == 3
    db.close()


def test_delete_removes_row(tmp_path: Path) -> None:
    """delete(rowid) removes the vector; it no longer appears in search."""
    db, store = _store(tmp_path)
    store.add(1, _unit_vec(384, 0))
    store.add(2, _unit_vec(384, 1))
    store.delete(2)
    n = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames_vec WHERE rowid=2"
    ).fetchone()[0]
    assert n == 0
    hits = store.search(_unit_vec(384, 1), k=2)
    assert all(r != 2 for r, _ in hits)
    db.close()


def test_fingerprint_guard_rejects_wrong_dim(tmp_path: Path) -> None:
    """add(3, zeros(256)) into a 384 store raises and writes no row."""
    db, store = _store(tmp_path)
    wrong = np.zeros(256, dtype=np.float32)
    with pytest.raises(VectorDimMismatchError):
        store.add(3, wrong)
    # No vec row was written for rowid 3.
    n = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames_vec WHERE rowid=3"
    ).fetchone()[0]
    assert n == 0
    db.close()


def test_fingerprint_guard_in_add_batch(tmp_path: Path) -> None:
    """The fingerprint guard also fires inside add_batch and writes nothing."""
    db, store = _store(tmp_path)
    good = (1, _unit_vec(384, 0))
    bad = (2, np.zeros(256, dtype=np.float32))
    with pytest.raises(VectorDimMismatchError):
        store.add_batch([good, bad])
    # On a batch failure nothing should be committed for the bad row.
    n_bad = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames_vec WHERE rowid=2"
    ).fetchone()[0]
    assert n_bad == 0
    db.close()
