"""SqliteVecStore exact KNN, batching, deletion, and dimension guard tests."""

from __future__ import annotations

import numpy as np
import pytest

from gluomem.storage.db import MindDB
from gluomem.storage.sqlite_vec import SqliteVecStore, VectorDimMismatchError


def _unit_vec(dim: int, idx: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


def test_exact_knn_batch_and_delete(mind_db: MindDB) -> None:
    mind_db.execute("INSERT INTO sessions(gop_id) VALUES ('vectors')")
    mind_db.executemany(
        "INSERT INTO memory_frames(id, frame_type, gop_id, content) VALUES (?, 'I', 'vectors', ?)",
        [(i, f"frame {i}") for i in range(1, 4)],
    )
    store = SqliteVecStore(mind_db.conn, table="memory_frames_vec", dim=384)
    v1, v2, v3 = _unit_vec(384, 0), _unit_vec(384, 1), _unit_vec(384, 2)
    store.add(1, v1)
    store.add(2, v2)

    hits = store.search(v2, k=2)
    assert [r for r, _ in hits] == [2, 1]
    assert hits[0][1] == pytest.approx(0.0, abs=1e-6)

    length = mind_db.conn.execute(
        "SELECT vec_length(embedding) FROM memory_frames_vec WHERE rowid=1"
    ).fetchone()[0]
    assert length == 384

    store.add_batch([(3, v3)])
    assert store.search(v3, k=1)[0][0] == 3

    store.delete(2)
    assert all(r != 2 for r, _ in store.search(v2, k=3))


def test_dimension_guard_rejects_wrong_dim(mind_db: MindDB) -> None:
    store = SqliteVecStore(mind_db.conn, table="memory_frames_vec", dim=384)
    wrong = np.zeros(256, dtype=np.float32)
    with pytest.raises(VectorDimMismatchError):
        store.add(3, wrong)
    with pytest.raises(VectorDimMismatchError):
        store.add_batch([(1, _unit_vec(384, 0)), (2, wrong)])
    n = mind_db.conn.execute("SELECT COUNT(*) FROM memory_frames_vec").fetchone()[0]
    assert n == 0


def test_orphan_vectors_are_not_recalled(mind_db: MindDB) -> None:
    store = mind_db.frame_vec
    vec = _unit_vec(384, 0)
    store.add(99, vec)
    assert store.search(vec, k=1) == []
