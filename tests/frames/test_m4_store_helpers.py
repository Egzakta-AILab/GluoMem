"""FrameStore helper tests — find_frame_by_hash + set_importance.

``find_frame_by_hash`` is the gate-level dedup probe (all frame types);
``set_importance`` powers the ``overwrite`` frame-model demotion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gluomem.frames.store import FrameStore
from gluomem.storage.db import MindDB


def _store(tmp_path: Path, *, dim: int = 384) -> tuple[MindDB, FrameStore]:
    db = MindDB(tmp_path / "test.mind", dim=dim)
    return db, FrameStore(db)


def test_find_frame_by_hash_hit_and_miss(tmp_path: Path) -> None:
    _, store = _store(tmp_path)
    frame = store.create_i_frame("g1", "Alice lives in Boston.")
    hit = store.find_frame_by_hash("g1", "Alice lives in Boston.")
    assert hit is not None and hit.id == frame.id
    # Whitespace-normalized content hashes equal (content_hash contract).
    hit2 = store.find_frame_by_hash("g1", "Alice  lives   in Boston.")
    assert hit2 is not None and hit2.id == frame.id
    assert store.find_frame_by_hash("g1", "Bob lives in Boston.") is None
    # Per-GOP scoping: same content in another GOP is a miss.
    assert store.find_frame_by_hash("g2", "Alice lives in Boston.") is None


def test_find_frame_by_hash_sees_p_and_b(tmp_path: Path) -> None:
    """The probe covers P/B rows too — that's what makes gate dedup universal."""
    _, store = _store(tmp_path)
    base = store.create_i_frame("g1", "base fact")
    p = store.create_p_frame("g1", "delta on base", base.id)
    b = store.create_b_frame("g1", "correction text", base.id, [base.id])
    hit_p = store.find_frame_by_hash("g1", "delta on base")
    assert hit_p is not None and hit_p.id == p.id
    # B-frames hash their *description*, not the JSON payload.
    hit_b = store.find_frame_by_hash("g1", "correction text")
    assert hit_b is not None and hit_b.id == b.id


def test_set_importance_roundtrip(tmp_path: Path) -> None:
    _, store = _store(tmp_path)
    frame = store.create_i_frame("g1", "will be demoted", importance="normal")
    store.set_importance(frame.id, "deprecated")
    assert store.get_frame(frame.id).importance == "deprecated"
    # Content untouched — pure importance flip.
    assert store.get_frame(frame.id).content == "will be demoted"


def test_set_importance_missing_frame_raises(tmp_path: Path) -> None:
    _, store = _store(tmp_path)
    with pytest.raises(KeyError):
        store.set_importance(999, "deprecated")
