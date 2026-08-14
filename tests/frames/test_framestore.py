"""FrameStore I/P/B tests.

Drives the public ``FrameStore`` surface against a ``tmp_path`` ``*.mind`` file.
All writes funnel through the internal ``_index()`` hook (FTS + vec + chunk
sync). These tests assert frame *semantics* — type, base linkage, B correction
payload, importance inheritance, content-hash dedup idempotence, and
``reconstruct_state``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gluomem.frames.content_hash import content_hash
from gluomem.frames.store import FrameStore
from gluomem.storage.db import MindDB


def _store(tmp_path: Path, *, dim: int = 384) -> tuple[MindDB, FrameStore]:
    db = MindDB(tmp_path / "test.mind", dim=dim)
    return db, FrameStore(db)


# --- I-frame round-trip -----------------------------------------------------


def test_i_frame_roundtrip(tmp_path: Path) -> None:
    """An I-frame persists and reads back as type='I', base NULL."""
    db, store = _store(tmp_path)
    frame = store.create_i_frame("gop-1", "remember the milk")
    assert frame.frame_type == "I"
    assert frame.gop_id == "gop-1"
    assert frame.content == "remember the milk"
    assert frame.base_frame_id is None
    assert frame.importance == "normal"
    assert frame.source == "user_stated"

    # Read back via the accessor returns the same typed Frame.
    back = store.get_frame(frame.id)
    assert back.id == frame.id
    assert back.frame_type == "I"
    assert back.content == "remember the milk"
    assert back.gop_id == "gop-1"
    assert back.base_frame_id is None
    db.close()


def test_i_frame_event_time_split(tmp_path: Path) -> None:
    """An I-frame with an explicit event_time stores it distinct from created_at."""
    db, store = _store(tmp_path)
    frame = store.create_i_frame(
        "gop-1", "meeting happened", event_time="2020-01-01T00:00:00"
    )
    back = store.get_frame(frame.id)
    assert back.event_time == "2020-01-01T00:00:00"
    assert back.created_at != back.event_time
    db.close()


# --- P-frame base linkage ---------------------------------------------------


def test_p_frame_attaches_to_base(tmp_path: Path) -> None:
    """A P-frame reads back as type='P' with base_frame_id == i."""
    db, store = _store(tmp_path)
    i = store.create_i_frame("gop-1", "base fact")
    p = store.create_p_frame("gop-1", "refinement of base fact", base_id=i.id)
    assert p.frame_type == "P"
    assert p.base_frame_id == i.id
    assert p.content == "refinement of base fact"

    back = store.get_frame(p.id)
    assert back.frame_type == "P"
    assert back.base_frame_id == i.id
    db.close()


# --- B-frame payload --------------------------------------------------------


def test_b_frame_payload_and_link(tmp_path: Path) -> None:
    """A B-frame stores {description, references} and links to base."""
    db, store = _store(tmp_path)
    i = store.create_i_frame("gop-1", "the sky is green")
    refs = [i.id, 99]
    b = store.create_b_frame(
        "gop-1",
        description="the sky is actually blue",
        base_id=i.id,
        references=refs,
    )
    assert b.frame_type == "B"
    assert b.base_frame_id == i.id

    back = store.get_frame(b.id)
    assert back.frame_type == "B"
    assert back.base_frame_id == i.id

    # The B accessor exposes the description + references.
    correction = store.get_b_correction(b.id)
    assert correction.description == "the sky is actually blue"
    assert correction.references == refs
    db.close()


def test_b_correction_missing_keys_raises_value_error(tmp_path: Path) -> None:
    """A B-frame whose payload lacks description/references raises ValueError.

    A corrupted row must not surface as a bare KeyError.
    """
    db, store = _store(tmp_path)
    i = store.create_i_frame("gop-1", "fact")
    b = store.create_b_frame(
        "gop-1", description="correction", base_id=i.id, references=[i.id]
    )
    # Corrupt the payload in place: valid JSON, missing the required keys.
    db.conn.execute(
        "UPDATE memory_frames SET content='{\"oops\": 1}' WHERE id=?", (b.id,)
    )
    db.conn.commit()
    with pytest.raises(ValueError, match="not a valid correction payload"):
        store.get_b_correction(b.id)
    db.close()


# --- B-frame importance inheritance -----------------------------------------


def test_b_frame_inherits_critical_importance(tmp_path: Path) -> None:
    """A B-frame with importance=None inherits the base's 'critical'."""
    db, store = _store(tmp_path)
    i = store.create_i_frame(
        "gop-1", "critical fact", importance="critical"
    )
    b = store.create_b_frame(
        "gop-1",
        description="correction of critical fact",
        base_id=i.id,
        references=[i.id],
        importance=None,
    )
    assert b.importance == "critical", (
        "B-frame with importance=None must inherit the corrected base's "
        "importance ('critical'), not the 'normal' default"
    )
    db.close()


def test_b_frame_explicit_importance_overrides(tmp_path: Path) -> None:
    """An explicit B importance is used as-is (no inheritance)."""
    db, store = _store(tmp_path)
    i = store.create_i_frame("gop-1", "critical fact", importance="critical")
    b = store.create_b_frame(
        "gop-1",
        description="correction",
        base_id=i.id,
        references=[i.id],
        importance="temporary",
    )
    assert b.importance == "temporary"
    db.close()


# --- dedup idempotence ------------------------------------------------------


def test_dedup_is_idempotent(tmp_path: Path) -> None:
    """Strip/trim-equal content writes ONE frame, same id."""
    db, store = _store(tmp_path)
    f1 = store.create_i_frame("gop-1", "  remember the milk  ")
    f2 = store.create_i_frame("gop-1", "remember the milk")
    assert f1.id == f2.id, "dedup must return the existing frame, not a new one"
    count = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames WHERE content_hash=?",
        (content_hash("remember the milk"),),
    ).fetchone()[0]
    assert count == 1
    db.close()


def test_dedup_is_per_gop(tmp_path: Path) -> None:
    """Identical content in different GOPs is NOT deduped (GOP is the dedup scope)."""
    db, store = _store(tmp_path)
    a = store.create_i_frame("gop-a", "same content")
    b = store.create_i_frame("gop-b", "same content")
    assert a.id != b.id
    db.close()


def test_dedup_normalizes_internal_whitespace(tmp_path: Path) -> None:
    """Collapsed internal whitespace is dedup-equal to single-spaced content."""
    db, store = _store(tmp_path)
    f1 = store.create_i_frame("gop-1", "remember   the   milk")
    f2 = store.create_i_frame("gop-1", "remember the milk")
    assert f1.id == f2.id
    db.close()


# --- reconstruct_state ------------------------------------------------------


def test_reconstruct_state_latest_i_and_ps_since(tmp_path: Path) -> None:
    """I1, I2, P1-on-I2 -> (I2, [P1])."""
    db, store = _store(tmp_path)
    i1 = store.create_i_frame("gop-1", "first fact")
    i2 = store.create_i_frame("gop-1", "second fact")
    p1 = store.create_p_frame("gop-1", "refinement of second", base_id=i2.id)

    latest_i, ps = store.reconstruct_state("gop-1")
    assert latest_i is not None
    assert latest_i.id == i2.id
    assert i1.id < i2.id  # I1 written before I2
    assert [p.id for p in ps] == [p1.id]
    db.close()


def test_reconstruct_state_no_frames(tmp_path: Path) -> None:
    """reconstruct_state on an empty GOP returns (None, [])."""
    db, store = _store(tmp_path)
    latest_i, ps = store.reconstruct_state("gop-empty")
    assert latest_i is None
    assert ps == []
    db.close()


def test_reconstruct_state_only_i(tmp_path: Path) -> None:
    """reconstruct_state with only I-frames returns the latest I and no P-frames."""
    db, store = _store(tmp_path)
    i1 = store.create_i_frame("gop-1", "first")
    i2 = store.create_i_frame("gop-1", "second")
    latest_i, ps = store.reconstruct_state("gop-1")
    assert latest_i is not None
    assert latest_i.id == i2.id
    assert i1.id < i2.id
    assert ps == []
    db.close()


def test_reconstruct_state_p_before_latest_i_excluded(tmp_path: Path) -> None:
    """A P-frame written BEFORE the latest I is excluded from reconstruct_state."""
    db, store = _store(tmp_path)
    i1 = store.create_i_frame("gop-1", "first fact")
    p_old = store.create_p_frame("gop-1", "old refinement", base_id=i1.id)
    i2 = store.create_i_frame("gop-1", "second fact")
    latest_i, ps = store.reconstruct_state("gop-1")
    assert latest_i is not None
    assert latest_i.id == i2.id
    assert [p.id for p in ps] == []
    assert p_old.id < i2.id
    db.close()
