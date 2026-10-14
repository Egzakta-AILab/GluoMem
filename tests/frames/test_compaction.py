"""Compaction tests.

``compact(gop, cfg)`` triggers when a GOP's base I accumulates MORE THAN
``write.compaction.p_frame_trigger`` (default 10) P-frames; keeps the
``write.compaction.keep_recent`` (default 5) most recent P-frames; merges the
older P-frame bodies INTO the base I-frame content joined by
``write.compaction.separator`` (default ``"\\n---\\n"``); RE-EMBEDS the merged
I-frame (updates its vec + FTS + chunks); and referentially DELETES the merged
P-frames and their FTS/vec/chunk rows.
"""

from __future__ import annotations

from pathlib import Path

from gluomem.config.app_config import AppConfig
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.storage.db import MindDB


def _store(tmp_path: Path, *, dim: int = 384) -> tuple[MindDB, FrameStore, AppConfig]:
    db = MindDB(tmp_path / "test.mind", dim=dim)
    emb = MockEmbedder(dim=dim)
    return db, FrameStore(db, embedder=emb), AppConfig()


def _index_counts(db: MindDB, frame_id: int) -> tuple[int, int, int]:
    """Return (fts_rows, vec_rows, chunk_rows) for ``frame_id``."""
    fts = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames_fts WHERE rowid=?", (frame_id,)
    ).fetchone()[0]
    vec = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames_vec WHERE rowid=?", (frame_id,)
    ).fetchone()[0]
    chunks = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frame_chunks WHERE frame_id=?", (frame_id,)
    ).fetchone()[0]
    return fts, vec, chunks


# --- compaction -------------------------------------------------------------


def test_compaction_merges_old_pframes_into_i(tmp_path: Path) -> None:
    """11 P-frames on one I, compact() -> 5 P remain, I merged + re-embedded."""
    db, store, cfg = _store(tmp_path)
    sep = cfg.write.compaction.separator
    trigger = cfg.write.compaction.p_frame_trigger
    keep = cfg.write.compaction.keep_recent
    assert trigger == 10 and keep == 5, "defaults must come from config"

    i = store.create_i_frame("gop-1", "base fact")
    p_ids: list[int] = []
    for k in range(11):
        p = store.create_p_frame("gop-1", f"delta body {k}", base_id=i.id)
        p_ids.append(p.id)
    # 11 > 10 trigger -> compaction should fire.
    older = p_ids[:6]  # the 6 oldest (merged into I)
    kept = p_ids[6:]  # the 5 most recent (retained)
    assert len(kept) == 5

    # Capture the I-frame vector BEFORE compaction.
    before_vec = db.conn.execute(
        "SELECT embedding FROM memory_frames_vec WHERE rowid=?", (i.id,)
    ).fetchone()[0]

    store.compact("gop-1", cfg)

    # Exactly 5 P-frames remain in the GOP.
    p_count = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames WHERE frame_type='P' AND gop_id=?",
        ("gop-1",),
    ).fetchone()[0]
    assert p_count == 5, f"expected 5 P-frames after compaction, got {p_count}"

    # The 5 retained P-frames are exactly the most-recent ones.
    remaining = [
        r[0]
        for r in db.conn.execute(
            "SELECT id FROM memory_frames WHERE frame_type='P' AND gop_id=? "
            "ORDER BY id ASC",
            ("gop-1",),
        ).fetchall()
    ]
    assert remaining == kept

    # The I-frame content contains the 6 older bodies joined by the separator.
    i_back = store.get_frame(i.id)
    older_bodies = [f"delta body {k}" for k in range(6)]
    assert sep.join(older_bodies) in i_back.content, (
        f"I content did not contain the 6 merged bodies joined by {sep!r}: "
        f"{i_back.content!r}"
    )
    # The original base content is still present (merged INTO, not replaced).
    assert "base fact" in i_back.content

    # The I-frame vector changed (re-embedded after the content merge).
    after_vec = db.conn.execute(
        "SELECT embedding FROM memory_frames_vec WHERE rowid=?", (i.id,)
    ).fetchone()[0]
    assert after_vec != before_vec, "I-frame vector was not re-embedded"

    # The I-frame still has exactly 1 vec row, 1 FTS row, >=1 chunk row.
    i_fts, i_vec, i_chunks = _index_counts(db, i.id)
    assert i_vec == 1
    assert i_fts == 1
    assert i_chunks >= 1

    # Each deleted (merged) P-frame id has ZERO fts / vec / chunk rows.
    for oid in older:
        fts_n, vec_n, chunk_n = _index_counts(db, oid)
        assert fts_n == 0, f"deleted P {oid} still has {fts_n} FTS rows"
        assert vec_n == 0, f"deleted P {oid} still has {vec_n} vec rows"
        assert chunk_n == 0, f"deleted P {oid} still has {chunk_n} chunk rows"
        # The frame row itself is gone.
        n = db.conn.execute(
            "SELECT COUNT(*) FROM memory_frames WHERE id=?", (oid,)
        ).fetchone()[0]
        assert n == 0

    # Each retained P-frame still has its index rows intact.
    for kid in kept:
        fts_n, vec_n, chunk_n = _index_counts(db, kid)
        assert fts_n == 1
        assert vec_n == 1
        assert chunk_n >= 1
    db.close()


def test_compaction_no_trigger_below_threshold(tmp_path: Path) -> None:
    """With <= trigger P-frames, compact() is a no-op (no merge, no deletes)."""
    db, store, cfg = _store(tmp_path)
    i = store.create_i_frame("gop-1", "base fact")
    p_ids = []
    for k in range(10):  # exactly 10, not MORE THAN 10
        p = store.create_p_frame("gop-1", f"delta body {k}", base_id=i.id)
        p_ids.append(p.id)

    store.compact("gop-1", cfg)

    p_count = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames WHERE frame_type='P' AND gop_id=?",
        ("gop-1",),
    ).fetchone()[0]
    assert p_count == 10, "10 P-frames (not > trigger) must all remain"
    i_back = store.get_frame(i.id)
    assert i_back.content == "base fact", "I content must be unchanged below trigger"
    db.close()


def test_compaction_respects_config_override(tmp_path: Path) -> None:
    """A lowered trigger / keep_recent override changes compaction behavior (config-purity)."""
    db, store, cfg = _store(tmp_path)
    cfg = cfg.with_overrides(
        {"write.compaction.p_frame_trigger": 3, "write.compaction.keep_recent": 1}
    )
    i = store.create_i_frame("gop-1", "base fact")
    p_ids = []
    for k in range(4):  # 4 > 3 trigger
        p = store.create_p_frame("gop-1", f"delta body {k}", base_id=i.id)
        p_ids.append(p.id)

    store.compact("gop-1", cfg)

    # keep_recent=1 -> 1 P remains; the 3 oldest are merged.
    p_count = db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames WHERE frame_type='P' AND gop_id=?",
        ("gop-1",),
    ).fetchone()[0]
    assert p_count == 1
    remaining = [
        r[0]
        for r in db.conn.execute(
            "SELECT id FROM memory_frames WHERE frame_type='P' AND gop_id=?",
            ("gop-1",),
        ).fetchall()
    ]
    assert remaining == [p_ids[-1]], "the single retained P must be the most recent"
    i_back = store.get_frame(i.id)
    sep = cfg.write.compaction.separator
    older_bodies = [f"delta body {k}" for k in range(3)]
    assert sep.join(older_bodies) in i_back.content
    db.close()


# --- trust / scope compatibility ----------------------------------------------


def _p_rows(db: MindDB, gop: str) -> list[tuple[int, str]]:
    return [
        (int(r[0]), str(r[1]))
        for r in db.conn.execute(
            "SELECT id, trust FROM memory_frames WHERE frame_type='P' AND gop_id=? "
            "ORDER BY id ASC",
            (gop,),
        ).fetchall()
    ]


def test_compaction_never_promotes_quarantined_pframes(tmp_path: Path) -> None:
    """Quarantined P-frames on a trusted base are not merged: the base keeps
    its content and trust; every quarantined P stays as its own row."""
    db, store, cfg = _store(tmp_path)
    i = store.create_i_frame("gop-1", "base fact", trust="trusted")
    for k in range(11):
        store.create_p_frame(
            "gop-1", f"injected delta {k}", base_id=i.id, trust="quarantined"
        )

    store.compact("gop-1", cfg)

    back = store.get_frame(i.id)
    assert back.content == "base fact"
    assert back.trust == "trusted"
    rows = _p_rows(db, "gop-1")
    assert len(rows) == 11 and {t for _, t in rows} == {"quarantined"}
    db.close()


def test_compaction_merges_only_compatible_pframes(tmp_path: Path) -> None:
    """Mixed deltas: eligible same-source ones merge, quarantined, lower-trust
    and other-source ones are left alone and do not count toward the trigger."""
    db, store, cfg = _store(tmp_path)
    i = store.create_i_frame("gop-1", "base fact", trust="normal")
    ok_ids = []
    for k in range(11):
        ok_ids.append(store.create_p_frame("gop-1", f"delta body {k}", base_id=i.id).id)
        if k % 4 == 0:
            store.create_p_frame(
                "gop-1", f"injected {k}", base_id=i.id, trust="quarantined"
            )
    store.create_p_frame("gop-1", "weak hint", base_id=i.id, trust="low")
    store.create_p_frame("gop-1", "inferred note", base_id=i.id, source="agent_inferred")
    store.create_p_frame("gop-1", "verified detail", base_id=i.id, trust="trusted")

    store.compact("gop-1", cfg)

    back = store.get_frame(i.id)
    assert back.trust == "normal"
    for k in range(6):
        assert f"delta body {k}" in back.content
    for absent in ("injected", "weak hint", "inferred note"):
        assert absent not in back.content
    remaining = {pid: t for pid, t in _p_rows(db, "gop-1")}
    assert sum(1 for t in remaining.values() if t == "quarantined") == 3
    assert "low" in remaining.values()
    db.close()


def test_compaction_without_enough_compatible_pframes_is_noop(tmp_path: Path) -> None:
    """11 P-frames but only 10 compatible: below trigger, nothing merges."""
    db, store, cfg = _store(tmp_path)
    i = store.create_i_frame("gop-1", "base fact")
    for k in range(10):
        store.create_p_frame("gop-1", f"delta body {k}", base_id=i.id)
    store.create_p_frame("gop-1", "injected", base_id=i.id, trust="quarantined")

    store.compact("gop-1", cfg)

    assert store.get_frame(i.id).content == "base fact"
    assert len(_p_rows(db, "gop-1")) == 11
    db.close()


def test_compaction_rejects_cross_gop_base(tmp_path: Path) -> None:
    """P-frames in one GOP pointing at a base in another GOP are not merged."""
    db, store, cfg = _store(tmp_path)
    i = store.create_i_frame("gop-a", "base fact")
    for k in range(11):
        store.create_p_frame("gop-b", f"delta body {k}", base_id=i.id)

    store.compact("gop-b", cfg)

    assert store.get_frame(i.id).content == "base fact"
    assert len(_p_rows(db, "gop-b")) == 11
    db.close()
