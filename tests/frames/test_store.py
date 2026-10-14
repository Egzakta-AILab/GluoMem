"""FrameStore I/P/B semantics, helpers, index sync, and failure-atomicity tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gluomem.config.app_config import load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.content_hash import content_hash
from gluomem.frames.sessions import get_or_create_session
from gluomem.frames.store import FrameStore, FrameType
from gluomem.storage.db import MindDB
from gluomem.write.gate import InjectionScanner, admit


class FlakyEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self._inner = MockEmbedder(dim=dim)
        self.fail_embed = False
        self.fail_batch = False

    @property
    def dim(self) -> int:
        return self._inner.dim

    def embed(self, text: str) -> np.ndarray:
        if self.fail_embed:
            raise RuntimeError("embedder down (embed)")
        return self._inner.embed(text)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if self.fail_batch:
            raise RuntimeError("embedder down (embed_batch)")
        return self._inner.embed_batch(texts)


def _fts_ids(db: MindDB, term: str) -> list[int]:
    return [
        int(r[0])
        for r in db.conn.execute(
            "SELECT rowid FROM memory_frames_fts WHERE memory_frames_fts MATCH ?",
            (term,),
        ).fetchall()
    ]


def test_ipb_roundtrip_and_importance_inheritance(
    mind_db: MindDB, frame_store: FrameStore
) -> None:
    i = frame_store.create_i_frame(
        "gop-1",
        "remember the milk",
        importance="critical",
        event_time="2020-01-01T00:00:00",
    )
    assert i.frame_type == "I" and i.base_frame_id is None
    back_i = frame_store.get_frame(i.id)
    assert back_i.event_time == "2020-01-01T00:00:00"
    assert back_i.created_at != back_i.event_time

    p = frame_store.create_p_frame("gop-1", "refinement of milk", base_id=i.id)
    assert p.frame_type == "P" and p.base_frame_id == i.id

    b = frame_store.create_b_frame(
        "gop-1",
        description="actually oat milk",
        base_id=i.id,
        references=[i.id, p.id],
        importance=None,
    )
    assert b.frame_type == "B" and b.base_frame_id == i.id
    assert b.importance == "critical"
    corr = frame_store.get_b_correction(b.id)
    assert corr.description == "actually oat milk" and corr.references == [i.id, p.id]

    b_exp = frame_store.create_b_frame(
        "gop-1",
        description="temp correction",
        base_id=i.id,
        references=[i.id],
        importance="temporary",
    )
    assert b_exp.importance == "temporary"

    mind_db.conn.execute(
        'UPDATE memory_frames SET content=\'{"oops": 1}\' WHERE id=?', (b_exp.id,)
    )
    mind_db.conn.commit()
    with pytest.raises(ValueError, match="not a valid correction payload"):
        frame_store.get_b_correction(b_exp.id)


def test_dedup_hash_lookup_and_set_importance(
    mind_db: MindDB, frame_store: FrameStore
) -> None:
    f1 = frame_store.create_i_frame("gop-1", "  remember   the   milk  ")
    f2 = frame_store.create_i_frame("gop-1", "remember the milk")
    assert f1.id == f2.id
    f_other = frame_store.create_i_frame("gop-2", "remember the milk")
    assert f_other.id != f1.id

    p = frame_store.create_p_frame("gop-1", "delta on base", f1.id)
    b = frame_store.create_b_frame("gop-1", "correction text", f1.id, [f1.id])
    assert frame_store.find_frame_by_hash("gop-1", "remember the milk").id == f1.id  # type: ignore[union-attr]
    assert frame_store.find_frame_by_hash("gop-1", "delta on base").id == p.id  # type: ignore[union-attr]
    assert frame_store.find_frame_by_hash("gop-1", "correction text").id == b.id  # type: ignore[union-attr]
    assert frame_store.find_frame_by_hash("gop-1", "missing") is None

    frame_store.set_importance(f1.id, "deprecated")
    assert frame_store.get_frame(f1.id).importance == "deprecated"
    with pytest.raises(KeyError):
        frame_store.set_importance(999, "deprecated")


def test_reconstruct_state_and_eligibility(frame_store: FrameStore) -> None:
    assert frame_store.reconstruct_state("gop-empty") == (None, [])

    i1 = frame_store.create_i_frame("gop-1", "first fact")
    frame_store.create_p_frame("gop-1", "old refinement", base_id=i1.id)
    i2 = frame_store.create_i_frame("gop-1", "second fact")
    p_ok = frame_store.create_p_frame("gop-1", "delta ok", base_id=i2.id)
    frame_store.create_p_frame("gop-1", "delta bad", base_id=i2.id, trust="quarantined")
    i_bad = frame_store.create_i_frame("gop-1", "flagged fact", trust="quarantined")

    latest_i, ps = frame_store.reconstruct_state("gop-1")
    assert latest_i is not None and latest_i.id == i2.id
    assert [p.id for p in ps] == [p_ok.id]

    latest_all, ps_all = frame_store.reconstruct_state("gop-1", include_ineligible=True)
    assert latest_all is not None and latest_all.id == i_bad.id
    assert ps_all == []


def test_index_sync_for_i_p_b_and_dedup(
    mind_db: MindDB, frame_store: FrameStore
) -> None:
    i = frame_store.create_i_frame("gop-1", "alpha beta gamma")
    p = frame_store.create_p_frame("gop-1", "refinement delta", base_id=i.id)
    b = frame_store.create_b_frame(
        "gop-1", description="the sky is blue", base_id=i.id, references=[i.id]
    )

    assert i.id in _fts_ids(mind_db, "beta")
    assert p.id in _fts_ids(mind_db, "refinement")
    assert b.id in _fts_ids(mind_db, "blue")
    assert b.id not in _fts_ids(mind_db, "references")

    for fid in (i.id, p.id, b.id):
        assert frame_store.index_state(fid).ready


@pytest.mark.parametrize("failure", ["fail_embed", "fail_batch"])
@pytest.mark.parametrize("kind", ["I", "P", "B"])
def test_embed_failure_leaves_nothing_and_retry_succeeds(
    tmp_path: Path, failure: str, kind: str
) -> None:
    db = MindDB(tmp_path / "test.mind", dim=384)
    emb = FlakyEmbedder(384)
    store = FrameStore(db, embedder=emb)
    base = store.create_i_frame("gop-1", "base alpha")

    def write():
        if kind == "I":
            return store.create_i_frame("gop-1", "delta epsilon zeta")
        if kind == "P":
            return store.create_p_frame("gop-1", "delta epsilon zeta", base.id)
        return store.create_b_frame("gop-1", "delta epsilon zeta", base.id, [base.id])

    setattr(emb, failure, True)
    with pytest.raises(RuntimeError, match="embedder down"):
        write()
    assert db.conn.execute("SELECT COUNT(*) FROM memory_frames").fetchone()[0] == 1
    assert _fts_ids(db, "epsilon") == []

    setattr(emb, failure, False)
    frame = write()
    assert store.index_state(frame.id).ready
    db.close()


def test_sql_failure_rollback_and_repair_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = MindDB(tmp_path / "test.mind", dim=384)
    store = FrameStore(db, embedder=FlakyEmbedder(384))

    def boom(*_a, **_k):
        raise RuntimeError("chunk write failed")

    monkeypatch.setattr(store, "_index_chunks", boom)
    with pytest.raises(RuntimeError, match="chunk write failed"):
        store.create_i_frame("gop-1", "delta epsilon zeta")
    monkeypatch.undo()
    assert db.conn.execute("SELECT COUNT(*) FROM memory_frames").fetchone()[0] == 0

    def _legacy_unindexed(
        content: str, frame_type: FrameType = "I", base_id: int | None = None
    ) -> int:
        get_or_create_session(store._conn, "gop-1")
        fid = store._insert(
            gop="gop-1",
            frame_type=frame_type,
            base_frame_id=base_id,
            content=content,
            importance="normal",
            source="user_stated",
            trust="normal",
            event_time=None,
            content_hash=content_hash(content),
        )
        store._conn.commit()
        return fid

    fid = _legacy_unindexed("delta epsilon zeta")
    assert not store.index_state(fid).ready
    again = store.create_i_frame("gop-1", "delta epsilon zeta")
    assert again.id == fid and store.index_state(fid).ready

    cfg = load_config()
    pid = _legacy_unindexed("pframe unindexed", frame_type="P", base_id=fid)
    res = admit(
        "pframe unindexed",
        gop="gop-1",
        scanner=InjectionScanner(cfg.safety),
        store=store,
        kg=None,
        cfg=cfg,
        frame_type="P",
        base_id=fid,
    )
    assert res.deduped and res.frame is not None and res.frame.id == pid
    assert store.index_state(pid).ready

    missing = _legacy_unindexed("beta missing")
    stale = store.create_i_frame("gop-1", "gamma stale")
    db.conn.execute(
        "UPDATE memory_frames SET content='gamma rewritten' WHERE id=?", (stale.id,)
    )
    db.conn.commit()
    assert store.repair_indexes() == [missing, stale.id]
    assert store.repair_indexes() == []
    db.close()
