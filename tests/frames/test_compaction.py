"""Compaction trigger, trust compatibility, failure atomicity, and lineage tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from gluomem.config.app_config import AppConfig
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.content_hash import content_hash
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.storage.db import MindDB


class FlakyEmbedder(MockEmbedder):
    fail_embed = False
    fail_batch = False

    def embed(self, text: str):
        if self.fail_embed:
            raise RuntimeError("embedder down (embed)")
        return super().embed(text)

    def embed_batch(self, texts: list[str]):
        if self.fail_batch:
            raise RuntimeError("embedder down (embed_batch)")
        return super().embed_batch(texts)


def _index_counts(db: MindDB, frame_id: int) -> tuple[int, int, int]:
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


def _p_rows(db: MindDB, gop: str) -> list[tuple[int, str]]:
    return [
        (int(r[0]), str(r[1]))
        for r in db.conn.execute(
            "SELECT id, trust FROM memory_frames WHERE frame_type='P' AND gop_id=? ORDER BY id ASC",
            (gop,),
        ).fetchall()
    ]


def test_compaction_trigger_merge_and_override(
    mind_db: MindDB, frame_store: FrameStore, cfg: AppConfig
) -> None:
    i = frame_store.create_i_frame("gop-1", "base fact")
    p_ids = [
        frame_store.create_p_frame("gop-1", f"delta body {k}", base_id=i.id).id for k in range(10)
    ]
    frame_store.compact("gop-1", cfg)
    assert len(_p_rows(mind_db, "gop-1")) == 10
    assert frame_store.get_frame(i.id).content == "base fact"

    p_ids.append(frame_store.create_p_frame("gop-1", "delta body 10", base_id=i.id).id)
    before_vec = mind_db.conn.execute(
        "SELECT embedding FROM memory_frames_vec WHERE rowid=?", (i.id,)
    ).fetchone()[0]

    frame_store.compact("gop-1", cfg)
    assert [pid for pid, _ in _p_rows(mind_db, "gop-1")] == p_ids[6:]
    i_back = frame_store.get_frame(i.id)
    expected_merged = cfg.write.compaction.separator.join(f"delta body {k}" for k in range(6))
    assert expected_merged in i_back.content
    after_vec = mind_db.conn.execute(
        "SELECT embedding FROM memory_frames_vec WHERE rowid=?", (i.id,)
    ).fetchone()[0]
    assert after_vec != before_vec
    for oid in p_ids[:6]:
        assert _index_counts(mind_db, oid) == (0, 0, 0)

    low_cfg = cfg.with_overrides(
        {"write.compaction.p_frame_trigger": 3, "write.compaction.keep_recent": 1}
    )
    frame_store.compact("gop-1", low_cfg)
    assert [pid for pid, _ in _p_rows(mind_db, "gop-1")] == [p_ids[-1]]


def test_compaction_trust_and_scope_compatibility(
    mind_db: MindDB, frame_store: FrameStore, cfg: AppConfig
) -> None:
    i = frame_store.create_i_frame("gop-1", "base fact", trust="normal")
    for k in range(11):
        frame_store.create_p_frame("gop-1", f"delta body {k}", base_id=i.id)
        if k % 4 == 0:
            frame_store.create_p_frame("gop-1", f"injected {k}", base_id=i.id, trust="quarantined")
    frame_store.create_p_frame("gop-1", "weak hint", base_id=i.id, trust="low")
    frame_store.create_p_frame("gop-1", "inferred note", base_id=i.id, source="agent_inferred")
    frame_store.create_p_frame("gop-1", "verified detail", base_id=i.id, trust="trusted")

    frame_store.compact("gop-1", cfg)
    back = frame_store.get_frame(i.id)
    assert back.trust == "normal"
    for k in range(6):
        assert f"delta body {k}" in back.content
    for absent in ("injected", "weak hint", "inferred note"):
        assert absent not in back.content

    i_cross = frame_store.create_i_frame("gop-a", "base a")
    for k in range(11):
        frame_store.create_p_frame("gop-b", f"cross {k}", base_id=i_cross.id)
    frame_store.compact("gop-b", cfg)
    assert frame_store.get_frame(i_cross.id).content == "base a"
    assert len(_p_rows(mind_db, "gop-b")) == 11


@pytest.mark.parametrize("failure", ["fail_embed", "fail_batch"])
def test_compaction_embed_failure_changes_nothing(tmp_path: Path, failure: str) -> None:
    db = MindDB(tmp_path / "test.mind", dim=384)
    emb = FlakyEmbedder(dim=384)
    store = FrameStore(db, embedder=emb)
    cfg = AppConfig()
    i = store.create_i_frame("gop-1", "base fact")
    for k in range(11):
        store.create_p_frame("gop-1", f"delta body {k}", base_id=i.id)

    setattr(emb, failure, True)
    with pytest.raises(RuntimeError, match="embedder down"):
        store.compact("gop-1", cfg)
    assert store.get_frame(i.id).content == "base fact"
    assert len(_p_rows(db, "gop-1")) == 11
    db.close()


@pytest.mark.parametrize("hook", ["_record_lineage", "_delete_index_rows"])
def test_compaction_sql_failure_rollback_and_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hook: str
) -> None:
    db = MindDB(tmp_path / "test.mind", dim=384)
    store = FrameStore(db, embedder=FlakyEmbedder(dim=384))
    kg = KnowledgeGraph(db)
    cfg = AppConfig()
    i = store.create_i_frame("gop-1", "base fact")
    p_ids = [store.create_p_frame("gop-1", f"delta body {k}", base_id=i.id).id for k in range(11)]
    ent = kg.create_entity("concept", "delta")
    kg.link_entity_frame(ent.id, p_ids[0])

    real = getattr(store, hook)
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("publication failed")
        return real(*a, **k)

    monkeypatch.setattr(store, hook, flaky)
    with pytest.raises(RuntimeError, match="publication failed"):
        store.compact("gop-1", cfg)
    monkeypatch.undo()
    assert store.get_frame(i.id).content == "base fact"
    assert len(_p_rows(db, "gop-1")) == 11

    store.compact("gop-1", cfg)
    entries = store.lineage(i.id)
    assert [(e.source_frame_id, e.operation) for e in entries] == [(i.id, "compaction_base")] + [
        (pid, "compaction_merge") for pid in p_ids[:6]
    ]
    assert entries[0].source_content_hash == content_hash("base fact")
    assert i.id in kg.frame_ids_for_entity(ent.id)
    db.close()
