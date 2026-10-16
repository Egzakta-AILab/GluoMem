"""Standalone callers must not project an erased snapshot (no outer transaction)."""

import pytest

from gluomem.frames.generation import current_scope_generation
from gluomem.frames.lifecycle import retract_source
from gluomem.write.events import SourceEvent, ingest_source_event
from gluomem.write.gate import admit, make_scanner


@pytest.mark.parametrize("flags", [{}, {"verified_by_user": True}, {"pinned": True}])
def test_standalone_admit_revalidates_before_repair(frame_store, mind_db, cfg, monkeypatch, flags):
    event = SourceEvent(
        event_id="standalone", source_ref="doc:standalone", content="Private original"
    )
    frame = ingest_source_event(event, store=frame_store, cfg=cfg).frames[0]
    original = frame_store.find_frame_by_hash

    def retract_after_lookup(*args, **kwargs):
        found = original(*args, **kwargs)
        retract_source(frame_store, event.source_ref)
        return found

    monkeypatch.setattr(frame_store, "find_frame_by_hash", retract_after_lookup)
    with pytest.raises(ValueError, match="changed before admission"):
        admit(
            frame.content,
            gop=frame.gop_id,
            scanner=make_scanner(cfg),
            store=frame_store,
            kg=None,
            cfg=cfg,
            **flags,
        )
    assert frame_store.get_frame(frame.id).content == "[forgotten]"
    assert mind_db.scalar("SELECT COUNT(*) FROM memory_frames_fts") == 0
    assert mind_db.scalar("SELECT COUNT(*) FROM memory_frame_chunks") == 0


def test_public_repair_revalidates_after_index_state(frame_store, mind_db, cfg, monkeypatch):
    event = SourceEvent(event_id="repair", source_ref="doc:repair", content="Private original")
    frame = ingest_source_event(event, store=frame_store, cfg=cfg).frames[0]
    original = frame_store.index_state
    interleaved = []

    def retract_before_state(*args, **kwargs):
        if not interleaved:
            interleaved.append(True)
            retract_source(frame_store, event.source_ref)
        return original(*args, **kwargs)

    monkeypatch.setattr(frame_store, "index_state", retract_before_state)
    repaired = frame_store.ensure_indexed(frame)
    assert mind_db.scalar("SELECT COUNT(*) FROM memory_frames_fts") == 0
    assert mind_db.scalar("SELECT COUNT(*) FROM memory_frame_chunks") == 0
    assert not repaired
    assert interleaved
    assert frame_store.get_frame(frame.id).content == "[forgotten]"
    assert mind_db.scalar("SELECT COUNT(*) FROM memory_frames_fts") == 0
    assert mind_db.scalar("SELECT COUNT(*) FROM memory_frame_chunks") == 0


@pytest.mark.parametrize("flags", [{}, {"verified_by_user": True}, {"pinned": True}])
@pytest.mark.parametrize("phase", ["index_state", "graph"])
def test_standalone_admission_preserves_reentrant_erasure(
    frame_store, mind_db, cfg, monkeypatch, flags, phase, kg, assert_no_recall
):
    event = SourceEvent(
        event_id="reentrant", source_ref="doc:reentrant", content="Private original"
    )
    frame = ingest_source_event(event, store=frame_store, cfg=cfg).frames[0]
    entity = kg.create_entity("person", event.content)
    kg.link_entity_frame(entity.id, frame.id)
    target = frame_store if phase == "index_state" else kg
    method = "index_state" if phase == "index_state" else "link_entity_frame"
    original = getattr(target, method)
    calls = []

    def retract_before_state(*args, **kwargs):
        if not calls:
            calls.append(mind_db.backend.in_transaction)
            retract_source(frame_store, event.source_ref)
        return original(*args, **kwargs)

    monkeypatch.setattr(target, method, retract_before_state)
    result = admit(
        frame.content,
        gop=frame.gop_id,
        scanner=make_scanner(cfg),
        store=frame_store,
        kg=kg,
        entity_ids=[entity.id],
        cfg=cfg,
        **flags,
    )
    assert frame_store.get_frame(frame.id).content == "[forgotten]"
    assert mind_db.scalar("SELECT COUNT(*) FROM source_retractions") == 1
    assert mind_db.scalar("SELECT COUNT(*) FROM memory_frames_fts") == 0
    assert mind_db.scalar("SELECT COUNT(*) FROM memory_frame_chunks") == 0
    assert mind_db.scalar("SELECT COUNT(*) FROM memory_frames_vec") == 0
    assert mind_db.scalar("SELECT COUNT(*) FROM kg_entity_frames") == 0
    assert_no_recall(event.content, db=mind_db, store=frame_store, kg=kg)
    assert not result.admitted
    assert result.frame is None
    assert calls == [True]


def test_rebuild_rejects_erased_prepared_snapshot(frame_store, mind_db, cfg, monkeypatch):
    event = SourceEvent(event_id="rebuild", source_ref="doc:rebuild", content="Private original")
    frame = ingest_source_event(event, store=frame_store, cfg=cfg).frames[0]
    original = frame_store.embedder.embed
    calls = []

    def retract_during_embedding(text):
        result = original(text)
        if not calls:
            calls.append(True)
            retract_source(frame_store, event.source_ref)
        return result

    monkeypatch.setattr(frame_store.embedder, "embed", retract_during_embedding)
    with pytest.raises(ValueError, match="changed during index rebuild"):
        type(frame_store).rebuild_index(mind_db, frame_store.embedder, chunking=cfg.chunking)
    assert frame_store.get_frame(frame.id).content == "[forgotten]"
    assert mind_db.scalar("SELECT COUNT(*) FROM source_retractions") == 1
    assert mind_db.scalar("SELECT COUNT(*) FROM memory_frames_fts") == 0
    assert mind_db.scalar("SELECT COUNT(*) FROM memory_frame_chunks") == 0


@pytest.mark.parametrize("phase", ["index_state", "graph"])
def test_semantic_dedup_preserves_reentrant_erasure(
    frame_store, mind_db, cfg, monkeypatch, phase, kg, assert_no_recall
):
    from gluomem.write.dedup import SemanticDeduper

    event = SourceEvent(event_id="semantic", source_ref="doc:semantic", content="Private original")
    frame = ingest_source_event(event, store=frame_store, cfg=cfg).frames[0]
    deduper = SemanticDeduper(frame_store.embedder, threshold=-1.0)
    deduper.record(frame.gop_id, frame.id, frame_store.embedder.embed(frame.content))
    entity = kg.create_entity("person", event.content)
    kg.link_entity_frame(entity.id, frame.id)
    target = frame_store if phase == "index_state" else kg
    method = "index_state" if phase == "index_state" else "link_entity_frame"
    original = getattr(target, method)
    calls = []

    def erase(*args, **kwargs):
        if not calls:
            calls.append(True)
            retract_source(frame_store, event.source_ref)
        return original(*args, **kwargs)

    monkeypatch.setattr(target, method, erase)
    before = current_scope_generation(mind_db.backend, frame.scope)
    result = admit(
        "Paraphrase",
        gop=frame.gop_id,
        scanner=make_scanner(cfg),
        store=frame_store,
        kg=kg,
        entity_ids=[entity.id],
        cfg=cfg,
        deduper=deduper,
    )
    assert not result.admitted
    assert result.frame is None
    assert current_scope_generation(mind_db.backend, frame.scope) == before + 1
    assert mind_db.scalar("SELECT COUNT(*) FROM source_retractions") == 1
    assert_no_recall(event.content, db=mind_db, store=frame_store, kg=kg)


def test_repair_rejects_mismatched_prepared_text(frame_store, mind_db):
    frame = frame_store.create_i_frame("prepared", "Authoritative value")
    stale = frame_store._prepare_index("Wrong private value")
    frame_store._delete_index_rows(frame.id)
    assert frame_store.ensure_indexed(frame, prepared=stale)
    assert mind_db.fts.get_text(frame.id) == frame.content
    assert "Wrong private value" not in str(
        [dict(row) for row in mind_db.fetchall("SELECT * FROM memory_frame_chunks")]
    )


def test_repair_does_not_bump_generation(frame_store, mind_db):
    frame = frame_store.create_i_frame("repair-counter", "Harmless value")
    before = current_scope_generation(mind_db.backend, frame.scope)
    frame_store._delete_index_rows(frame.id)
    assert frame_store.ensure_indexed(frame)
    assert current_scope_generation(mind_db.backend, frame.scope) == before
