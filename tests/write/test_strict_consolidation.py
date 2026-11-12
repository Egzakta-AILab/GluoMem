"""Portable critical contracts; authored inputs and deterministic embeddings only."""

import pytest

from gluomem.llm.fake import FakeExtractors
from gluomem.llm.outputs import ConsolidationDecision, ExtractedClaimSpec
from gluomem.write.events import SourceEvent, commit_source_event, prepare_source_event


def prepare_correction(store, cfg, decision, **event_options):
    return prepare_source_event(
        SourceEvent(
            event_id="correction",
            source_ref="chat:correction",
            content="Database port is 6432.",
            **event_options,
        ),
        store=store,
        cfg=cfg,
        extractors=FakeExtractors(
            event_claims=[[ExtractedClaimSpec(text="Database port is 6432.")]],
            consolidation_decisions=[decision],
            banner=False,
        ),
    )


@pytest.mark.parametrize("foreign", [False, True], ids=["unobserved", "cross-scope"])
def test_commit_refuses_unobserved_target(frame_store, cfg, foreign):
    from gluomem.storage.scope import MemoryScope

    original = frame_store.create_i_frame("database", "Database port is 5432.")
    plan = prepare_correction(
        frame_store, cfg, ConsolidationDecision(action="SUPERSEDE", target_frame_id=original.id)
    )
    unseen = frame_store.create_i_frame(
        "other",
        "Another database port is 1234.",
        scope=MemoryScope("other", "owner", "private") if foreign else None,
    )
    plan.decisions[0] = ConsolidationDecision(action="SUPERSEDE", target_frame_id=unseen.id)
    with pytest.raises(ValueError, match="outside the prepared candidate set"):
        commit_source_event(plan, store=frame_store)
    assert frame_store.get_frame(original.id) == original
    assert frame_store.get_frame(unseen.id, scope=unseen.scope) == unseen
    assert frame_store._backend.scalar("SELECT COUNT(*) FROM source_events") == 0
    assert frame_store._backend.scalar("SELECT COUNT(*) FROM memory_provenance") == 0


@pytest.mark.parametrize("update", [{"pinned": 1}, {"trust": "low"}, {"status": "candidate"}])
def test_stale_policy_revision_cannot_publish(frame_store, cfg, update):
    original = frame_store.create_i_frame("database", "Database port is 5432.")
    plan = prepare_correction(
        frame_store, cfg, ConsolidationDecision(action="SUPERSEDE", target_frame_id=original.id)
    )
    with frame_store.db.transaction():
        frame_store._update(original.id, original.scope, **update)
    changed = frame_store.get_frame(original.id)
    with pytest.raises(ValueError, match="Stale consolidation target"):
        commit_source_event(plan, store=frame_store)
    assert frame_store.all_frames() == [changed]
    assert frame_store._backend.scalar("SELECT COUNT(*) FROM source_events") == 0


@pytest.mark.parametrize("erase", ["forget", "retract"])
def test_reentrant_erasure_survives_refused_correction(frame_store, cfg, monkeypatch, erase):
    from gluomem.frames.lifecycle import forget_memory, retract_source
    from gluomem.write import gate
    from gluomem.write.events import ingest_source_event

    original = ingest_source_event(
        SourceEvent(event_id="old", source_ref="chat:old", content="Database port is 5432."),
        store=frame_store,
        cfg=cfg,
    ).frames[0]
    plan = prepare_correction(
        frame_store, cfg, ConsolidationDecision(action="SUPERSEDE", target_frame_id=original.id)
    )
    real_admit = gate.admit
    calls = []

    def erase_during_admission(*args, **kwargs):
        admitted = real_admit(*args, **kwargs)
        calls.append(True)
        if erase == "forget":
            forget_memory(frame_store, original.id)
        else:
            retract_source(frame_store, "chat:old")
        return admitted

    monkeypatch.setattr(gate, "admit", erase_during_admission)
    with pytest.raises(ValueError, match="Stale consolidation target"):
        commit_source_event(plan, store=frame_store)
    assert calls == [True]
    erased = frame_store.get_frame(original.id)
    assert erased.status == "forgotten" and erased.content == "[forgotten]"
    assert (
        frame_store._backend.scalar("SELECT COUNT(*) FROM memory_frames WHERE status='active'") == 0
    )
    assert frame_store._backend.scalar("SELECT COUNT(*) FROM memory_frames_fts") == 0
    assert frame_store._backend.scalar("SELECT COUNT(*) FROM memory_frame_chunks") == 0
    assert frame_store._backend.scalar("SELECT COUNT(*) FROM memory_tombstones") > 0
    assert (
        frame_store._backend.scalar(
            "SELECT COUNT(*) FROM source_events WHERE event_id='correction'"
        )
        == 0
    )
    assert "5432" not in str(frame_store._backend.fetchall("SELECT * FROM memory_provenance"))
    if erase == "retract":
        assert frame_store._backend.scalar("SELECT COUNT(*) FROM source_retractions") == 1


def test_later_erasure_rolls_back_earlier_event_correction(frame_store, cfg, monkeypatch):
    from gluomem.frames.lifecycle import forget_memory
    from gluomem.write import gate
    from gluomem.write.events import ingest_source_event

    first = frame_store.create_i_frame("database", "Database primary port is 5432.")
    second = frame_store.create_i_frame("database", "Database replica port is 5433.")
    ext = FakeExtractors(
        event_claims=[
            [
                ExtractedClaimSpec(text="Database primary port is 6432."),
                ExtractedClaimSpec(text="Database replica port is 6433."),
            ]
        ],
        consolidation_decisions=[
            ConsolidationDecision(action="SUPERSEDE", target_frame_id=first.id),
            ConsolidationDecision(action="SUPERSEDE", target_frame_id=second.id),
        ],
        banner=False,
    )
    real_admit = gate.admit
    calls = []

    def erase(*args, **kwargs):
        result = real_admit(*args, **kwargs)
        calls.append(True)
        if len(calls) == 2:
            forget_memory(frame_store, second.id)
        return result

    monkeypatch.setattr(gate, "admit", erase)
    with pytest.raises(ValueError, match="Stale consolidation target"):
        ingest_source_event(
            SourceEvent(event_id="atomic", source_ref="chat:atomic", content="Port corrections."),
            store=frame_store,
            cfg=cfg,
            extractors=ext,
        )
    assert len(calls) == 2
    assert frame_store.get_frame(first.id) == first
    assert frame_store.get_frame(second.id).status == "forgotten"
    assert frame_store.count_frames() == 2
    assert frame_store._backend.scalar("SELECT COUNT(*) FROM source_events") == 0
    assert frame_store._backend.scalar("SELECT COUNT(*) FROM memory_provenance") == 0
    assert frame_store.lineage(first.id) == []


def test_reentrant_dedup_erasure_is_not_a_success_receipt(frame_store, cfg, monkeypatch):
    from gluomem.frames.lifecycle import forget_memory

    original = frame_store.create_i_frame("database", "Database port is 6432.")
    plan = prepare_correction(
        frame_store,
        cfg,
        ConsolidationDecision(action="SUPERSEDE", target_frame_id=original.id),
        pinned=True,
    )
    index_state = frame_store.index_state
    calls = []

    def erase(*args, **kwargs):
        if not calls:
            calls.append(True)
            forget_memory(frame_store, original.id)
        return index_state(*args, **kwargs)

    monkeypatch.setattr(frame_store, "index_state", erase)
    with pytest.raises(ValueError, match="Stale consolidation target"):
        commit_source_event(plan, store=frame_store)
    assert calls == [True]
    assert frame_store.get_frame(original.id).status == "forgotten"
    assert frame_store._backend.scalar("SELECT COUNT(*) FROM source_events") == 0
