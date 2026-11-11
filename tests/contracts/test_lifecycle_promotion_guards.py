"""Portable critical contracts; authored inputs and deterministic embeddings only."""

from __future__ import annotations

import pytest

from gluomem.frames.lifecycle import forget_memory, retract_source
from gluomem.frames.provenance import get_frames_provenance
from gluomem.write.events import (
    SourceEvent,
    SourceEventMessage,
    commit_source_event,
    ingest_source_event,
    prepare_source_event,
)


@pytest.mark.parametrize("erasure_boundary", ["before_commit", "write_guard"])
def test_prepared_duplicate_reloads_after_hard_forget(frame_store, mind_db, cfg, erasure_boundary):
    secret = "Prepared duplicate private token CERULEAN."
    safe = "Unrelated authorized fact stays available."
    event = SourceEvent(
        event_id="duplicate-forget",
        source_ref="doc:duplicate-forget",
        messages=[
            SourceEventMessage(message_id="private", content=secret),
            SourceEventMessage(message_id="safe", content=safe),
        ],
    )
    first = ingest_source_event(event, store=frame_store, cfg=cfg)
    removed, retained = first.frames
    prepared = prepare_source_event(event, store=frame_store, cfg=cfg)
    assert prepared.early_duplicate is not None
    assert secret in str(prepared.early_duplicate)

    guards = []

    def erase_in_guard():
        guards.append(removed.id)
        forget_memory(frame_store, removed.id)

    if erasure_boundary == "before_commit":
        forget_memory(frame_store, removed.id)
    result = commit_source_event(
        prepared,
        store=frame_store,
        write_guard=erase_in_guard if erasure_boundary == "write_guard" else None,
    )
    assert guards == ([removed.id] if erasure_boundary == "write_guard" else [])

    assert result.deduplicated
    assert [f.id for f in result.frames] == [retained.id]
    assert result.frames == [frame_store.get_frame(retained.id)]
    assert secret not in str(result)
    current_provenance = {
        p.id: p
        for p in get_frames_provenance(
            mind_db.backend, [removed.id, retained.id], scope=frame_store.scope
        )
    }
    assert all(p == current_provenance[p.id] for p in result.provenance)
    assert frame_store.get_frame(removed.id).content == "[forgotten]"
    assert frame_store.index_state(retained.id).ready
    assert mind_db.scalar("SELECT COUNT(*) FROM source_events") == 1


def _support(store, frame, source_ref, *, span=None):
    """Importer-style provenance without manufacturing a source-event receipt."""
    with store.db.transaction():
        store.db.backend.execute(
            "INSERT INTO memory_provenance "
            "(tenant_id, principal_id, namespace, frame_id, source_type, "
            "source_ref, verbatim_span, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                *store.scope.as_tuple(),
                frame.id,
                "import",
                source_ref,
                frame.indexable_text if span is None else span,
                frame.created_at,
            ),
        )


@pytest.mark.parametrize("operation", ["forget", "retract"])
@pytest.mark.parametrize("erased_input", ["base", "delta"])
def test_compaction_revalidates_erased_inputs_before_publish(
    frame_store, mind_db, cfg, monkeypatch, kg, assert_no_recall, operation, erased_input
):
    cfg = cfg.with_overrides(
        {"write.compaction.p_frame_trigger": 1, "write.compaction.keep_recent": 0}
    )
    base = frame_store.create_i_frame("compact-race", "Base private token MAGENTA.")
    delta = frame_store.create_p_frame(base.gop_id, "Delta private token VIOLET.", base.id)
    survivor = frame_store.create_p_frame(base.gop_id, "Safe refinement remains.", base.id)
    for frame in (base, delta, survivor):
        _support(frame_store, frame, "doc:compact-race")
    unrelated = frame_store.create_i_frame("unrelated", "Keep unrelated independent root.")
    target = base if erased_input == "base" else delta
    prepare = frame_store._prepare_index
    calls = []

    def erase_after_preparing(text):
        result = prepare(text)
        assert not mind_db.in_transaction
        assert not calls
        calls.append(text)
        if operation == "forget":
            forget_memory(frame_store, target.id)
        else:
            # All inputs share a source: withdrawal invalidates the entire plan.
            retract_source(frame_store, "doc:compact-race")
        return result

    monkeypatch.setattr(frame_store, "_prepare_index", erase_after_preparing)
    frame_store.compact(base.gop_id, cfg)

    assert len(calls) == 1
    erased = (
        [base, delta, survivor] if operation == "retract" or erased_input == "base" else [delta]
    )
    for frame in erased:
        assert_no_recall(frame.content, db=mind_db, store=frame_store, kg=kg)
        assert frame_store.get_frame(frame.id).content == "[forgotten]"
    if operation == "forget" and erased_input == "delta":
        assert frame_store.get_frame(base.id).content == base.content
        assert frame_store.get_frame(survivor.id).content == survivor.content
        assert frame_store.index_state(base.id).ready
        assert frame_store.index_state(survivor.id).ready
    assert frame_store.get_frame(unrelated.id).content == unrelated.content
    assert frame_store.index_state(unrelated.id).ready
    assert frame_store.lineage(base.id) == []
    assert mind_db.scalar("SELECT COUNT(*) FROM source_retractions") == int(operation == "retract")


@pytest.mark.parametrize("target_type", ["I", "P"])
@pytest.mark.parametrize("target_status", ["active", "superseded"])
def test_hard_forget_closes_explicit_b_references(
    frame_store, mind_db, cfg, kg, assert_no_recall, target_type, target_status
):
    carrier = frame_store.create_i_frame("b-reference", "Independent authorized carrier.")
    if target_type == "I":
        target = frame_store.create_i_frame(carrier.gop_id, "Original private token AMBER.")
    else:
        target = frame_store.create_p_frame(
            carrier.gop_id, "Original private token AMBER.", carrier.id
        )
    frame_store.set_status(target.id, target_status)
    correction = frame_store.create_b_frame(
        carrier.gop_id, "Correction private token TURQUOISE.", carrier.id, [target.id]
    )
    followup = frame_store.create_b_frame(
        carrier.gop_id, "Transitive private token SCARLET.", carrier.id, [correction.id]
    )
    child = frame_store.create_p_frame(
        carrier.gop_id, "Dependent private token INDIGO.", correction.id
    )
    unrelated = frame_store.create_i_frame(carrier.gop_id, "Unrelated authorized root.")
    for frame in (target, correction, followup, child):
        _support(frame_store, frame, "doc:b-reference")

    forget_memory(frame_store, target.id)

    for frame in (target, correction, followup, child):
        assert frame_store.get_frame(frame.id).status == "forgotten"
        assert_no_recall(frame.indexable_text, db=mind_db, store=frame_store, kg=kg)
    for frame in (carrier, unrelated):
        assert frame_store.get_frame(frame.id).content == frame.content
        assert frame_store.index_state(frame.id).ready


def test_mixed_constituent_sources_are_not_corroboration_for_whole_compaction(
    frame_store, mind_db, cfg, kg, assert_no_recall
):
    cfg = cfg.with_overrides(
        {"write.compaction.p_frame_trigger": 1, "write.compaction.keep_recent": 0}
    )
    base = frame_store.create_i_frame("mixed-support", "Independent authorized foundation.")
    private = frame_store.create_p_frame(base.gop_id, "Exclusive private token LILAC.", base.id)
    safe = frame_store.create_p_frame(base.gop_id, "Independent authorized refinement.", base.id)
    _support(frame_store, base, "doc:foundation")
    _support(frame_store, private, "doc:private")
    _support(frame_store, safe, "doc:safe")
    unrelated = frame_store.create_i_frame("unrelated", "Unrelated root keeps its support.")

    frame_store.compact(base.gop_id, cfg)
    result = retract_source(frame_store, "doc:private")

    assert_no_recall(private.content, db=mind_db, store=frame_store, kg=kg)
    assert result.forgotten_frame_ids == [private.id]
    assert result.retained_frame_ids == []
    for frame in (base, safe, unrelated):
        assert frame_store.get_frame(frame.id).content == frame.content
        assert frame_store.index_state(frame.id).ready
    assert [
        p.source_ref
        for p in get_frames_provenance(mind_db.backend, [base.id], scope=frame_store.scope)
    ] == ["doc:foundation"]


def test_corroborated_constituents_survive_compaction_and_source_withdrawal(
    frame_store, mind_db, cfg
):
    cfg = cfg.with_overrides(
        {"write.compaction.p_frame_trigger": 1, "write.compaction.keep_recent": 0}
    )
    base = frame_store.create_i_frame("shared-compact", "Shared foundation.")
    deltas = [
        frame_store.create_p_frame(base.gop_id, f"Shared refinement {i}.", base.id)
        for i in range(2)
    ]
    for frame in (base, *deltas):
        for source in ("doc:withdrawn", "doc:corroborating"):
            _support(frame_store, frame, source)
    frame_store.compact(base.gop_id, cfg)
    expected = cfg.write.compaction.separator.join(f.content for f in (base, *deltas))
    assert frame_store.get_frame(base.id).content == expected
    result = retract_source(frame_store, "doc:withdrawn")
    assert result.forgotten_frame_ids == []
    assert result.retained_frame_ids == [base.id]
    assert frame_store.get_frame(base.id).content == expected
    assert frame_store.index_state(base.id).ready
    assert {
        p.source_ref
        for p in get_frames_provenance(mind_db.backend, [base.id], scope=frame_store.scope)
    } == {"doc:corroborating"}
