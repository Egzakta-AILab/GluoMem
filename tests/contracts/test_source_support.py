"""Portable critical contracts; authored inputs and deterministic embeddings only."""

import pytest

from gluomem.llm.outputs import ExtractedClaimSpec
from gluomem.search.evidence import evidence_frames
from gluomem.service.jobs import JobQueue
from gluomem.write.events import (
    SourceEvent,
    SourceEventMessage,
    commit_source_event,
    prepare_source_event,
)
from tests._source_helpers import (
    AuthoredProjection,
    SafeScanner,
    choice,
    config,
)

TARGET = "The workshop gate opens at eight."

SIBLING = "The workshop canopy is violet."

REPLACEMENT = "The workshop gate opens at ten."


def ingest(store, event, specs=None, decisions=None):
    facade = AuthoredProjection(specs, decisions) if specs is not None else None
    plan = prepare_source_event(
        event, store=store, cfg=config(), scanner=SafeScanner(), extractors=facade
    )
    result = commit_source_event(plan, store=store)
    if facade is not None:
        assert [d.model_dump() for d in plan.decisions] == decisions == facade.observed
    return plan, result


def shared_source(store):
    event = SourceEvent(
        event_id="shared",
        source_ref="authored:shared",
        capture_policy="attributed-interaction",
        kind="fact",
        created_at="2026-01-01T08:00:00Z",
        messages=[
            SourceEventMessage(message_id="target", content=TARGET),
            SourceEventMessage(
                message_id="sibling",
                role="assistant",
                content=SIBLING,
                timestamp="2026-01-01T07:59:00Z",
            ),
        ],
    )
    plan, result = ingest(store, event)
    return event, plan, result


def supersede(store, target):
    event = SourceEvent(
        event_id="correction",
        source_ref="authored:correction",
        capture_policy="trusted-only",
        created_at="2026-01-02T08:00:00Z",
        messages=[SourceEventMessage(message_id="correct", content=REPLACEMENT)],
    )
    return ingest(
        store,
        event,
        [ExtractedClaimSpec(text=REPLACEMENT, message_id="correct", verbatim_span=REPLACEMENT)],
        [choice("SUPERSEDE", frame_id=target.id)],
    )


def proof(store, frame):
    return evidence_frames(store, [store.get_frame(frame.id)], store.scope)[0].provenance


def test_ordinary_supersession_preserves_cross_message_sibling_proof(frame_store):
    event, initial_plan, initial = shared_source(frame_store)
    target, sibling = initial.frames
    queue = JobQueue(frame_store.db.backend)
    job = queue.enqueue(dedup_key="shared:1", payload=event.model_dump(mode="json"))
    assert proof(frame_store, sibling)[0].span_verified
    _, correction = supersede(frame_store, target)
    replacement = correction.frames[0]
    retired = frame_store.get_frame(target.id)
    assert retired.status == "superseded" and retired.content == "[forgotten]"
    assert retired.superseded_by == replacement.id and replacement.supersedes == target.id
    assert frame_store.lineage(replacement.id)[0].operation == "supersede"
    assert frame_store.get_frame(sibling.id) == sibling
    row = frame_store.db.fetchone(
        "SELECT payload_json, payload_hash FROM source_events WHERE id = ?", (initial.event_row_id,)
    )
    assert row["payload_json"] == "{}" and row["payload_hash"] == initial_plan.payload_hash
    assert queue.get_job(job.id).payload == {}
    record = proof(frame_store, sibling)[0]
    assert (record.verbatim_span, record.span_verified) == (SIBLING, True)
    assert (record.message_role, record.message_time) == ("assistant", "2026-01-01T07:59:00Z")
    assert record.observed_at == "2026-01-01T07:59:00Z"
    assert proof(frame_store, replacement)[0].span_verified


@pytest.mark.parametrize("quote_mode", ["disjoint", "full-message", "overlapping", "ambiguous"])
def test_same_message_fragment_boundaries_are_never_trimmed_or_fabricated(frame_store, quote_mode):
    text = TARGET + " " + SIBLING
    if quote_mode == "ambiguous":
        text += " " + SIBLING
    quote = {
        "disjoint": SIBLING,
        "full-message": text,
        "overlapping": "at eight. " + SIBLING,
        "ambiguous": SIBLING,
    }[quote_mode]
    event = SourceEvent(
        event_id="one-message",
        source_ref="authored:one-message",
        capture_policy="trusted-only",
        created_at="2026-01-01T08:00:00Z",
        messages=[SourceEventMessage(message_id="m", content=text)],
    )
    specs = [
        ExtractedClaimSpec(text=TARGET, message_id="m", verbatim_span=TARGET),
        ExtractedClaimSpec(text=SIBLING, message_id="m", verbatim_span=quote),
    ]
    _, initial = ingest(frame_store, event, specs, [choice("ADD"), choice("ADD")])
    target, sibling = initial.frames
    assert proof(frame_store, sibling)[0].verbatim_span == quote
    supersede(frame_store, target)
    record = proof(frame_store, sibling)[0]
    assert frame_store.get_frame(sibling.id) == sibling
    if quote_mode == "disjoint":
        assert (record.verbatim_span, record.span_verified) == (quote, True)
        witness = dict(frame_store.db.fetchone("SELECT * FROM memory_source_support"))
        assert witness["char_start"] == text.index(quote)
        assert witness["char_end"] == text.index(quote) + len(quote)
        assert all(not isinstance(v, str) or quote not in v for v in witness.values())
    else:
        assert (record.verbatim_span, record.span_verified) == (None, False)
        assert "verification_basis" not in record.model_dump()
        assert frame_store.db.scalar("SELECT COUNT(*) FROM memory_source_support") == 0
        assert (
            frame_store.db.scalar(
                "SELECT verbatim_span FROM memory_provenance WHERE frame_id = ?", (sibling.id,)
            )
            is None
        )
    assert (
        frame_store.db.scalar(
            "SELECT payload_json FROM source_events WHERE id = ?", (initial.event_row_id,)
        )
        == "{}"
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "quote",
        "role",
        "time",
        "source",
        "message",
        "frame",
        "observed",
        "created",
        "scope",
        "hash",
        "offset",
        "witness-time",
    ],
)
def test_retained_witness_is_bound_to_exact_scope_source_frame_span_role_time(frame_store, tamper):
    _, _, initial = shared_source(frame_store)
    target, sibling = initial.frames
    supersede(frame_store, target)
    p = proof(frame_store, sibling)[0]
    assert p.span_verified
    with frame_store.db.transaction():
        table, column, value = {
            "quote": ("memory_provenance", "verbatim_span", SIBLING + " invented"),
            "role": ("memory_provenance", "message_role", "user"),
            "time": ("source_events", "created_at", "1999-01-01"),
            "source": ("memory_provenance", "source_ref", "forged"),
            "message": ("memory_provenance", "message_id", "target"),
            "frame": ("memory_frames", "content_hash", "forged"),
            "observed": ("memory_provenance", "observed_at", "1999-01-01"),
            "created": ("memory_provenance", "created_at", "1999-01-01"),
            "scope": ("memory_source_support", "namespace", "other"),
            "hash": ("memory_source_support", "event_payload_hash", "forged"),
            "offset": ("memory_source_support", "char_start", 1),
            "witness-time": ("memory_source_support", "message_time", "1999-01-01"),
        }[tamper]
        key = "provenance_id" if table == "memory_source_support" else "id"
        row_id = {
            "memory_provenance": p.id,
            "memory_source_support": p.id,
            "source_events": initial.event_row_id,
            "memory_frames": sibling.id,
        }[table]
        frame_store.db.execute(f"UPDATE {table} SET {column} = ? WHERE {key} = ?", (value, row_id))
    projected = proof(frame_store, sibling)[0]
    assert not projected.span_verified and projected.verbatim_span is None
    assert "verification_basis" not in projected.model_dump()


@pytest.mark.parametrize("operation", ["forget-sibling", "forget-target", "retract", "scope"])
def test_retained_proofs_are_revocable_and_prepared_duplicates_do_not_restore_them(
    frame_store,
    operation,
    kg,
    assert_no_recall,
):
    from gluomem.frames.lifecycle import forget_memory, forget_scope, retract_source
    from gluomem.frames.source_fences import SourceRetractedError

    event, _, initial = shared_source(frame_store)
    target, sibling = initial.frames
    independent_event = SourceEvent(
        event_id="independent",
        source_ref="authored:independent",
        kind="fact",
        content=SIBLING,
        created_at="2026-01-01T09:00:00Z",
    )
    _, independent = ingest(frame_store, independent_event)
    assert independent.frames[0].id == sibling.id
    duplicate = prepare_source_event(event, store=frame_store, cfg=config(), scanner=SafeScanner())
    _, correction = supersede(frame_store, target)
    assert [p.verification_basis for p in proof(frame_store, sibling)] == [
        "retained-fragment",
        "raw-source",
    ]
    if operation == "forget-sibling":
        forget_memory(frame_store, sibling.id)
        assert_no_recall(SIBLING, db=frame_store.db, store=frame_store, kg=kg)
    elif operation == "forget-target":
        forget_memory(frame_store, target.id)
        assert frame_store.get_frame(correction.frames[0].id).content == "[forgotten]"
        assert [p.span_verified for p in proof(frame_store, sibling)] == [False, True]
    elif operation == "retract":
        removed = retract_source(frame_store, event.source_ref)
        assert sibling.id in removed.retained_frame_ids
        assert [p.source_event_id for p in proof(frame_store, sibling)] == [
            independent.event_row_id
        ]
        with pytest.raises(SourceRetractedError):
            commit_source_event(duplicate, store=frame_store)
        with pytest.raises(SourceRetractedError):
            prepare_source_event(
                event.model_copy(update={"force_reassert": True}),
                store=frame_store,
                cfg=config(),
                scanner=SafeScanner(),
            )
    else:
        forget_scope(frame_store)
        assert_no_recall(SIBLING, db=frame_store.db, store=frame_store, kg=kg)
    assert frame_store.db.scalar("SELECT COUNT(*) FROM memory_source_support") == 0
    if operation != "retract":
        replay = commit_source_event(duplicate, store=frame_store)
        assert replay.deduplicated
        assert all(
            p.verbatim_span != SIBLING or p.source_event_id == independent.event_row_id
            for p in replay.provenance
        )
        assert (
            frame_store.db.scalar(
                "SELECT payload_json FROM source_events WHERE id = ?", (initial.event_row_id,)
            )
            == "{}"
        )


def test_mint_and_revocation_follow_transaction_rollback(frame_store):
    from gluomem.frames.lifecycle import correct_memory, forget_memory

    _, _, initial = shared_source(frame_store)
    target, sibling = initial.frames
    with pytest.raises(RuntimeError, match="rollback-mint"):
        with frame_store.db.transaction():
            correct_memory(frame_store, target.id, REPLACEMENT)
            assert proof(frame_store, sibling)[0].verification_basis == "retained-fragment"
            raise RuntimeError("rollback-mint")
    assert frame_store.get_frame(target.id) == target
    assert proof(frame_store, sibling)[0].verification_basis == "raw-source"
    assert frame_store.db.scalar("SELECT COUNT(*) FROM memory_source_support") == 0
    supersede(frame_store, target)
    witness = dict(frame_store.db.fetchone("SELECT * FROM memory_source_support"))
    with pytest.raises(RuntimeError, match="rollback-revoke"):
        with frame_store.db.transaction():
            forget_memory(frame_store, sibling.id)
            assert frame_store.db.scalar("SELECT COUNT(*) FROM memory_source_support") == 0
            raise RuntimeError("rollback-revoke")
    assert dict(frame_store.db.fetchone("SELECT * FROM memory_source_support")) == witness
    assert proof(frame_store, sibling)[0].verification_basis == "retained-fragment"
