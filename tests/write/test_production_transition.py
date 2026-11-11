"""Portable critical contracts; authored inputs and deterministic embeddings only."""

from datetime import UTC, datetime

from gluomem.frames.temporal import TemporalPolicy, TemporalQuery, TemporalStore
from gluomem.llm import ConsolidationDecision, ExtractedClaimSpec, FakeExtractors
from gluomem.write.events import (
    SourceEvent,
    SourceEventMessage,
    ingest_source_event,
)


def ingest_state(store, cfg, *, event_id, text, observed_at, effective_at, decision=None):
    return ingest_source_event(
        SourceEvent(
            event_id=event_id,
            source_ref=f"chat:{event_id}",
            capture_policy="trusted-only",
            created_at=observed_at,
            messages=[
                SourceEventMessage(
                    message_id=event_id,
                    role="user",
                    content=text,
                    timestamp=observed_at,
                )
            ],
        ),
        store=store,
        cfg=cfg,
        extractors=FakeExtractors(
            event_claims=[
                [
                    ExtractedClaimSpec(
                        text=text,
                        message_id=event_id,
                        verbatim_span=text,
                        event_date=datetime.fromisoformat(effective_at).replace(tzinfo=UTC)
                        if effective_at
                        else None,
                    )
                ]
            ],
            consolidation_decisions=[decision] if decision else [],
            banner=False,
        ),
    ).frames[0]


def test_strict_writer_never_invents_effective_date_from_message_timestamp(frame_store, cfg):
    frame = ingest_state(
        frame_store,
        cfg,
        event_id="undated",
        text="Nora owns a telescope.",
        observed_at="2025-01-02T08:00:00Z",
        effective_at=None,
    )
    metadata = TemporalStore(frame_store).metadata(frame.id)
    assert metadata is not None
    assert metadata.observation.effective_at is None


def test_writer_ordinary_dated_move_preserves_historical_truth(frame_store, cfg):
    old = ingest_state(
        frame_store,
        cfg,
        event_id="initial",
        text="Nora lives in Oslo.",
        observed_at="2025-01-02T08:00:00Z",
        effective_at="2025-01-01",
    )
    new = ingest_state(
        frame_store,
        cfg,
        event_id="move",
        text="Nora lives in Lyon starting June 1, 2025.",
        observed_at="2025-05-01T08:00:00Z",
        effective_at="2025-06-01",
        decision=ConsolidationDecision.model_construct(action="TRANSITION", target_frame_id=old.id),
    )
    assert frame_store.get_frame(old.id) == old
    assert new.frame_type == "I" and new.base_frame_id is None
    assert new.supersedes is None and frame_store.get_frame(old.id).superseded_by is None
    assert new.gop_id != old.gop_id  # GOP remains source/topic identity, not state precedence.
    temporal = TemporalStore(frame_store)
    metadata = temporal.metadata(new.id)
    assert metadata is not None
    assert metadata.operation == "transition" and metadata.previous_frame_id == old.id
    assert metadata.observation.observed_at == "2025-05-01T08:00:00.000000Z"
    assert metadata.observation.effective_at == "2025-06-01T00:00:00.000000Z"
    policy = TemporalPolicy(allow_historical=True)
    for seed in (old.id, new.id):
        for as_of, known_at, expected in [
            (None, None, new.id),
            ("2025-03-01", None, old.id),
            ("2025-07-01", "2025-04-01", old.id),
            ("2025-07-01", "2025-05-02", new.id),
        ]:
            resolved = temporal.resolve(
                seed,
                policy=policy,
                query=TemporalQuery(
                    as_of=as_of,
                    known_at=known_at,
                    now="2025-07-01",
                ),
            )
            assert resolved.status == "resolved" and resolved.frame_id == expected
    assert frame_store.lineage(new.id) == []  # No destructive correction/B annotation.


def initial_state(store, cfg):
    return ingest_state(
        store,
        cfg,
        event_id="initial",
        text="Nora lives in Oslo.",
        observed_at="2025-01-02T08:00:00Z",
        effective_at="2025-01-01",
    )


def test_production_timeline_move_then_wrong_assertion_correction_has_usable_current_and_history(
    frame_store,
    mind_db,
    cfg,
    monkeypatch,
):
    from gluomem.client.sdk import EmbeddedMemoryClient

    cfg = cfg.with_overrides({"search.abstain.enabled": False})
    old = initial_state(frame_store, cfg)
    moved = ingest_state(
        frame_store,
        cfg,
        event_id="move",
        text="Nora lives in Lyon starting June 1, 2025.",
        observed_at="2025-05-01T08:00:00Z",
        effective_at="2025-06-01",
        decision=ConsolidationDecision(action="TRANSITION", target_frame_id=old.id),
    )
    assert frame_store.get_frame(old.id).content == old.content
    corrected = ingest_state(
        frame_store,
        cfg,
        event_id="correction",
        text="Nora lives in Paris starting June 1, 2025.",
        observed_at="2025-08-01T08:00:00Z",
        effective_at="2025-06-01",
        decision=ConsolidationDecision(action="SUPERSEDE", target_frame_id=moved.id),
    )
    assert corrected.frame_type == "I" and corrected.supersedes == moved.id
    assert frame_store.get_frame(moved.id).content == "[forgotten]"
    assert frame_store.get_frame(moved.id).superseded_by == corrected.id
    assert [(x.source_frame_id, x.operation) for x in frame_store.lineage(corrected.id)] == [
        (moved.id, "supersede")
    ]
    moved_metadata = TemporalStore(frame_store).metadata(moved.id)
    corrected_metadata = TemporalStore(frame_store).metadata(corrected.id)
    assert moved_metadata is not None and moved_metadata.operation == "transition"
    assert corrected_metadata is not None and corrected_metadata.operation == "observation"
    monkeypatch.setattr(mind_db.chunk_vec, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(mind_db.frame_vec, "search", lambda *args, **kwargs: [])
    client = EmbeddedMemoryClient(
        mind_db,
        frame_store,
        cfg=cfg,
        temporal_policy=TemporalPolicy(allow_historical=True),
    )
    current = client.recall("Nora lives", k=1)
    assert [hit.frame.id for hit in current.hits] == [corrected.id]
    assert (
        "Paris" in current.context
        and "Lyon" not in current.context
        and "Oslo" not in current.context
    )
    historical = client.recall("Nora lives", k=1, as_of="2025-03-01")
    assert [hit.frame.id for hit in historical.hits] == [old.id]
    assert "Oslo" in historical.context and "Lyon" not in historical.context
    # Known-at cannot resurrect the now-corrected false Lyon assertion.
    earlier_knowledge = client.recall("Nora lives", k=1, as_of="2025-07-01", known_at="2025-06-01")
    assert not earlier_knowledge.hits and not earlier_knowledge.context
