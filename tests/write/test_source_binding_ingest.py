"""Production SourceEvents preserve trusted role/quote attribution under strict capture."""

import pytest

from gluomem.llm.fake import FakeExtractors
from gluomem.llm.outputs import ExtractedClaimSpec
from gluomem.write.events import SourceEvent, SourceEventMessage, ingest_source_event


def _strict(cfg):
    return cfg.model_copy(
        update={
            "write": cfg.write.model_copy(
                update={"source_capture_policy": "attributed-interaction"}
            )
        }
    )


def test_production_ingest_keeps_assistant_evidence_distinct_from_user_facts(frame_store, cfg):
    user = "Mara chose the copper trail."
    assistant = "I recommended the blue trail."
    event = SourceEvent(
        event_id="mixed-authority",
        source_ref="authored/mixed-authority",
        created_at="2026-01-07T08:00:00Z",
        messages=[
            SourceEventMessage(message_id="u1", role="user", content=user),
            SourceEventMessage(message_id="a1", role="assistant", content=assistant),
        ],
    )
    extractors = FakeExtractors(
        event_claims=[
            [
                ExtractedClaimSpec(text=user, message_id="u1", verbatim_span=user),
                ExtractedClaimSpec(
                    text="The assistant recommended the blue trail.",
                    message_id="a1",
                    verbatim_span=assistant,
                ),
            ]
        ],
        banner=False,
    )
    result = ingest_source_event(event, store=frame_store, cfg=_strict(cfg), extractors=extractors)
    by_text = {frame.content: frame.source for frame in result.frames}
    assert by_text == {
        user: "user_stated",
        "The assistant recommended the blue trail.": "agent_inferred",
    }
    assert sorted((p.message_id, p.message_role, p.observed_at) for p in result.provenance) == [
        ("a1", "assistant", "2026-01-07T08:00:00Z"),
        ("u1", "user", "2026-01-07T08:00:00Z"),
    ]
    # Attribution must not depend on a source payload that lifecycle may redact.
    from gluomem.frames.provenance import get_frames_provenance

    with frame_store.db.backend.transaction():
        frame_store.db.execute("UPDATE source_events SET payload_json = '{}' ")
    persisted = get_frames_provenance(
        frame_store.db.backend, [f.id for f in result.frames], scope=frame_store.scope
    )
    roles: list[str] = []
    for provenance in persisted:
        assert provenance.message_role is not None
        roles.append(provenance.message_role)
    assert sorted(roles) == ["assistant", "user"]


def test_production_ingest_rejects_unknown_source_id_before_any_mutation(frame_store, cfg):
    text = "Mara chose the copper trail."
    event = SourceEvent(
        event_id="forged-authority",
        source_ref="authored/forged-authority",
        messages=[SourceEventMessage(message_id="u1", role="user", content=text)],
    )
    extractors = FakeExtractors(
        event_claims=[
            [
                ExtractedClaimSpec(text=text, message_id="not-in-envelope", verbatim_span=text),
            ]
        ],
        banner=False,
    )
    with pytest.raises(ValueError, match="source message"):
        ingest_source_event(event, store=frame_store, cfg=_strict(cfg), extractors=extractors)
    assert frame_store.db.scalar("SELECT COUNT(*) FROM memory_frames") == 0
    assert frame_store.db.scalar("SELECT COUNT(*) FROM source_events") == 0
    assert frame_store.db.scalar("SELECT COUNT(*) FROM memory_provenance") == 0
