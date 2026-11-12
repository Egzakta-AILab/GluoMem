"""Trusted source identity must not be repaired from a model's unsupported ID."""

from typing import Literal, cast

import pytest

from gluomem.write.events import SourceEvent, SourceEventMessage


def test_unknown_message_identity_is_not_replaced_by_only_eligible_message():
    from gluomem.write.source_binding import bind_source_claim

    event = SourceEvent(
        event_id="identity-1",
        source_ref="authored/identity",
        messages=[SourceEventMessage(message_id="m1", role="user", content="Code is LIME-427.")],
    )
    with pytest.raises(ValueError, match="source message"):
        bind_source_claim(event, message_id="fabricated", quote="Code is LIME-427.")


def test_assistant_capture_is_explicit_and_remains_attributed():
    from gluomem.write.source_binding import bind_source_claim

    event = SourceEvent(
        event_id="assistant-1",
        source_ref="authored/assistant",
        messages=[
            SourceEventMessage(
                message_id="a1",
                role="assistant",
                content="I recommended the blue trail.",
                timestamp="2026-01-03T08:00:00Z",
            )
        ],
    )
    with pytest.raises(ValueError, match="capture policy"):
        bind_source_claim(event, message_id="a1", quote="I recommended the blue trail.")
    bound = bind_source_claim(
        event,
        message_id="a1",
        quote="I recommended the blue trail.",
        capture_policy="attributed-interaction",
    )
    assert bound.message_id == "a1"
    assert bound.role == "assistant"
    assert bound.frame_source == "agent_inferred"
    assert bound.observed_at == "2026-01-03T08:00:00Z"


@pytest.mark.parametrize("unsupported_quote", ["", "   ", "Code is AMBER-693."])
def test_supporting_span_belongs_to_selected_source_without_normalization(unsupported_quote):
    from gluomem.write.source_binding import bind_source_claim

    literal = "  Code is LIME-427.\n"
    event = SourceEvent(
        event_id="quote-1",
        source_ref="authored/quote",
        messages=[
            SourceEventMessage(message_id="m1", role="user", content=literal),
            SourceEventMessage(message_id="m2", role="user", content="Code is AMBER-693."),
        ],
    )
    with pytest.raises(ValueError, match="supporting span"):
        bind_source_claim(event, message_id="m1", quote=unsupported_quote)
    bound = bind_source_claim(event, message_id="m1", quote=literal)
    assert bound.verbatim_span == literal
    assert bound.frame_source == "user_stated"


def test_document_binding_does_not_confuse_world_time_with_learned_time():
    from gluomem.write.source_binding import bind_source_claim

    text = "  The expedition departs in May.\n"
    event = SourceEvent(
        event_id="document-1",
        source_ref="authored/plan",
        source_type="document",
        content=text,
        created_at="2026-01-03T08:00:00Z",
        event_time="2026-05-01T00:00:00Z",
    )
    bound = bind_source_claim(event, message_id=None, quote=text)
    assert bound.role == "document"
    assert bound.message_id is None
    assert bound.frame_source == "import"
    assert bound.observed_at == "2026-01-03T08:00:00Z"
    assert bound.verbatim_span == event.content
    with pytest.raises(ValueError, match="source message"):
        bind_source_claim(event, message_id="invented", quote=text)
    with pytest.raises(ValueError, match="supporting span"):
        bind_source_claim(event, message_id=None, quote="The expedition departs in June.")


@pytest.mark.parametrize("identifiers", [["m1", "m1"], [""]])
def test_source_identity_table_must_be_unambiguous(identifiers):
    from gluomem.write.source_binding import bind_source_claim

    event = SourceEvent(
        event_id="ambiguous-1",
        source_ref="authored/ambiguous",
        messages=[
            SourceEventMessage(message_id=mid, role="user", content="Shared text.")
            for mid in identifiers
        ],
    )
    with pytest.raises(ValueError, match="source message identities"):
        bind_source_claim(event, message_id=identifiers[0], quote="Shared text.")


def test_capture_policy_is_explicit_and_closed_in_production_config():
    from pydantic import ValidationError

    from gluomem.config.app_config import WriteConfig

    assert WriteConfig().source_capture_policy is None
    assert WriteConfig(source_capture_policy="attributed-interaction").source_capture_policy == (
        "attributed-interaction"
    )
    with pytest.raises(ValidationError):
        # Deliberately invalid constructor probe: preserve this exact rejected token.
        WriteConfig(
            source_capture_policy=cast(
                Literal["trusted-only", "attributed-interaction"], "assistant-is-user"
            )
        )
