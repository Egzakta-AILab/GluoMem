"""Observed source clocks are not inferred world-effective claim dates."""

import pytest

from gluomem.llm.fake import FakeExtractors
from gluomem.llm.outputs import ExtractedClaimSpec
from gluomem.timeutil import to_storage
from gluomem.write.events import SourceEvent, extract_event_claims_from_envelope


@pytest.mark.parametrize(
    ("policy", "world_date"),
    [
        (None, None),
        ("trusted-only", None),
        ("attributed-interaction", None),
        ("attributed-interaction", "2026-06-01T00:00:00Z"),
    ],
)
def test_strict_claim_validity_requires_explicit_model_date_not_source_clock(policy, world_date):
    source_time = "2026-05-01T08:00:00Z"
    text = (
        "The workshop moves on June 1, 2026."
        if world_date
        else "The earlier workshop statement was correct before the move."
    )
    event = SourceEvent.model_validate(
        {
            "event_id": "validity-clock-contract",
            "source_ref": "authored/validity-clock",
            "event_time": source_time,
            "capture_policy": policy,
            "messages": [
                {"message_id": "m1", "role": "user", "timestamp": source_time, "content": text}
            ],
        }
    )
    spec = ExtractedClaimSpec.model_validate(
        {
            "text": text,
            "message_id": "m1",
            "verbatim_span": text,
            "event_date": world_date,
        }
    )
    facade = FakeExtractors(event_claims=[[spec]], banner=False)
    claims = extract_event_claims_from_envelope(event, extractors=facade)
    assert facade.calls["extract_event_claims"] == 1 and len(claims) == 1
    if spec.event_date is not None:
        assert claims[0].effective_at == to_storage(spec.event_date)
    elif policy is not None:
        assert claims[0].effective_at is None
        assert claims[0].observed_at == source_time
        assert claims[0].event_time == source_time
    else:
        assert claims[0].effective_at == source_time  # Existing legacy envelope behavior.
