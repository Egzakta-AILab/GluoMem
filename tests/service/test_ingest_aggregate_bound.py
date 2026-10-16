"""Request-wide UTF-8 content budget, including mixed content/message envelopes."""

import pytest
from pydantic import ValidationError

from gluomem.service.schemas import IngestEventRequest


@pytest.mark.parametrize(
    "payload",
    [
        {
            "messages": [
                {"message_id": "a", "content": "x" * 100_000},
                {"message_id": "b", "content": "x"},
            ]
        },
        {"content": "x" * 100_000, "messages": [{"message_id": "a", "content": "x"}]},
        {"content": "\U0001f680" * 25_001},
    ],
)
def test_rejects_aggregate_content_above_100000_utf8_bytes(payload):
    with pytest.raises(ValidationError, match="aggregate content"):
        IngestEventRequest(event_id="bounded", source_ref="test", **payload)


def test_accepts_exact_aggregate_content_budget():
    request = IngestEventRequest(
        event_id="bounded",
        source_ref="test",
        content="x" * 50_000,
        messages=[{"message_id": "a", "content": "\U0001f680" * 12_500}],
    )
    assert request.event_id == "bounded"
