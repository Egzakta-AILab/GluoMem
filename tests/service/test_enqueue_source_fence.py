"""A removed source must not enter the durable queue again."""

import pytest

from gluomem.frames.lifecycle import retract_source
from gluomem.frames.source_fences import SourceRetractedError
from gluomem.service.jobs import JobQueue
from gluomem.storage.scope import MemoryScope


def test_enqueue_rejects_retracted_source_without_affecting_siblings(mind_db, frame_store) -> None:
    queue = JobQueue(mind_db.backend)
    retract_source(frame_store, "removed-document")
    payload = {
        "event_id": "replayed",
        "source_ref": "removed-document",
        "content": "This plaintext must not be re-enqueued.",
    }
    with pytest.raises(SourceRetractedError):
        queue.enqueue(dedup_key="replayed:1", payload=payload)
    assert queue.pending_count(frame_store.scope) == 0
    sibling = MemoryScope("other-tenant", "other-user")
    assert queue.enqueue(sibling, dedup_key="replayed:1", payload=payload).status == "queued"
