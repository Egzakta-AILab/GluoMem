"""SQL-backed source fences; reused by the live PostgreSQL suite."""

from __future__ import annotations

import pytest

from gluomem.frames.lifecycle import retract_source
from gluomem.write.events import SourceEvent, ingest_source_event


def test_removed_source_without_receipt_cannot_be_reasserted(mind_db, frame_store, cfg):
    retract_source(frame_store, "document:removed")
    event = SourceEvent(
        event_id="new-event",
        source_type="document",
        source_ref="document:removed",
        content="Erased launch credentials",
        force_reassert=True,
    )
    with pytest.raises(ValueError, match="retracted"):
        ingest_source_event(event, store=frame_store, cfg=cfg)
    assert mind_db.scalar("SELECT COUNT(*) FROM memory_frames") == 0
    assert mind_db.scalar("SELECT COUNT(*) FROM source_events") == 0


@pytest.mark.parametrize(
    "other_scope",
    [
        ("other", "default", "personal"),
        ("local", "other", "personal"),
        ("local", "default", "other"),
    ],
)
def test_scope_forget_scrubs_session_summaries_without_cross_scope_damage(
    mind_db,
    frame_store,
    other_scope,
):
    from gluomem.frames.lifecycle import forget_scope
    from gluomem.frames.sessions import end_session, get_or_create_session
    from gluomem.storage.scope import MemoryScope, scope_sql

    other = MemoryScope(*other_scope)
    for scope in (frame_store.scope, other):
        get_or_create_session(mind_db, "summary-only", scope=scope)
        end_session(mind_db, "summary-only", "Sensitive summary", scope=scope)
    forget_scope(frame_store)
    assert (
        mind_db.scalar(
            f"SELECT summary FROM sessions WHERE {scope_sql()}",
            frame_store.scope.as_tuple(),
        )
        is None
    )
    assert (
        mind_db.scalar(
            f"SELECT summary FROM sessions WHERE {scope_sql()}",
            other.as_tuple(),
        )
        == "Sensitive summary"
    )


@pytest.mark.parametrize("operation", ["forget", "correct", "retract"])
def test_frame_erasure_clears_affected_session_summary_only(
    mind_db,
    frame_store,
    cfg,
    operation,
):
    from gluomem.frames.lifecycle import correct_memory, forget_memory
    from gluomem.frames.sessions import end_session, get_or_create_session

    result = ingest_source_event(
        SourceEvent(
            event_id="original", source_ref="document:summary", content="Secret from this document"
        ),
        store=frame_store,
        cfg=cfg,
    )
    frame = result.frames[0]
    end_session(mind_db, frame.gop_id, "Summary of secret from this document")
    get_or_create_session(mind_db, "unrelated")
    end_session(mind_db, "unrelated", "Keep unrelated summary")
    if operation == "forget":
        forget_memory(frame_store, frame.id)
    elif operation == "correct":
        correct_memory(frame_store, frame.id, "New harmless statement")
    else:
        retract_source(frame_store, "document:summary")
    assert mind_db.scalar("SELECT summary FROM sessions WHERE gop_id = ?", (frame.gop_id,)) is None
    assert (
        mind_db.scalar(
            "SELECT summary FROM sessions WHERE gop_id = ?",
            ("unrelated",),
        )
        == "Keep unrelated summary"
    )


def test_retraction_scrubs_pending_jobs_without_source_event_receipt(mind_db, frame_store):
    from gluomem.service.jobs import JobQueue

    queue = JobQueue(mind_db.backend)
    queue.enqueue(
        frame_store.scope,
        "ingest_event",
        "pending:1",
        {
            "event_id": "pending",
            "source_ref": "document:pending",
            "content": "Pending secret",
        },
    )
    queue.enqueue(
        frame_store.scope,
        "ingest_event",
        "keep:1",
        {
            "event_id": "keep",
            "source_ref": "document:other",
            "content": "Keep this",
        },
    )
    retract_source(frame_store, "document:pending")
    assert (
        mind_db.scalar(
            "SELECT payload FROM memory_jobs WHERE dedup_key = ?",
            ("pending:1",),
        )
        == "{}"
    )
    assert "Keep this" in mind_db.scalar(
        "SELECT payload FROM memory_jobs WHERE dedup_key = ?",
        ("keep:1",),
    )


def test_retraction_clears_summary_when_all_frames_have_shared_support(
    mind_db,
    frame_store,
    cfg,
):
    from gluomem.frames.sessions import end_session

    event = SourceEvent(event_id="shared-a", source_ref="doc:a", content="Shared safe evidence")
    frame = ingest_source_event(event, store=frame_store, cfg=cfg).frames[0]
    corroborated = ingest_source_event(
        event.model_copy(update={"event_id": "shared-b", "source_ref": "doc:b"}),
        store=frame_store,
        cfg=cfg,
    )
    assert corroborated.frames[0].id == frame.id
    end_session(mind_db, frame.gop_id, "May include source-specific private material")
    result = retract_source(frame_store, "doc:a")
    assert result.retained_frame_ids == [frame.id]
    assert frame_store.get_frame(frame.id).content == "Shared safe evidence"
    assert mind_db.scalar("SELECT summary FROM sessions WHERE gop_id = ?", (frame.gop_id,)) is None


@pytest.mark.parametrize("duplicate", [False, True])
def test_retraction_between_prepare_and_commit_blocks_stale_work(
    mind_db,
    frame_store,
    cfg,
    duplicate,
):
    from gluomem.frames.source_fences import SourceRetractedError
    from gluomem.write.events import commit_source_event, prepare_source_event

    event = SourceEvent(event_id="race", source_ref="doc:race", content="Racing secret")
    if duplicate:
        ingest_source_event(event, store=frame_store, cfg=cfg)
    prepared = prepare_source_event(event, store=frame_store, cfg=cfg)
    retract_source(frame_store, event.source_ref)
    with pytest.raises(SourceRetractedError):
        commit_source_event(prepared, store=frame_store)
    assert mind_db.scalar("SELECT COUNT(*) FROM memory_frames WHERE status = 'active'") == 0


@pytest.mark.parametrize(
    "other",
    [
        ("other", "default", "personal"),
        ("local", "other", "personal"),
        ("local", "default", "other"),
    ],
)
def test_source_retraction_is_exactly_scoped(mind_db, frame_store, cfg, other):
    from gluomem.storage.scope import MemoryScope

    retract_source(frame_store, "doc:scoped")
    result = ingest_source_event(
        SourceEvent(
            event_id="scoped",
            source_ref="doc:scoped",
            content="Other scope content",
            scope=MemoryScope(*other),
        ),
        store=frame_store,
        cfg=cfg,
    )
    assert len(result.frames) == 1


def test_source_retraction_precedes_extractor_and_embedder_calls(
    mind_db,
    frame_store,
    cfg,
    monkeypatch,
):
    from gluomem.frames.source_fences import SourceRetractedError
    from gluomem.llm.fake import FakeExtractors

    def forbidden(*args, **kwargs):
        pytest.fail("Removed plaintext reached expensive processing/cache boundary")

    extractors = FakeExtractors()
    monkeypatch.setattr(extractors, "extract_event_claims", forbidden)
    monkeypatch.setattr(frame_store, "_prepare_index", forbidden)
    retract_source(frame_store, "doc:early")
    with pytest.raises(SourceRetractedError):
        ingest_source_event(
            SourceEvent(event_id="early", source_ref="doc:early", content="Must not process"),
            store=frame_store,
            cfg=cfg,
            extractors=extractors,
        )


def test_repeated_retraction_and_scope_forget_preserve_fence(mind_db, frame_store, cfg):
    from gluomem.frames.lifecycle import forget_scope
    from gluomem.frames.source_fences import SourceRetractedError

    retract_source(frame_store, "doc:repeat")
    retract_source(frame_store, "doc:repeat")
    forget_scope(frame_store)
    assert mind_db.scalar("SELECT COUNT(*) FROM source_retractions") == 1
    with pytest.raises(SourceRetractedError):
        ingest_source_event(
            SourceEvent(
                event_id="repeat",
                source_ref="doc:repeat",
                content="New revision",
                revision=9,
                force_reassert=True,
            ),
            store=frame_store,
            cfg=cfg,
        )
