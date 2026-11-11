"""Portable critical contracts; authored inputs and deterministic embeddings only."""

from __future__ import annotations

import pytest

from gluomem.frames.lifecycle import forget_memory, retract_source
from gluomem.frames.store import FrameStore
from gluomem.llm.fake import FakeExtractors
from gluomem.llm.outputs import ConsolidationDecision, ExtractedClaimSpec
from gluomem.service.jobs import JobQueue
from gluomem.service.worker import MemoryWorker
from gluomem.storage.db import MindDB
from gluomem.storage.postgres_backend import PostgresBackend
from gluomem.write import gate
from gluomem.write.events import SourceEvent, ingest_source_event

NOW = "2026-11-02T12:00:00"


def _observer(db: MindDB) -> MindDB:
    backend = db.backend
    if isinstance(backend, PostgresBackend):
        return MindDB(
            dim=db.dim,
            backend_type="postgres",
            dsn=backend.dsn,
            postgres_schema=backend.postgres_schema,
        )
    return MindDB(db.path, dim=db.dim)


@pytest.mark.parametrize("erasure", ["forget", "retract_target", "retract_incoming"])
@pytest.mark.parametrize("max_attempts", [1, 3])
def test_worker_commits_erasure_before_refusal_failure(
    mind_db,
    frame_store,
    cfg,
    monkeypatch,
    erasure,
    max_attempts,
):
    original = ingest_source_event(
        SourceEvent(event_id="old", source_ref="old-source", content="Database port is 5432."),
        store=frame_store,
        cfg=cfg,
    ).frames[0]
    queue = JobQueue(mind_db.backend)
    queued = queue.enqueue(
        dedup_key="correction:1",
        max_attempts=max_attempts,
        now_iso=NOW,
        payload={
            "event_id": "correction",
            "source_ref": "new-source",
            "content": "Database port is 6432.",
        },
    )
    worker = MemoryWorker(
        mind_db,
        frame_store,
        cfg=cfg,
        worker_id="authored-worker",
        queue=queue,
        retry_backoff_seconds=0,
        extractors=FakeExtractors(
            event_claims=[[ExtractedClaimSpec(text="Database port is 6432.")]],
            consolidation_decisions=[
                ConsolidationDecision(
                    action="SUPERSEDE",
                    target_frame_id=original.id,
                )
            ],
            banner=False,
        ),
    )
    admit = gate.admit
    callbacks = []

    def erase_after_admission(*args, **kwargs):
        result = admit(*args, **kwargs)
        if erasure == "forget":
            forget_memory(frame_store, original.id)
        else:
            retract_source(
                frame_store, "old-source" if erasure == "retract_target" else "new-source"
            )
        callbacks.append(True)
        return result

    monkeypatch.setattr(gate, "admit", erase_after_admission)
    fail = queue.fail
    failures = []

    def inspect_committed_erasure(job_id, error, **kwargs):
        assert not mind_db.in_transaction, "queue failure must follow owner commit"
        with _observer(mind_db) as observed:
            if erasure != "retract_incoming":
                erased = FrameStore(observed, frame_store.embedder).get_frame(original.id)
                assert erased.status == "forgotten", "outer worker rollback resurrected erasure"
                assert erased.content == "[forgotten]"
                assert observed.scalar("SELECT COUNT(*) FROM memory_tombstones") > 0
            if erasure != "forget":
                source = "old-source" if erasure == "retract_target" else "new-source"
                assert (
                    observed.scalar(
                        "SELECT COUNT(*) FROM source_retractions WHERE source_ref = ?",
                        (source,),
                    )
                    == 1
                ), "outer worker rollback removed the source fence"
            assert (
                observed.scalar(
                    "SELECT COUNT(*) FROM source_events WHERE event_id = 'correction'",
                )
                == 0
            )
            assert (
                observed.scalar(
                    "SELECT COUNT(*) FROM memory_frames WHERE content = 'Database port is 6432.'",
                )
                == 0
            )
            leased = JobQueue(observed.backend).get_job(job_id)
            assert leased.status == "leased"
            assert leased.attempt_count == kwargs["attempt_count"] == 1
        failures.append(error)
        return fail(job_id, error, **kwargs)

    monkeypatch.setattr(queue, "fail", inspect_committed_erasure)
    result = worker.run_once(now_iso=NOW)
    assert callbacks == [True]
    assert len(failures) == 1
    assert result is not None and result.id == queued.id
    assert result.status == ("dead_letter" if max_attempts == 1 else "queued")
    assert result.attempt_count == 1 and result.max_attempts == max_attempts
    assert result.worker_id is None and result.lease_expires_at is None
    assert result.last_error and "LeaseLostError" not in result.last_error
    assert queue.get_job(queued.id) == result
    assert not mind_db.in_transaction
    if erasure == "retract_incoming":
        assert result.payload == {}
    if erasure != "retract_incoming":
        assert frame_store.get_frame(original.id).status == "forgotten"
        assert original.id not in mind_db.fts.search("5432", 10, scope=frame_store.scope)
        assert original.id not in mind_db.frame_vec.ids()
        assert (
            mind_db.scalar(
                "SELECT COUNT(*) FROM memory_provenance WHERE verbatim_span = ?",
                (original.content,),
            )
            == 0
        )


@pytest.mark.parametrize("action", ["ADD", "NOOP", "SUPERSEDE"])
def test_positive_worker_publication_completes_atomically(
    mind_db,
    frame_store,
    cfg,
    monkeypatch,
    action,
):
    from gluomem.service import worker as worker_module

    old_text = "Database port is 5432."
    original = ingest_source_event(
        SourceEvent(event_id="seed", source_ref="seed-source", content=old_text),
        store=frame_store,
        cfg=cfg,
    ).frames[0]
    text = old_text if action == "NOOP" else "Database port is 6432."
    target = None if action == "ADD" else original.id
    worker = MemoryWorker(
        mind_db,
        frame_store,
        cfg=cfg,
        worker_id="authored-worker",
        extractors=FakeExtractors(
            event_claims=[[ExtractedClaimSpec(text=text)]],
            consolidation_decisions=[ConsolidationDecision(action=action, target_frame_id=target)],
            banner=False,
        ),
    )
    queued = worker.queue.enqueue(
        dedup_key="positive:1",
        now_iso=NOW,
        payload={"event_id": "positive", "source_ref": "positive-source", "content": text},
    )
    attempt = worker_module.attempt_commit_source_event
    outcomes = []

    def capture(*args, **kwargs):
        outcome = attempt(*args, **kwargs)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(worker_module, "attempt_commit_source_event", capture)
    completed = worker.run_once(now_iso=NOW)
    assert completed is not None and completed.id == queued.id and completed.status == "succeeded"
    assert len(outcomes) == 1 and not outcomes[0].refused and outcomes[0].error is None
    receipt = outcomes[0].result
    assert receipt is not None and len(receipt.frames) == 1
    published = receipt.frames[0]
    assert published.content == text and published.status == "active"
    if action == "SUPERSEDE":
        assert published.supersedes == original.id
        retired = frame_store.get_frame(original.id)
        assert retired.status == "superseded" and retired.superseded_by == published.id
        assert retired.content == "[forgotten]"
    elif action == "NOOP":
        assert published.id == original.id
    with _observer(mind_db) as observed:
        assert JobQueue(observed.backend).get_job(queued.id).status == "succeeded"
        assert (
            observed.scalar("SELECT COUNT(*) FROM source_events WHERE event_id = 'positive'") == 1
        )
        assert (
            observed.scalar(
                "SELECT COUNT(*) FROM memory_provenance WHERE source_ref = 'positive-source'",
            )
            == 1
        )


def test_worker_refusal_failure_cannot_overwrite_a_new_lease(
    mind_db,
    frame_store,
    cfg,
    monkeypatch,
):
    original = frame_store.create_i_frame("seed", "Private erasure lease sample.")
    worker = MemoryWorker(
        mind_db, frame_store, cfg=cfg, worker_id="authored-worker", retry_backoff_seconds=0
    )
    queued = worker.queue.enqueue(
        dedup_key="lease-refusal:1",
        now_iso=NOW,
        payload={"event_id": "lease-refusal", "source_ref": "new", "content": "Replacement fact."},
    )
    admit = gate.admit
    fail = worker.queue.fail
    newer = []

    def erase_and_refuse(*args, **kwargs):
        admit(*args, **kwargs)
        forget_memory(frame_store, original.id)
        raise ValueError("authored refusal")

    def replace_lease_before_failure(*args, **kwargs):
        assert not mind_db.in_transaction
        with _observer(mind_db) as observed:
            assert (
                FrameStore(observed, frame_store.embedder).get_frame(original.id).status
                == "forgotten"
            )
            queue = JobQueue(observed.backend)
            queue.reclaim_expired_leases(now_iso="2026-11-02T12:02:00")
            lease = queue.lease_next("new-owner", now_iso="2026-11-02T12:02:00")
            assert lease is not None and lease.attempt_count == 2
            newer.append(lease)
        return fail(*args, **kwargs)

    monkeypatch.setattr(gate, "admit", erase_and_refuse)
    monkeypatch.setattr(worker.queue, "fail", replace_lease_before_failure)
    assert worker.run_once(now_iso=NOW) is None
    assert len(newer) == 1 and worker.queue.get_job(queued.id) == newer[0]
    assert frame_store.get_frame(original.id).status == "forgotten"
    assert (
        mind_db.scalar("SELECT COUNT(*) FROM source_events WHERE event_id = 'lease-refusal'") == 0
    )
