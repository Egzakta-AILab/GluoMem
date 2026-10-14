"""Tests for candidate memory review queue (accept, edit, dismiss) and dismissal tombstones."""

from __future__ import annotations

from pathlib import Path

from gluomem.config.app_config import AppConfig
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.hybrid import HybridSearch
from gluomem.storage.db import MindDB
from gluomem.storage.scope import MemoryScope
from gluomem.write.events import (
    SourceEvent,
    ingest_source_event,
    list_review_queue,
    review_memory,
)


def test_review_queue_accept_edit_and_dismiss(tmp_path: Path) -> None:
    cfg = AppConfig()
    db = MindDB(tmp_path / "review.mind", vector_dim=32)
    embedder = MockEmbedder(dim=32)
    scope = MemoryScope("egzakta", "mikhail", "personal")
    store = FrameStore(db, embedder=embedder, scope=scope)
    kg = KnowledgeGraph(db, scope=scope)
    search = HybridSearch(db, store, cfg, embedder=embedder, kg=kg, scope=scope)

    # 1. Low-confidence inferred memory enters status='candidate' and is excluded from recall
    cand_event = SourceEvent(
        event_id="evt-cand-1",
        revision=1,
        source_type="chat_turn",
        source_ref="conv-100",
        scope=scope,
        confidence=0.55,
        content="User might prefer weekly digest emails on Friday mornings.",
    )
    ingested = ingest_source_event(cand_event, store=store, review_threshold=0.75)
    assert len(ingested.frames) == 1
    cand = ingested.frames[0]
    assert cand.status == "candidate"
    assert cand.verified_by_user is False

    # Candidate is listed in review queue but NOT returned by HybridSearch
    queue = list_review_queue(store, scope=scope)
    assert [f.id for f in queue] == [cand.id]
    assert search.search("weekly digest emails Friday", scope=scope).hits == []

    # 2. Accepting promotes candidate to active and verified_by_user=True
    accepted = review_memory(store, cand.id, action="accept", scope=scope)
    assert accepted.status == "active"
    assert accepted.verified_by_user is True
    assert list_review_queue(store, scope=scope) == []
    assert [
        h.frame.id for h in search.search("weekly digest emails Friday", scope=scope).hits
    ] == [cand.id]

    # 3. Editing updates content, re-indexes FTS/vectors, and marks active + verified
    cand_event_2 = SourceEvent(
        event_id="evt-cand-2",
        revision=1,
        source_type="chat_turn",
        source_ref="conv-101",
        scope=scope,
        require_review=True,
        content="Deployments happen on Tuesdays at noon.",
    )
    cand2 = ingest_source_event(cand_event_2, store=store).frames[0]
    assert cand2.status == "candidate"

    edited = review_memory(
        store,
        cand2.id,
        action="edit",
        edited_content="Deployments happen on Wednesdays at 14:00 UTC.",
        scope=scope,
    )
    assert edited.status == "active"
    assert edited.verified_by_user is True
    assert edited.content == "Deployments happen on Wednesdays at 14:00 UTC."
    wed_hits = search.search("Wednesdays 14:00 UTC", scope=scope).hits
    assert wed_hits and wed_hits[0].frame.id == cand2.id
    assert db.fts.search('"Tuesdays"', 10, scope=scope) == []
    assert db.fts.search('"Wednesdays"', 10, scope=scope) == [cand2.id]

    # 4. Dismissing writes a SuppressionTombstone(reason='dismissed') keyed to event_id
    cand_event_3 = SourceEvent(
        event_id="evt-cand-3",
        revision=1,
        source_type="chat_turn",
        source_ref="conv-102",
        scope=scope,
        require_review=True,
        content="User prefers tabs over spaces in Python.",
    )
    cand3 = ingest_source_event(cand_event_3, store=store).frames[0]
    dismissed = review_memory(store, cand3.id, action="dismiss", scope=scope)
    assert dismissed.status == "tombstoned"
    assert all(
        h.frame.id != cand3.id
        for h in search.search("tabs over spaces in Python", scope=scope).hits
    )

    # Re-running extraction on a new revision of the SAME event_id is suppressed
    cand_event_3_rev2 = cand_event_3.model_copy(update={"revision": 2})
    res_rev2 = ingest_source_event(cand_event_3_rev2, store=store)
    assert res_rev2.frames == []
    assert len(res_rev2.suppressed_hashes) == 1

    # However, a NEW independent source_event_id with fresh evidence CAN propose it again
    fresh_event = SourceEvent(
        event_id="evt-cand-4-fresh",
        revision=1,
        source_type="chat_turn",
        source_ref="conv-200",
        scope=scope,
        require_review=True,
        content="User prefers tabs over spaces in Python.",
    )
    res_fresh = ingest_source_event(fresh_event, store=store)
    assert len(res_fresh.frames) == 1
    assert res_fresh.frames[0].status == "candidate"

    db.close()
