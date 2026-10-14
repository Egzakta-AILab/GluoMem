"""Tests for SourceEvent idempotency, provenance spans, classification, and 90-day episode TTL."""

from __future__ import annotations

from pathlib import Path

import pytest

from gluomem.config.app_config import AppConfig
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.hybrid import HybridSearch
from gluomem.storage.db import MindDB
from gluomem.storage.scope import MemoryScope
from gluomem.write.events import (
    SourceEvent,
    SourceEventMessage,
    classify_memory_kind,
    get_frame_provenance,
    ingest_source_event,
    pin_memory,
    promote_episode_to_fact,
)


def test_classify_memory_kind_bilingual() -> None:
    assert classify_memory_kind("I always prefer dark mode in the editor.") == "preference"
    assert classify_memory_kind("Увек користим ћирилицу за званичне дописе.") == "preference"
    assert classify_memory_kind("We decided to migrate the cluster to PostgreSQL 18.") == "decision"
    assert classify_memory_kind("Одлучили смо да пређемо на нови сервис.") == "decision"
    assert classify_memory_kind("Runbook step 1: drain worker nodes before upgrade.") == "procedure"
    assert (
        classify_memory_kind("Yesterday we had an incident during the release meeting.")
        == "episode"
    )
    assert classify_memory_kind("My role is Principal Engineer at Egzakta.") == "profile"


def test_source_event_idempotency_and_provenance(tmp_path: Path) -> None:
    db = MindDB(tmp_path / "events.mind", vector_dim=32)
    embedder = MockEmbedder(dim=32)
    scope = MemoryScope("egzakta", "mikhail", "personal")
    store = FrameStore(db, embedder=embedder, scope=scope)

    event = SourceEvent(
        event_id="evt-chat-001",
        revision=1,
        source_type="chat_turn",
        source_ref="conv-42",
        scope=scope,
        created_at="2026-03-01T10:00:00",
        messages=[
            SourceEventMessage(
                message_id="msg-1",
                role="user",
                content="We decided to use PostgreSQL 18 with pgvector for GluoMem.",
                timestamp="2026-03-01T09:59:00",
            ),
            SourceEventMessage(
                message_id="msg-2",
                role="assistant",
                content="Understood, I will note that architectural decision.",
            ),
        ],
    )

    res1 = ingest_source_event(event, store=store)
    assert res1.deduplicated is False
    assert len(res1.frames) == 1
    frame = res1.frames[0]
    assert frame.kind == "decision"
    assert frame.status == "active"
    assert frame.scope == scope

    prov = get_frame_provenance(db.backend, frame.id, scope=scope)
    assert len(prov) == 1
    assert prov[0].source_event_id == res1.event_row_id
    assert prov[0].source_type == "chat_turn"
    assert prov[0].source_ref == "conv-42"
    assert prov[0].message_id == "msg-1"
    assert prov[0].verbatim_span == "We decided to use PostgreSQL 18 with pgvector for GluoMem."

    # Re-submitting the identical (event_id, revision) is a no-op returning existing frames
    res2 = ingest_source_event(event, store=store)
    assert res2.deduplicated is True
    assert res2.event_row_id == res1.event_row_id
    assert [f.id for f in res2.frames] == [frame.id]
    assert len(get_frame_provenance(db.backend, frame.id, scope=scope)) == 1

    # Submitting a conflicting payload for the same (event_id, revision) raises ValueError
    conflicting = event.model_copy(
        update={
            "messages": [
                SourceEventMessage(
                    message_id="msg-1",
                    role="user",
                    content="Completely different payload text.",
                )
            ]
        }
    )
    with pytest.raises(ValueError, match="different payload"):
        ingest_source_event(conflicting, store=store)

    db.close()


def test_episode_90_day_ttl_and_pin_or_promote_exemption(tmp_path: Path) -> None:
    cfg = AppConfig()
    db = MindDB(tmp_path / "ttl.mind", vector_dim=32)
    embedder = MockEmbedder(dim=32)
    scope = MemoryScope("egzakta", "mikhail", "personal")
    store = FrameStore(db, embedder=embedder, scope=scope)
    kg = KnowledgeGraph(db, scope=scope)
    search = HybridSearch(db, store, cfg, embedder=embedder, kg=kg, scope=scope)

    ep_event = SourceEvent(
        event_id="evt-ep-1",
        revision=1,
        source_type="chat_turn",
        source_ref="conv-ep",
        scope=scope,
        created_at="2026-01-01T12:00:00",
        content="Yesterday we had a deployment incident on staging node alpha.",
    )
    res = ingest_source_event(ep_event, store=store)
    assert len(res.frames) == 1
    ep_frame = res.frames[0]
    assert ep_frame.kind == "episode"
    assert ep_frame.expires_at == "2026-04-01T12:00:00"

    # Recallable before 90 days (e.g. day 60: 2026-03-02)
    before_res = search.search(
        "staging node alpha incident",
        scope=scope,
        now_iso="2026-03-02T12:00:00",
    )
    assert [h.frame.id for h in before_res.hits] == [ep_frame.id]

    # Excluded once 90 days have elapsed (e.g. day 91: 2026-04-02)
    after_res = search.search(
        "staging node alpha incident",
        scope=scope,
        now_iso="2026-04-02T12:00:00",
    )
    assert after_res.hits == []

    # Pinning the episode clears its TTL so it remains recallable past 90 days
    pinned_frame = pin_memory(store, ep_frame.id, pinned=True, scope=scope)
    assert pinned_frame.pinned is True
    assert pinned_frame.expires_at is None
    pinned_res = search.search(
        "staging node alpha incident",
        scope=scope,
        now_iso="2026-04-02T12:00:00",
    )
    assert [h.frame.id for h in pinned_res.hits] == [ep_frame.id]

    # Unpinning restores the 90-day episode TTL, and promoting to fact clears it permanently
    unpinned = pin_memory(store, ep_frame.id, pinned=False, scope=scope)
    assert unpinned.expires_at == "2026-04-01T12:00:00"
    promoted = promote_episode_to_fact(store, ep_frame.id, new_kind="fact", scope=scope)
    assert promoted.kind == "fact"
    assert promoted.expires_at is None
    promoted_res = search.search(
        "staging node alpha incident",
        scope=scope,
        now_iso="2026-08-01T12:00:00",
    )
    assert [h.frame.id for h in promoted_res.hits] == [ep_frame.id]

    db.close()
