"""Tests for single-pass SourceEvent extraction & ADD/SUPERSEDE/NOOP/CANDIDATE consolidation."""

from __future__ import annotations

from datetime import UTC, datetime

from gluomem.frames.lifecycle import version_lineage
from gluomem.llm import (
    ConsolidationDecision,
    ExtractedClaimSpec,
    FakeExtractors,
    KGEntity,
    KGRelation,
)
from gluomem.write.events import (
    SourceEvent,
    SourceEventMessage,
    get_frame_provenance,
    ingest_source_event,
)


def test_single_pass_source_event_extraction_and_kg_triples(frame_store, kg, cfg) -> None:
    extractors = FakeExtractors(
        event_claims=[
            [
                ExtractedClaimSpec(
                    text="User prefers dark mode in all IDEs.",
                    kind="preference",
                    confidence=0.95,
                    event_date=datetime(2026, 10, 15, tzinfo=UTC),
                    verbatim_span="I always use dark mode in my IDEs.",
                    message_id="m1",
                    entities=[
                        KGEntity(entity_type="person", name="Mikhail"),
                        KGEntity(entity_type="concept", name="dark mode"),
                    ],
                    relations=[
                        KGRelation(
                            source="Mikhail",
                            target="dark mode",
                            relation_type="PREFERS",
                        )
                    ],
                ),
                ExtractedClaimSpec(
                    text="Team decided to deploy PostgreSQL 17 on Friday.",
                    kind="decision",
                    confidence=0.98,
                    event_date=datetime(2026, 10, 16, tzinfo=UTC),
                    verbatim_span="We decided to deploy Postgres 17 on Friday.",
                    message_id="m1",
                ),
            ]
        ],
        banner=False,
    )
    event = SourceEvent(
        event_id="turn-101",
        source_type="chat_turn",
        source_ref="session:abc",
        messages=[
            SourceEventMessage(
                message_id="m1",
                role="user",
                content=(
                    "I always use dark mode in my IDEs. "
                    "We decided to deploy Postgres 17 on Friday."
                ),
            ),
            SourceEventMessage(
                message_id="m2",
                role="assistant",
                content="Got it! I will note your preference and Friday's deployment.",
            ),
        ],
    )
    res = ingest_source_event(event, store=frame_store, kg=kg, cfg=cfg, extractors=extractors)
    assert extractors.calls["extract_event_claims"] == 1
    assert extractors.calls["cognify"] == 0
    assert len(res.frames) == 2
    assert [f.kind for f in res.frames] == ["preference", "decision"]
    assert res.frames[0].event_time == "2026-10-15T00:00:00"
    provs = get_frame_provenance(frame_store._backend, res.frames[0].id)
    assert len(provs) == 1
    assert provs[0].message_id == "m1"
    assert provs[0].verbatim_span == "I always use dark mode in my IDEs."
    assert len(kg.find_valid_entities("Mikhail", cfg.kg)) == 1


def test_consolidation_supersede_updates_trusted_frame_and_scrubs_old(frame_store, cfg) -> None:
    first_ext = FakeExtractors(
        event_claims=[
            [
                ExtractedClaimSpec(
                    text="Primary database port is 5432.",
                    kind="fact",
                    verbatim_span="Primary database port is 5432.",
                )
            ]
        ],
        banner=False,
    )
    r1 = ingest_source_event(
        SourceEvent(
            event_id="evt-1",
            source_type="chat_turn",
            source_ref="session:1",
            content="Primary database port is 5432.",
        ),
        store=frame_store,
        cfg=cfg,
        extractors=first_ext,
    )
    old_frame = r1.frames[0]
    assert old_frame.status == "active"

    second_ext = FakeExtractors(
        event_claims=[
            [
                ExtractedClaimSpec(
                    text="Primary database port is 6432 via PgBouncer.",
                    kind="fact",
                    verbatim_span="We moved the primary database port to 6432 via PgBouncer.",
                )
            ]
        ],
        consolidation_decisions=[
            ConsolidationDecision(
                action="SUPERSEDE",
                target_frame_id=old_frame.id,
                rationale="Port updated from 5432 to 6432.",
            )
        ],
        banner=False,
    )
    r2 = ingest_source_event(
        SourceEvent(
            event_id="evt-2",
            source_type="chat_turn",
            source_ref="session:2",
            content="We moved the primary database port to 6432 via PgBouncer.",
        ),
        store=frame_store,
        cfg=cfg,
        extractors=second_ext,
    )
    new_frame = r2.frames[0]
    assert new_frame.status == "active"
    assert new_frame.supersedes == old_frame.id
    reloaded_old = frame_store.get_frame(old_frame.id)
    assert reloaded_old.status == "superseded"
    assert reloaded_old.superseded_by == new_frame.id
    assert [f.id for f in version_lineage(frame_store, new_frame.id)] == [
        old_frame.id,
        new_frame.id,
    ]


def test_consolidation_holds_untrusted_supersede_of_user_stated_fact_as_candidate(
    frame_store, cfg
) -> None:
    ext = FakeExtractors(
        event_claims=[
            [ExtractedClaimSpec(text="Staging SSH bastion port is 22022.", kind="fact")],
            [ExtractedClaimSpec(text="Staging SSH bastion port is 2222.", kind="fact")],
        ],
        banner=False,
    )
    base = ingest_source_event(
        SourceEvent(
            event_id="trusted-1",
            source_type="chat_turn",
            source_ref="chat:1",
            content="Staging SSH bastion port is 22022.",
        ),
        store=frame_store,
        cfg=cfg,
        extractors=ext,
    ).frames[0]

    ext._consolidation_queue.append(
        ConsolidationDecision(
            action="SUPERSEDE",
            target_frame_id=base.id,
            rationale="Imported document claims a different port.",
        )
    )
    imported = ingest_source_event(
        SourceEvent(
            event_id="import-1",
            source_type="import",
            source_ref="doc:untrusted",
            content="Staging SSH bastion port is 2222.",
        ),
        store=frame_store,
        cfg=cfg,
        extractors=ext,
    ).frames[0]

    # Untrusted source cannot silently overwrite a user_stated frame; held as candidate!
    assert imported.status == "candidate"
    assert frame_store.get_frame(base.id).status == "active"


def test_consolidation_noop_corroborates_existing_frame_provenance(frame_store, cfg) -> None:
    ext = FakeExtractors(
        event_claims=[
            [ExtractedClaimSpec(text="Service runs on Python 3.12.", kind="fact")],
            [ExtractedClaimSpec(text="The service runs on Python 3.12.", kind="fact")],
        ],
        banner=False,
    )
    first = ingest_source_event(
        SourceEvent(
            event_id="e1",
            source_type="chat_turn",
            source_ref="session:a",
            content="Service runs on Python 3.12.",
        ),
        store=frame_store,
        cfg=cfg,
        extractors=ext,
    ).frames[0]

    ext._consolidation_queue.append(
        ConsolidationDecision(
            action="NOOP",
            target_frame_id=first.id,
            rationale="Already recorded.",
        )
    )
    second = ingest_source_event(
        SourceEvent(
            event_id="e2",
            source_type="chat_turn",
            source_ref="session:b",
            content="The service runs on Python 3.12.",
        ),
        store=frame_store,
        cfg=cfg,
        extractors=ext,
    )
    assert [f.id for f in second.frames] == [first.id]
    assert frame_store.count_frames() == 1
    provs = get_frame_provenance(frame_store._backend, first.id)
    assert sorted(p.source_ref for p in provs) == ["session:a", "session:b"]
