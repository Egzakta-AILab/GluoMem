"""Portable critical contracts; authored inputs and deterministic embeddings only."""

from datetime import UTC, datetime

from gluomem.client.sdk import EmbeddedMemoryClient
from gluomem.llm import ConsolidationDecision, ExtractedClaimSpec, FakeExtractors


def test_writer_correction_recall_has_current_lineage_and_trusted_source(mind_db, frame_store, cfg):
    cfg = cfg.with_overrides({"search.abstain.enabled": False})
    client = EmbeddedMemoryClient(mind_db, frame_store, cfg=cfg)
    old = client.ingest_event(
        event_id="original",
        source_ref="chat:original",
        kind="fact",
        messages=[
            {"message_id": "old-msg", "role": "user", "content": "The launch owner is Morgan."}
        ],
    ).frames[0]
    source = "Correction: the launch owner is Avery, not Morgan."
    client.extractors = FakeExtractors(
        event_claims=[
            [
                ExtractedClaimSpec(
                    text="The launch owner is Avery.",
                    kind="fact",
                    event_date=datetime(2026, 11, 5, 12, tzinfo=UTC),
                    message_id="new-msg",
                    verbatim_span=source,
                )
            ]
        ],
        consolidation_decisions=[
            ConsolidationDecision(
                action="SUPERSEDE",
                target_frame_id=old.id,
            )
        ],
        banner=False,
    )
    replacement = client.ingest_event(
        event_id="correction",
        source_ref="chat:correction",
        created_at="2026-10-03T09:00:00Z",
        event_time="2026-11-05T12:00:00Z",
        messages=[
            {
                "message_id": "new-msg",
                "role": "user",
                "content": source,
                "timestamp": "2026-10-03T08:59:00Z",
            }
        ],
    ).frames[0]
    assert replacement.supersedes == old.id
    result = client.recall("launch owner", k=1)
    assert [hit.frame.id for hit in result.hits] == [replacement.id]
    assert len(result.groups) == 1
    group = result.groups[0]
    assert group.frame_ids == [replacement.id]
    assert group.seed_ids == [replacement.id]
    assert [(link.frame_id, link.related_frame_id, link.relation) for link in group.links] == [
        (replacement.id, old.id, "supersedes")
    ]
    provenance = result.hits[0].frame.provenance[0]
    assert provenance.message_id == "new-msg"
    assert provenance.message_role == "user"
    assert provenance.message_time == "2026-10-03T08:59:00Z"
    assert provenance.observed_at == "2026-10-03T09:00:00"
    assert result.hits[0].frame.event_time == "2026-11-05T12:00:00"
    assert '"event_time":"2026-11-05T12:00:00"' in result.context
    assert provenance.verbatim_span == source
    assert '"message_role":"user"' in result.context
    assert '"supersedes"' in result.context
    assert "The launch owner is Morgan." not in result.context


def test_refinement_seed_returns_its_original_base_as_one_group(mind_db, frame_store, kg, cfg):
    from datetime import datetime

    from gluomem.llm import CognifyResult, ExtractedFact, KGEntity
    from gluomem.search.hybrid import HybridSearch
    from gluomem.write.conversation import Conversation, Session, Turn, ingest_conversation
    from gluomem.write.gate import InjectionScanner

    cfg = cfg.with_overrides(
        {
            "search.abstain.enabled": False,
            "write.frame_model": "ipb",
            "write.raw_turns.enabled": False,
            "write.lanes.event.enabled": False,
            "write.lanes.profile.enabled": False,
            "write.subject.min_entity_overlap": 1,
            "write.parallelism": 1,
        }
    )
    root_text = "Riley owns the release review."
    delta_text = "Their shortlist excludes weekend deployments."
    entity = KGEntity(entity_type="person", name="Riley")
    facade = FakeExtractors(
        facts=[[ExtractedFact(text=root_text)], [ExtractedFact(text=delta_text)]],
        cognify_results=[CognifyResult(entities=[entity]), CognifyResult(entities=[entity])],
        banner=False,
    )
    report = ingest_conversation(
        Conversation(
            "authored",
            ("Riley", "Casey"),
            [
                Session(i, datetime(2026, 10, i), [Turn("Riley", text)])
                for i, text in enumerate((root_text, delta_text), 1)
            ],
        ),
        facade=facade,
        scanner=InjectionScanner(cfg.safety),
        store=frame_store,
        kg=kg,
        cfg=cfg,
    )
    assert (report.i_frames, report.p_frames) == (1, 1)
    root, delta = frame_store.all_frames()
    assert delta.base_frame_id == root.id
    search = HybridSearch(mind_db, frame_store, cfg)
    search._vec_lane = lambda *args, **kwargs: []
    assert [hit.frame.id for hit in search.search("shortlist", limit=1).hits] == [delta.id]
    client = EmbeddedMemoryClient(mind_db, frame_store, cfg=cfg, search=search)
    result = client.recall("shortlist", k=1)
    assert not result.abstained
    assert [hit.frame.id for hit in result.hits] == [root.id, delta.id]
    assert result.groups[0].frame_ids == [root.id, delta.id]
    assert result.groups[0].seed_ids == [delta.id]
    assert result.groups[0].root_frame_id == root.id
    assert [
        (link.frame_id, link.related_frame_id, link.relation) for link in result.groups[0].links
    ] == [(delta.id, root.id, "refines")]
    assert len(result.context.splitlines()) == 1
    assert root_text in result.context and delta_text in result.context


def test_b_replacing_p_keeps_original_base_relationship_without_old_plaintext(
    mind_db, frame_store, cfg, monkeypatch
):
    from gluomem.frames.lifecycle import _scrub_frames
    from gluomem.search.hybrid import HybridSearch

    cfg = cfg.with_overrides({"search.abstain.enabled": False})
    root = frame_store.create_i_frame("topic", "Riley owns the release review.")
    delta = frame_store.create_p_frame("topic", "Only on weekdays.", root.id)
    correction = frame_store.create_b_frame(
        "topic",
        "Currentword only on Tuesdays.",
        delta.id,
        [delta.id],
        supersedes=delta.id,
    )
    with mind_db.transaction():
        frame_store._record_lineage(correction.id, delta, delta.content_hash, "correct")
        _scrub_frames(frame_store, [delta], scope=frame_store.scope, status="superseded")
        frame_store._update(delta.id, frame_store.scope, superseded_by=correction.id)
    search = HybridSearch(mind_db, frame_store, cfg)
    monkeypatch.setattr(mind_db.chunk_vec, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(mind_db.frame_vec, "search", lambda *args, **kwargs: [])
    result = search.recall("Currentword", limit=1)
    assert {hit.frame.id for hit in result.hits} == {root.id, correction.id}
    assert (delta.id, root.id, "refines") in [
        (link.frame_id, link.related_frame_id, link.relation) for link in result.groups[0].links
    ]
    assert "Only on weekdays." not in result.context
