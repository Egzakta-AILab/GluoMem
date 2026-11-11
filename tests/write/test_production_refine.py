"""Portable critical contracts; authored inputs and deterministic embeddings only."""

import pytest

from gluomem.client.sdk import EmbeddedMemoryClient
from gluomem.llm.fake import FakeExtractors
from gluomem.llm.outputs import ConsolidationDecision, ExtractedClaimSpec
from gluomem.search.hybrid import HybridSearch
from gluomem.write.events import (
    SourceEvent,
    SourceEventMessage,
    commit_source_event,
    ingest_source_event,
    prepare_source_event,
)

ROOT_TEXT = "Riley owns the release review."

DETAIL = "Riley's release review shortlist excludes weekend deployments."


def root_event(store, cfg, *, event_id="root", source_ref="chat:review", text=ROOT_TEXT):
    return ingest_source_event(
        SourceEvent(event_id=event_id, source_ref=source_ref, content=text, kind="fact"),
        store=store,
        cfg=cfg,
    ).frames[0]


def prepare_refinement(store, cfg, decision, *, event_id="refinement", **event_options):
    source = "For Riley's release review, the shortlist excludes weekend deployments."
    return prepare_source_event(
        SourceEvent(
            event_id=event_id,
            source_ref="chat:later-session",
            capture_policy="trusted-only",
            messages=[SourceEventMessage(message_id="detail-msg", role="user", content=source)],
            **event_options,
        ),
        store=store,
        cfg=cfg,
        extractors=FakeExtractors(
            event_claims=[
                [
                    ExtractedClaimSpec(
                        text=DETAIL,
                        message_id="detail-msg",
                        verbatim_span=source,
                    )
                ]
            ],
            consolidation_decisions=[decision],
            banner=False,
        ),
    )


def test_actual_source_writer_refines_observed_older_i_in_its_gop_then_usable_recall(
    frame_store,
    mind_db,
    cfg,
    monkeypatch,
):
    cfg = cfg.with_overrides({"search.abstain.enabled": False})
    original = root_event(frame_store, cfg)
    unrelated = root_event(
        frame_store, cfg, event_id="other", text="Casey owns the release budget."
    )
    # Bypass only the facade parser so RED reaches the production writer boundary.
    plan = prepare_refinement(
        frame_store,
        cfg,
        ConsolidationDecision.model_construct(action="REFINE", target_frame_id=original.id),
    )
    assert original.id in plan.candidate_revisions[0]
    with monkeypatch.context() as patch:

        def no_embedding(*args, **kwargs):
            raise AssertionError("Embedding work must stay outside SourceEvent commit")

        patch.setattr(frame_store.embedder, "embed", no_embedding)
        patch.setattr(frame_store.embedder, "embed_batch", no_embedding)
        result = commit_source_event(plan, store=frame_store)
    delta = frame_store.get_frame(result.frames[0].id)
    assert delta.frame_type == "P"
    assert delta.base_frame_id == original.id
    assert delta.gop_id == original.gop_id == unrelated.gop_id
    assert delta.id not in {original.id, unrelated.id}
    assert delta.content == DETAIL and delta.status == "active"
    assert frame_store.get_frame(original.id) == original
    assert frame_store.get_frame(unrelated.id) == unrelated
    assert [(p.frame_id, p.message_id, p.source_ref) for p in result.provenance] == [
        (delta.id, "detail-msg", "chat:later-session")
    ]
    search = HybridSearch(mind_db, frame_store, cfg, embedder=frame_store.embedder)
    monkeypatch.setattr(mind_db.chunk_vec, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(mind_db.frame_vec, "search", lambda *args, **kwargs: [])
    client = EmbeddedMemoryClient(mind_db, frame_store, cfg=cfg, search=search)
    recalled = client.recall("shortlist", k=1)
    assert not recalled.abstained
    assert [hit.frame.id for hit in recalled.hits] == [original.id, delta.id]
    assert recalled.groups[0].frame_ids == [original.id, delta.id]
    assert recalled.groups[0].seed_ids == [delta.id]
    assert recalled.groups[0].root_frame_id == original.id
    assert [
        (link.frame_id, link.related_frame_id, link.relation) for link in recalled.groups[0].links
    ] == [(delta.id, original.id, "refines")]
    assert ROOT_TEXT in recalled.context and DETAIL in recalled.context
    assert '"frame_type":"P"' in recalled.context
    assert "Casey" not in recalled.context


@pytest.mark.parametrize("erase", ["forget", "retract"])
def test_refine_reentrant_base_erasure_stays_authoritative(frame_store, cfg, monkeypatch, erase):
    from gluomem.frames.lifecycle import forget_memory, retract_source
    from gluomem.write import gate

    original = root_event(frame_store, cfg)
    plan = prepare_refinement(
        frame_store, cfg, ConsolidationDecision(action="REFINE", target_frame_id=original.id)
    )
    admit = gate.admit
    calls = []

    def erasing(*args, **kwargs):
        result = admit(*args, **kwargs)
        calls.append(True)
        if erase == "forget":
            forget_memory(frame_store, original.id)
        else:
            retract_source(frame_store, "chat:review")
        return result

    monkeypatch.setattr(gate, "admit", erasing)
    with pytest.raises(ValueError, match="Stale consolidation target"):
        commit_source_event(plan, store=frame_store)
    assert calls == [True]
    erased = frame_store.get_frame(original.id)
    assert erased.status == "forgotten" and erased.content == "[forgotten]"
    assert (
        frame_store._backend.scalar("SELECT COUNT(*) FROM memory_frames WHERE status='active'") == 0
    )
    assert frame_store._backend.scalar("SELECT COUNT(*) FROM memory_frame_chunks") == 0
    assert (
        frame_store._backend.scalar(
            "SELECT COUNT(*) FROM source_events WHERE event_id='refinement'"
        )
        == 0
    )
