"""Scoped lifecycle, crash recovery and replay contracts."""

from __future__ import annotations

from typing import Any

import pytest

from gluomem.frames.content_hash import content_hash
from gluomem.frames.lifecycle import (
    correct_memory,
    current_scope_generation,
    export_memories,
    forget_memory,
    forget_scope,
    retract_source,
    version_lineage,
)
from gluomem.frames.store import FrameStore
from gluomem.storage.scope import MemoryScope
from gluomem.write.events import (
    EventConflictError,
    SourceEvent,
    SourceEventMessage,
    get_frame_provenance,
    ingest_source_event,
    list_review_queue,
    pin_memory,
    promote_episode_to_fact,
    review_memory,
)


def _evt(text: str, event_id: str = "evt", **kwargs: Any) -> SourceEvent:
    return SourceEvent(
        event_id=event_id, source_ref=kwargs.pop("source_ref", "doc"), content=text, **kwargs
    )


def test_scopeless_delivery_is_scoped_idempotent_and_conflicts(frame_store, mind_db, mock_embedder):
    scope = MemoryScope("t2", "p2", "n2")
    store = FrameStore(mind_db, mock_embedder, scope=scope)
    event = _evt("PostgreSQL is the database.")
    results = [ingest_source_event(event, store=store) for _ in range(3)]
    frame = results[0].frames[0]
    assert frame.scope == scope
    assert [r.deduplicated for r in results] == [False, True, True]
    assert {r.frames[0].id for r in results} == {frame.id}
    assert store.count_frames() == 1 and frame_store.count_frames() == 0
    assert len(get_frame_provenance(mind_db.backend, frame.id, scope=scope)) == 1
    with pytest.raises(EventConflictError, match="different payload"):
        ingest_source_event(event.model_copy(update={"content": "Different."}), store=store)


def test_claims_embed_before_transaction_and_crash_retries(frame_store, mind_db, cfg, monkeypatch):
    import gluomem.write.gate as gate

    original = gate.admit
    prepare = frame_store._prepare_index
    scanner = gate.make_scanner(cfg)
    raw_scan = scanner.scan
    scan_calls = 0
    calls = 0

    def outside_transaction(text):
        assert not mind_db.in_transaction
        return prepare(text)

    def scan_outside_transaction(content: str, *, context: str = ""):
        nonlocal scan_calls
        assert not mind_db.in_transaction
        scan_calls += 1
        return raw_scan(content, context=context)

    def crash_on_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs.get("verdict") is not None
        if calls == 2:
            raise RuntimeError("crash")
        return original(*args, **kwargs)

    monkeypatch.setattr(frame_store, "_prepare_index", outside_transaction)
    monkeypatch.setattr(scanner, "scan", scan_outside_transaction)
    monkeypatch.setattr(gate, "admit", crash_on_second)
    event = SourceEvent(
        event_id="evt",
        source_ref="doc",
        messages=[
            SourceEventMessage(message_id=str(i), content=f"Distinct claim {i}.") for i in range(2)
        ],
    )
    with pytest.raises(RuntimeError, match="crash"):
        ingest_source_event(event, store=frame_store, cfg=cfg, scanner=scanner)
    assert scan_calls == 2
    assert frame_store.count_frames() == 0
    assert mind_db.scalar("SELECT COUNT(*) FROM source_events") == 0
    assert mind_db.scalar("SELECT COUNT(*) FROM memory_provenance") == 0
    assert mind_db.frame_vec.ids() == set()

    def guard():
        assert mind_db.in_transaction
        assert mind_db.scalar("SELECT COUNT(*) FROM source_events") == 1
        raise RuntimeError("fenced")

    with pytest.raises(RuntimeError, match="fenced"):
        ingest_source_event(event, store=frame_store, cfg=cfg, scanner=scanner, write_guard=guard)
    assert scan_calls == 4
    assert mind_db.scalar("SELECT COUNT(*) FROM source_events") == 0
    retry = ingest_source_event(event, store=frame_store, cfg=cfg, scanner=scanner)
    assert scan_calls == 6
    assert not retry.deduplicated and len(retry.frames) == 2


def test_forget_scrubs_payloads_kg_and_repair(frame_store, mind_db, kg, cfg, assert_no_recall):
    from gluomem.service.jobs import JobQueue

    text = "Kraken vault passcode ALPHA."
    event = _evt(text)
    frame = ingest_source_event(event, store=frame_store, cfg=cfg).frames[0]
    JobQueue(mind_db.backend).enqueue(dedup_key="evt:1", payload=event.model_dump(mode="json"))
    entity = kg.create_entity("secret", "ALPHA")
    other = kg.create_entity("project", "Kraken")
    kg.link_entity_frame(entity.id, frame.id)
    kg.link_entity_frame(other.id, frame.id)
    relation = kg.create_relation(other.id, entity.id, "uses")
    kg.link_relation_frame(relation.id, frame.id)
    result = forget_memory(frame_store, frame.id)
    assert set(result.retired_entity_ids) == {entity.id, other.id}
    assert result.retired_relation_ids == [relation.id]
    assert result.scope_generation == 1
    assert frame_store.repair_indexes() == []
    assert not frame_store.ensure_indexed(frame)
    assert not frame_store.ensure_indexed(frame_store.get_frame(frame.id))
    assert_no_recall(text, db=mind_db, store=frame_store, kg=kg)
    assert_no_recall("ALPHA", db=mind_db, store=frame_store, kg=kg)
    replay = ingest_source_event(event, store=frame_store, cfg=cfg)
    assert replay.deduplicated and replay.frames == []


@pytest.mark.parametrize(
    "variant", ["the secret is alpha", "THE SECRET IS ALPHA.", "The secret is ALPHA!"]
)
def test_tombstone_case_punctuation_suppression(
    variant, frame_store, cfg, mind_db, kg, assert_no_recall
):
    original = "The secret is ALPHA."
    frame = ingest_source_event(_evt(original), store=frame_store, cfg=cfg).frames[0]
    forget_memory(frame_store, frame.id)
    from gluomem.service.jobs import JobQueue

    JobQueue(mind_db.backend).enqueue(
        dedup_key="replay:1", payload=_evt(variant, "replay").model_dump(mode="json")
    )
    replay = ingest_source_event(_evt(variant, "replay"), store=frame_store, cfg=cfg)
    assert replay.frames == [] and replay.suppressed_hashes == [content_hash(variant)]
    assert_no_recall(variant, db=mind_db, store=frame_store, kg=kg)


def test_correct_scrubs_old_value_and_forget_scrubs_chain(
    frame_store, mind_db, kg, cfg, assert_no_recall
):
    event = _evt("Release secret Thursday.")
    old = ingest_source_event(event, store=frame_store, cfg=cfg).frames[0]
    with pytest.raises(ValueError, match="normalized content"):
        correct_memory(frame_store, old.id, "  Release  secret Thursday.  ")
    duplicate = frame_store.create_i_frame(old.gop_id, "Release secret Friday.")
    correction = correct_memory(frame_store, old.id, duplicate.content)
    new = correction.replacement_frame
    assert new.id != duplicate.id and new.supersedes == old.id
    assert correction.superseded_frame.superseded_by == new.id
    assert new.verified_by_user is True
    assert frame_store.lineage(new.id)[0].operation == "correct"
    assert_no_recall("Thursday", db=mind_db, store=frame_store, kg=kg)
    assert ingest_source_event(event, store=frame_store, cfg=cfg).frames == []
    assert mind_db.scalar("SELECT COUNT(*) FROM memory_tombstones") == 0
    with pytest.raises(ValueError, match="active or candidate"):
        correct_memory(frame_store, old.id, "Fork.")
    latest = correct_memory(frame_store, new.id, "Release secret Monday.").replacement_frame
    child = frame_store.create_p_frame(latest.gop_id, "Private delta.", latest.id)
    assert [f.id for f in version_lineage(frame_store, latest.id)] == [old.id, new.id, latest.id]
    assert export_memories(frame_store)["by_kind"]["fact"] == [duplicate.id, latest.id, child.id]
    forget_memory(frame_store, latest.id)
    assert {frame_store.get_frame(i).status for i in (old.id, new.id, latest.id, child.id)} == {
        "forgotten"
    }
    assert_no_recall("Monday", db=mind_db, store=frame_store, kg=kg)
    assert_no_recall("Private delta", db=mind_db, store=frame_store, kg=kg)
    replay = ingest_source_event(
        _evt("release secret thursday", "ancestor"), store=frame_store, cfg=cfg
    )
    assert replay.frames == []


def test_correction_allows_fresh_evidence_of_old_value(frame_store, cfg):
    event = _evt("Old observation.")
    old = ingest_source_event(event, store=frame_store, cfg=cfg).frames[0]
    correct_memory(frame_store, old.id, "New observation.")
    assert ingest_source_event(event, store=frame_store, cfg=cfg).frames == []
    assert (
        ingest_source_event(_evt(old.content, "fresh"), store=frame_store, cfg=cfg)
        .frames[0]
        .content
        == old.content
    )


def test_correction_failure_rolls_back(frame_store, mind_db, monkeypatch):
    old = frame_store.create_i_frame("gop", "Old value.")
    original = frame_store._record_lineage

    def crash(*args, **kwargs):
        raise RuntimeError("crash")

    monkeypatch.setattr(frame_store, "_record_lineage", crash)
    with pytest.raises(RuntimeError, match="crash"):
        correct_memory(frame_store, old.id, "New value.")
    assert frame_store.count_frames() == 1
    assert frame_store.get_frame(old.id).status == "active"
    assert frame_store.index_state(old.id).ready
    assert current_scope_generation(mind_db.backend) == 0
    monkeypatch.setattr(frame_store, "_record_lineage", original)
    assert correct_memory(frame_store, old.id, "New value.").scope_generation == 1


def test_shared_frame_retracts_source_without_resurrection(
    frame_store, mind_db, kg, cfg, assert_no_recall
):
    sole_event = _evt("Sole private Aurora.", "sole", source_ref="doc A")
    sole = ingest_source_event(sole_event, store=frame_store, cfg=cfg).frames[0]
    shared_event = _evt("Python is shared.", "shared", source_ref="doc A")
    shared = ingest_source_event(shared_event, store=frame_store, cfg=cfg).frames[0]
    corroborated = ingest_source_event(
        _evt(shared.content, "other", source_ref="doc_A"), store=frame_store, cfg=cfg
    )
    assert corroborated.frames[0].id == shared.id
    collision = ingest_source_event(
        _evt("Another value.", "distinct", source_ref="doc_A"), store=frame_store, cfg=cfg
    ).frames[0]
    assert collision.gop_id != sole.gop_id
    result = retract_source(frame_store, "doc A")
    assert result.forgotten_frame_ids == [sole.id]
    assert result.retained_frame_ids == [shared.id]
    assert result.scope_generation == 1
    assert_no_recall("Aurora", db=mind_db, store=frame_store, kg=kg)
    with pytest.raises(ValueError, match="retracted"):
        ingest_source_event(sole_event, store=frame_store, cfg=cfg)
    assert [p.source_ref for p in get_frame_provenance(mind_db.backend, shared.id)] == ["doc_A"]


def test_retraction_failure_restores_provenance_receipt_and_generation(
    frame_store, mind_db, cfg, monkeypatch
):
    import gluomem.frames.lifecycle as lifecycle

    event = _evt("Private value.")
    frame = ingest_source_event(event, store=frame_store, cfg=cfg).frames[0]

    def crash(*args, **kwargs):
        raise RuntimeError("crash")

    monkeypatch.setattr(lifecycle, "_scrub_frames", crash)
    with pytest.raises(RuntimeError, match="crash"):
        retract_source(frame_store, event.source_ref)
    assert frame_store.get_frame(frame.id).status == "active"
    assert get_frame_provenance(mind_db.backend, frame.id)[0].verbatim_span == frame.content
    assert frame.content in str(mind_db.scalar("SELECT payload_json FROM source_events"))
    assert current_scope_generation(mind_db.backend) == 0


def test_forget_retires_relation_with_independently_supported_entities(frame_store, kg):
    removed = frame_store.create_i_frame("gop", "Alice lives in Paris.")
    retained = frame_store.create_i_frame("gop", "Alice visited Paris.")
    alice = kg.create_entity("person", "Alice")
    paris = kg.create_entity("city", "Paris")
    for frame in (removed, retained):
        for entity in (alice, paris):
            kg.link_entity_frame(entity.id, frame.id)
    relation = kg.create_relation(alice.id, paris.id, "lives_in")
    kg.link_relation_frame(relation.id, removed.id)
    result = forget_memory(frame_store, removed.id)
    assert result.retired_entity_ids == []
    assert result.retired_relation_ids == [relation.id]
    assert kg.get_relation(relation.id).valid_to is not None
    assert kg.get_entity(alice.id).name == "Alice"
    assert kg.frame_ids_for_entity(alice.id) == [retained.id]


@pytest.mark.parametrize("status", ["active", "superseded", "tombstoned", "forgotten", "expired"])
@pytest.mark.parametrize("action", ["accept", "edit"])
def test_review_never_resurrects_non_candidates(status, action, frame_store):
    frame = frame_store.create_i_frame("gop", "Not a candidate.", status=status)
    with pytest.raises(ValueError, match="candidate"):
        review_memory(frame_store, frame.id, action=action, edited_content="Resurrection.")
    assert frame_store.get_frame(frame.id).status == status


def test_review_requires_candidate_and_bumps_generation(
    frame_store, mind_db, cfg, kg, assert_no_recall
):
    candidate = ingest_source_event(
        _evt("Candidate preference.", confidence=0.1), store=frame_store, cfg=cfg
    ).frames[0]
    assert [f.id for f in list_review_queue(frame_store)] == [candidate.id]
    accepted = review_memory(frame_store, candidate.id, action="accept")
    assert accepted.status == "active" and accepted.verified_by_user is True
    assert current_scope_generation(mind_db.backend) == 1
    for action in ("accept", "edit"):
        with pytest.raises(ValueError, match="candidate"):
            review_memory(frame_store, candidate.id, action=action, edited_content="Resurrection.")
    dismissed_event = _evt("Dismiss secret.", "dismiss", require_review=True)
    dismissed = ingest_source_event(dismissed_event, store=frame_store, cfg=cfg).frames[0]
    assert review_memory(frame_store, dismissed.id, action="dismiss").status == "tombstoned"
    assert_no_recall("Dismiss secret", db=mind_db, store=frame_store, kg=kg)
    revision = ingest_source_event(
        dismissed_event.model_copy(update={"revision": 2}), store=frame_store, cfg=cfg
    )
    assert revision.frames == []
    assert (
        ingest_source_event(
            _evt("Dismiss secret.", "fresh", require_review=True), store=frame_store, cfg=cfg
        )
        .frames[0]
        .status
        == "candidate"
    )


def test_review_edit_checks_suppression_and_reindexes(frame_store, mind_db, cfg):
    forgotten = ingest_source_event(_evt("Blocked ALPHA."), store=frame_store, cfg=cfg).frames[0]
    forget_memory(frame_store, forgotten.id)
    candidate = ingest_source_event(
        _evt("Draft Tuesday.", "candidate", require_review=True), store=frame_store, cfg=cfg
    ).frames[0]
    with pytest.raises(ValueError, match="suppressed"):
        review_memory(frame_store, candidate.id, action="edit", edited_content="blocked alpha")
    edited = review_memory(
        frame_store, candidate.id, action="edit", edited_content="Published Wednesday."
    )
    assert edited.status == "active" and frame_store.index_state(edited.id).ready
    assert mind_db.fts.search('"Tuesday"', 10) == []
    assert current_scope_generation(mind_db.backend) == 2


@pytest.mark.parametrize("pinned", [False, True])
def test_explicit_or_pinned_delivery_promotes_candidate(pinned, frame_store, cfg):
    candidate = ingest_source_event(
        _evt("Candidate fact.", require_review=True), store=frame_store, cfg=cfg
    ).frames[0]
    event = _evt(
        candidate.content,
        "explicit",
        pinned=pinned,
        source_type="chat_turn" if pinned else "explicit_memory",
    )
    promoted = ingest_source_event(event, store=frame_store, cfg=cfg).frames[0]
    assert promoted.id == candidate.id and promoted.status == "active"
    assert promoted.pinned if pinned else promoted.verified_by_user


def test_configured_episode_ttl_pin_and_promote(frame_store, mind_db, mock_embedder, cfg):
    cfg = cfg.with_overrides({"lifecycle.episode_ttl_days": 7, "lifecycle.review_threshold": 0.9})
    store = FrameStore(mind_db, mock_embedder, lifecycle=cfg.lifecycle)
    frame = ingest_source_event(
        _evt(
            "Yesterday an incident happened.",
            created_at="2026-01-01T12:00:00",
            confidence=0.8,
            kind="episode",
        ),
        store=store,
        cfg=cfg,
    ).frames[0]
    assert frame.status == "candidate" and frame.expires_at == "2026-01-08T12:00:00"
    assert pin_memory(store, frame.id).expires_at is None
    assert pin_memory(store, frame.id, pinned=False).expires_at == frame.expires_at
    durable = promote_episode_to_fact(store, frame.id)
    assert durable.kind == "fact" and durable.expires_at is None
    assert current_scope_generation(mind_db.backend) == 3


def test_lineage_and_hash_are_scoped(frame_store, mind_db, mock_embedder):
    frame = frame_store.create_i_frame("gop", "Private.")
    replacement = correct_memory(frame_store, frame.id, "Corrected.").replacement_frame
    other = FrameStore(mind_db, mock_embedder, scope=MemoryScope("other", "p", "n"))
    with pytest.raises(KeyError):
        other.lineage(replacement.id)
    with pytest.raises(KeyError):
        other._content_hash_of(frame.id)


def test_forget_scope_erases_all_scope_data_and_preserves_siblings(
    frame_store, mind_db, mock_embedder, kg, cfg, assert_no_recall
):
    from gluomem.kg.graph import KnowledgeGraph
    from gluomem.service.jobs import JobQueue

    sibling_scope = MemoryScope("other", "user", "personal")
    sibling_store = FrameStore(mind_db, mock_embedder, scope=sibling_scope)
    sibling_kg = KnowledgeGraph(mind_db, scope=sibling_scope)
    sibling_frame = ingest_source_event(
        _evt("Sibling secret BETA.", "sib-1", scope=sibling_scope),
        store=sibling_store,
        cfg=cfg,
    ).frames[0]
    sibling_ent = sibling_kg.create_entity("secret", "BETA", scope=sibling_scope)
    sibling_kg.link_entity_frame(sibling_ent.id, sibling_frame.id, scope=sibling_scope)

    text = "Primary scope secret GAMMA."
    evt = _evt(text, "pri-1")
    frame = ingest_source_event(evt, store=frame_store, cfg=cfg).frames[0]
    JobQueue(mind_db.backend).enqueue(dedup_key="pri-1:1", payload=evt.model_dump(mode="json"))
    ent = kg.create_entity("secret", "GAMMA")
    unlinked = kg.create_entity("project", "OrphanProject")
    kg.link_entity_frame(ent.id, frame.id)
    rel = kg.create_relation(unlinked.id, ent.id, "has_secret")

    res = forget_scope(frame_store)
    assert res.forgotten_frame_ids == [frame.id]
    assert set(res.retired_entity_ids) == {ent.id, unlinked.id}
    assert res.retired_relation_ids == [rel.id]
    assert res.scope_generation == 1
    assert_no_recall("GAMMA", db=mind_db, store=frame_store, kg=kg)
    assert_no_recall("OrphanProject", db=mind_db, store=frame_store, kg=kg)
    assert sibling_store.get_frame(sibling_frame.id).content == "Sibling secret BETA."
    assert sibling_kg.get_entity(sibling_ent.id, scope=sibling_scope).name == "BETA"


def test_sweep_expired_episodes_reclaims_indexes_and_retires_orphaned_kg(
    frame_store, mind_db, mock_embedder, kg, cfg
):
    from gluomem.frames.expiry import sweep_expired_episodes

    cfg = cfg.with_overrides({"lifecycle.episode_ttl_days": 7})
    store = FrameStore(mind_db, mock_embedder, lifecycle=cfg.lifecycle)
    expired = ingest_source_event(
        _evt(
            "Yesterday the staging Redis cluster restarted.",
            "ep-exp",
            created_at="2099-01-01T12:00:00",
            kind="episode",
        ),
        store=store,
        cfg=cfg,
    ).frames[0]
    pinned = ingest_source_event(
        _evt(
            "Yesterday the production failover drill completed.",
            "ep-pin",
            created_at="2099-01-01T12:00:00",
            kind="episode",
            pinned=True,
        ),
        store=store,
        cfg=cfg,
    ).frames[0]
    active_ep = ingest_source_event(
        _evt(
            "Today the canary deployment started.",
            "ep-act",
            created_at="2099-01-06T12:00:00",
            kind="episode",
        ),
        store=store,
        cfg=cfg,
    ).frames[0]

    orphan_ent = kg.create_entity("service", "StagingRedis")
    shared_ent = kg.create_entity("service", "CanaryDeploy")
    kg.link_entity_frame(orphan_ent.id, expired.id)
    kg.link_entity_frame(shared_ent.id, expired.id)
    kg.link_entity_frame(shared_ent.id, active_ep.id)

    assert store.index_state(expired.id).ready
    gen_before = current_scope_generation(mind_db.backend)

    res = sweep_expired_episodes(store, now_iso="2099-01-10T00:00:00")
    assert res.swept_frame_ids == [expired.id]
    assert res.retired_entity_ids == [orphan_ent.id]
    assert res.scope_generation == gen_before + 1

    swept_frame = store.get_frame(expired.id)
    assert swept_frame.status == "expired"
    idx_state = store.index_state(expired.id)
    assert not idx_state.fts_current
    assert idx_state.frame_vec == 0
    assert idx_state.chunks == 0
    assert idx_state.chunk_vecs == 0
    assert store.repair_indexes() == []

    assert store.get_frame(pinned.id).status == "active"
    assert store.index_state(pinned.id).ready
    assert store.get_frame(active_ep.id).status == "active"
    assert store.index_state(active_ep.id).ready
    assert kg.find_valid_entities("StagingRedis", cfg.kg) == []
    assert len(kg.find_valid_entities("CanaryDeploy", cfg.kg)) == 1

    noop = sweep_expired_episodes(store, now_iso="2099-01-10T00:00:00")
    assert noop.swept_frame_ids == []
    assert noop.scope_generation == res.scope_generation
