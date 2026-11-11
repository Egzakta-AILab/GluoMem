"""Usable recall integrates metadata-only state resolution, never ID precedence."""

from datetime import UTC, datetime

from gluomem.client.sdk import EmbeddedMemoryClient
from gluomem.frames.temporal import TemporalObservation, TemporalStore
from gluomem.search.hybrid import HybridSearch


def timeline(store):
    new = store.create_i_frame("new-session", "Riley lives in Lyon.")
    old = store.create_i_frame("old-session", "Riley lives in Oslo.")
    temporal = TemporalStore(store)
    temporal.record_observation(
        old.id,
        TemporalObservation(
            observed_at="2025-01-02",
            effective_at="2025-01-01",
        ),
    )
    temporal.record_transition(
        temporal.revision(old.id),
        new.id,
        TemporalObservation(
            observed_at="2025-05-01",
            effective_at="2025-06-01",
        ),
    )
    return old, new


def test_current_recall_resolves_active_old_match_to_explicit_current_identity(
    mind_db, frame_store, cfg, monkeypatch
):
    cfg = cfg.with_overrides({"search.abstain.enabled": False})
    old, new = timeline(frame_store)
    search = HybridSearch(mind_db, frame_store, cfg, clock=lambda: datetime(2025, 9, 1, tzinfo=UTC))
    monkeypatch.setattr(mind_db.chunk_vec, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(mind_db.frame_vec, "search", lambda *args, **kwargs: [])
    assert [hit.frame.id for hit in search.search("Oslo", limit=1).hits] == [old.id]
    result = EmbeddedMemoryClient(mind_db, frame_store, cfg=cfg, search=search).recall("Oslo", k=1)
    assert [hit.frame.id for hit in result.hits] == [new.id]
    assert result.groups[0].seed_ids == [old.id]
    selection = result.groups[0].temporal[0]
    assert selection.requested_frame_id == old.id and selection.selected_frame_id == new.id
    assert selection.path == [old.id, new.id]
    assert selection.effective_at == "2025-06-01T00:00:00.000000Z"
    assert selection.observed_at == "2025-05-01T00:00:00.000000Z"
    assert selection.selection_now == "2025-09-01T00:00:00.000000Z"
    assert "Lyon" in result.context and "Riley lives in Oslo." not in result.context
    assert frame_store.get_frame(old.id).status == frame_store.get_frame(new.id).status == "active"


def test_historical_views_need_host_policy_and_render_explicit_query(mind_db, frame_store, cfg):
    import pytest

    from gluomem.client.sdk import GluoMemApiError
    from gluomem.frames.temporal import TemporalPolicy

    cfg = cfg.with_overrides({"search.abstain.enabled": False})
    old, new = timeline(frame_store)
    denied = EmbeddedMemoryClient(mind_db, frame_store, cfg=cfg)
    with pytest.raises(GluoMemApiError) as error:
        denied.recall("Riley lives", as_of="2025-03-01")
    assert error.value.status_code == 403
    authorized = EmbeddedMemoryClient(
        mind_db, frame_store, cfg=cfg, temporal_policy=TemporalPolicy(allow_historical=True)
    )
    earlier = authorized.recall("Riley lives", as_of="2025-03-01", k=2)
    assert {hit.frame.id for hit in earlier.hits} == {old.id}
    assert earlier.view == "historical" and earlier.as_of == "2025-03-01"
    assert earlier.groups[0].as_of == "2025-03-01"
    assert '"view":"historical"' in earlier.context
    assert '"as_of":"2025-03-01"' in earlier.context
    known = authorized.recall("Riley lives", known_at="2025-04-01", k=2)
    assert {hit.frame.id for hit in known.hits} == {old.id}
    assert known.known_at == "2025-04-01"
    assert "Lyon" not in known.context
    later = authorized.recall("Riley lives", as_of="2025-07-01", known_at="2025-05-15", k=2)
    assert {hit.frame.id for hit in later.hits} == {new.id}


def test_known_at_does_not_render_a_later_refinement_or_its_provenance(
    mind_db, frame_store, cfg, monkeypatch
):
    from gluomem.frames.temporal import TemporalPolicy

    cfg = cfg.with_overrides({"search.abstain.enabled": False})
    old, new = timeline(frame_store)
    late = frame_store.create_p_frame(old.gop_id, "Later private qualifier.", old.id)
    mind_db.execute(
        "INSERT INTO memory_provenance(frame_id, source_type, source_ref, "
        "message_role, observed_at) "
        "VALUES (?, 'chat', 'later', 'user', '2025-06-01')",
        (late.id,),
    )
    for ref, observed in (("early", "2025-01-02"), ("future", "2025-07-01")):
        mind_db.execute(
            "INSERT INTO memory_provenance(frame_id, source_type, source_ref, observed_at) "
            "VALUES (?, 'chat', ?, ?)",
            (old.id, ref, observed),
        )
    mind_db.commit()
    search = HybridSearch(
        mind_db, frame_store, cfg, temporal_policy=TemporalPolicy(allow_historical=True)
    )
    monkeypatch.setattr(mind_db.chunk_vec, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(mind_db.frame_vec, "search", lambda *args, **kwargs: [])
    result = search.recall("Oslo", limit=1, known_at="2025-04-01")
    assert [hit.frame.id for hit in result.hits] == [old.id]
    assert late.content not in result.context
    assert [p.source_ref for p in result.hits[0].frame.provenance] == ["early"]


def test_hard_forget_fence_prevents_historical_temporal_plaintext(mind_db, frame_store, cfg):
    from gluomem.frames.lifecycle import forget_memory
    from gluomem.frames.temporal import TemporalPolicy

    cfg = cfg.with_overrides({"search.abstain.enabled": False})
    old, new = timeline(frame_store)
    client = EmbeddedMemoryClient(
        mind_db, frame_store, cfg=cfg, temporal_policy=TemporalPolicy(allow_historical=True)
    )
    forget_memory(frame_store, new.id)
    current = client.recall("Oslo", k=1)
    assert not current.hits and not current.context
    earlier = client.recall("Oslo", as_of="2025-03-01", k=1)
    assert not earlier.hits and not earlier.context
    for frame_id in (old.id, new.id):
        erased = frame_store.get_frame(frame_id)
        assert erased.status == "forgotten" and erased.content == "[forgotten]"
    assert not HybridSearch(mind_db, frame_store, cfg).search("Oslo", limit=2).hits


def test_retained_legacy_temporal_payload_is_blocked_by_durable_forget_fence(
    mind_db, frame_store, cfg
):
    from gluomem.frames.temporal import TemporalPolicy

    cfg = cfg.with_overrides({"search.abstain.enabled": False})
    old, new = timeline(frame_store)
    # An authored legacy/partial-retention fixture, not the supported forget path.
    with mind_db.transaction():
        frame_store._insert_tombstone(new, reason="forgotten")
    assert frame_store.get_frame(old.id).status == "active"
    client = EmbeddedMemoryClient(
        mind_db, frame_store, cfg=cfg, temporal_policy=TemporalPolicy(allow_historical=True)
    )
    earlier = client.recall("Oslo", as_of="2025-03-01", k=1)
    assert not earlier.hits and not earlier.context
    assert earlier.refusals[0].reason == "temporal_erased"


def test_old_refinement_is_not_rebased_onto_ordinary_new_state(
    mind_db, frame_store, cfg, monkeypatch
):
    from gluomem.frames.temporal import TemporalPolicy

    cfg = cfg.with_overrides({"search.abstain.enabled": False})
    old, new = timeline(frame_store)
    detail = frame_store.create_p_frame(old.gop_id, "Weekends only.", old.id)
    search = HybridSearch(
        mind_db, frame_store, cfg, temporal_policy=TemporalPolicy(allow_historical=True)
    )
    monkeypatch.setattr(mind_db.chunk_vec, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(mind_db.frame_vec, "search", lambda *args, **kwargs: [])
    result = search.recall("Weekends", limit=1)
    assert not result.hits and not result.context
    assert result.refusals[0].reason == "temporal_dependency_changed"
    historical = search.recall("Weekends", limit=1, as_of="2025-03-01")
    assert {hit.frame.id for hit in historical.hits} == {old.id, detail.id}
    assert new.id not in historical.groups[0].frame_ids


def test_withdrawn_latest_is_unknown_current_but_earlier_truth_can_be_authorized(
    mind_db, frame_store, cfg
):
    from gluomem.frames.lifecycle import retract_source
    from gluomem.frames.temporal import TemporalPolicy

    cfg = cfg.with_overrides({"search.abstain.enabled": False})
    old, new = timeline(frame_store)
    mind_db.execute(
        "INSERT INTO memory_provenance(frame_id, source_type, source_ref) "
        "VALUES (?, 'document', 'source:new')",
        (new.id,),
    )
    mind_db.commit()
    retract_source(frame_store, "source:new")
    client = EmbeddedMemoryClient(
        mind_db, frame_store, cfg=cfg, temporal_policy=TemporalPolicy(allow_historical=True)
    )
    assert frame_store.get_frame(old.id).status == "active"
    current = client.recall("Oslo", k=1)
    assert not current.hits and current.refusals[0].reason == "temporal_unavailable"
    earlier = client.recall("Oslo", k=1, as_of="2025-03-01")
    assert [hit.frame.id for hit in earlier.hits] == [old.id]


def test_unmanaged_frames_do_not_fabricate_authorized_calendar_history(mind_db, frame_store, cfg):
    from gluomem.frames.temporal import TemporalPolicy

    cfg = cfg.with_overrides({"search.abstain.enabled": False})
    legacy = frame_store.create_i_frame(
        "legacy", "Legacy unknown-date fact.", event_time="1900-01-01"
    )
    client = EmbeddedMemoryClient(
        mind_db, frame_store, cfg=cfg, temporal_policy=TemporalPolicy(allow_historical=True)
    )
    assert client.recall("Legacy", k=1).hits[0].frame.id == legacy.id
    history = client.recall("Legacy", k=1, as_of="2000-01-01")
    assert not history.hits and history.refusals[0].reason == "temporal_unmanaged"


def test_unmanaged_b_cannot_claim_a_dated_historical_view(mind_db, frame_store, cfg):
    from gluomem.frames.lifecycle import _scrub_frames
    from gluomem.frames.temporal import TemporalPolicy

    cfg = cfg.with_overrides({"search.abstain.enabled": False})
    old = frame_store.create_i_frame("topic", "Prior wrong assertion.")
    correction = frame_store.create_b_frame(
        "topic", "Currentword corrected assertion.", old.id, [old.id], supersedes=old.id
    )
    with mind_db.transaction():
        frame_store._record_lineage(correction.id, old, old.content_hash, "correct")
        _scrub_frames(frame_store, [old], scope=frame_store.scope, status="superseded")
        frame_store._update(old.id, frame_store.scope, superseded_by=correction.id)
    client = EmbeddedMemoryClient(
        mind_db, frame_store, cfg=cfg, temporal_policy=TemporalPolicy(allow_historical=True)
    )
    assert client.recall("Currentword", k=1).hits[0].frame.id == correction.id
    earlier = client.recall("Currentword", k=1, as_of="2000-01-01")
    assert not earlier.hits and earlier.refusals[0].reason == "temporal_unmanaged"
