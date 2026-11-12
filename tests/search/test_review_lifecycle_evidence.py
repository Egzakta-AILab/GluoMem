"""Offline regression contracts for compaction and identity-only dependencies."""

import pytest

from gluomem.frames.compaction import compact
from gluomem.frames.lifecycle import _scrub_frames, correct_memory, retract_source
from gluomem.frames.temporal import TemporalObservation, TemporalPolicy, TemporalStore
from gluomem.search.hybrid import HybridSearch


def test_withdrawn_ancestor_keeps_independent_replacement_usable(frame_store, cfg, monkeypatch):
    store = frame_store
    old = store.create_i_frame("topic", "Atlas old secret statement.")
    support(store, old, "old-source", "2025-01-01")
    new = correct_memory(store, old.id, "Atlas corrected statement.").replacement_frame
    support(store, new, "new-source", "2025-06-01")
    search = search_for(store, cfg, monkeypatch)
    assert [h.frame.id for h in search.recall("Atlas", limit=1).hits] == [new.id]
    retract_source(store, "old-source")
    assert store.get_frame(old.id).status == "forgotten"
    assert store.get_frame(old.id).content == "[forgotten]"
    assert store.get_frame(new.id).status == "active"
    result = search.recall("Atlas", limit=1)
    assert [h.frame.id for h in result.hits] == [new.id]
    assert "old secret" not in result.context
    assert [
        (link.frame_id, link.related_frame_id, link.relation) for link in result.groups[0].links
    ] == [(new.id, old.id, "supersedes")]


@pytest.mark.parametrize("broken", ["reciprocal", "typed", "hard-forgotten", "no-tombstone"])
def test_forgotten_identity_requires_authenticated_withdrawal(
    frame_store, cfg, monkeypatch, broken
):
    store = frame_store
    old = store.create_i_frame("topic", "Old secret statement.")
    support(store, old, "old-source", "2025-01-01")
    new = correct_memory(store, old.id, "UniqueReplacement statement.").replacement_frame
    support(store, new, "new-source", "2025-06-01")
    retract_source(store, "old-source")
    if broken == "reciprocal":
        store._update(old.id, store.scope, superseded_by=None)
    elif broken == "typed":
        store.db.execute("DELETE FROM memory_frame_lineage WHERE derived_frame_id = ?", (new.id,))
    elif broken == "no-tombstone":
        store.db.execute("DELETE FROM memory_tombstones WHERE frame_id = ?", (old.id,))
    else:
        store.db.execute(
            "UPDATE memory_tombstones SET reason = 'forgotten' WHERE frame_id = ?", (old.id,)
        )
    store.db.commit()
    result = search_for(store, cfg, monkeypatch).recall("UniqueReplacement", limit=1)
    assert not result.hits
    assert [ref.reason for ref in result.refusals] == ["invalid_version_lineage"]


def supersede_b(store, old, text):
    new = store.create_b_frame(old.gop_id, text, old.id, [old.id], supersedes=old.id)
    with store.db.transaction():
        store._record_lineage(new.id, old, old.content_hash, "correct")
        _scrub_frames(store, [old], scope=store.scope, status="superseded")
        store._update(old.id, store.scope, superseded_by=new.id)
    return new


@pytest.mark.parametrize("depth", [2, 5])
def test_repeated_corrections_inherit_original_context(frame_store, cfg, monkeypatch, depth):
    store = frame_store
    base = store.create_i_frame("topic", "Required original context.")
    original = store.create_p_frame("topic", "Obsolete private qualifier.", base.id)
    old = original
    ancestors = []
    for index in range(depth):
        ancestors.append(old)
        old = supersede_b(store, old, f"UniqueCorrection{index} updated qualifier.")
    result = search_for(store, cfg, monkeypatch).recall(f"UniqueCorrection{depth - 1}", limit=1)
    assert {hit.frame.id for hit in result.hits} == {base.id, old.id}
    assert not result.refusals
    assert "Obsolete private" not in result.context
    links = {
        (link.frame_id, link.related_frame_id, link.relation) for link in result.groups[0].links
    }
    assert (original.id, base.id, "refines") in links
    for previous, successor in zip(ancestors, [*ancestors[1:], old], strict=True):
        assert (successor.id, previous.id, "supersedes") in links


@pytest.mark.parametrize(
    ("broken", "reason"),
    [
        ("typed", "invalid_version_lineage"),
        ("reciprocal", "invalid_version_lineage"),
        ("correction-base", "invalid_correction_base"),
        ("foreign-base", "invalid_base"),
        ("missing-base", "missing_dependency"),
        ("ineligible-base", "ineligible_dependency"),
        ("cycle", "dependency_cycle"),
        ("budget", "evidence_budget_exhausted"),
    ],
)
def test_recursive_context_refuses_broken_dependencies(
    frame_store, cfg, monkeypatch, broken, reason
):
    store = frame_store
    base = store.create_i_frame("topic", "Required context.")
    original = store.create_p_frame("topic", "Private qualifier.", base.id)
    first = supersede_b(store, original, "Earlier correction.")
    latest = supersede_b(store, first, "UniqueLatest correction.")
    if broken == "typed":
        store.db.execute("DELETE FROM memory_frame_lineage WHERE derived_frame_id = ?", (first.id,))
    elif broken == "reciprocal":
        store._update(original.id, store.scope, superseded_by=latest.id)
    elif broken == "correction-base":
        store._update(first.id, store.scope, base_frame_id=base.id)
    elif broken == "foreign-base":
        foreign = store.create_i_frame("other-topic", "Foreign context.")
        store._update(original.id, store.scope, base_frame_id=foreign.id)
    elif broken == "missing-base":
        store._update(original.id, store.scope, base_frame_id=None)
    elif broken == "ineligible-base":
        store._update(base.id, store.scope, status="forgotten")
    elif broken == "cycle":
        store._update(original.id, store.scope, base_frame_id=latest.id)
    store.db.commit()
    result = search_for(store, cfg, monkeypatch).recall(
        "UniqueLatest",
        limit=1,
        evidence_frame_budget=1 if broken == "budget" else 256,
    )
    assert not result.hits
    assert [ref.reason for ref in result.refusals] == [reason]


def support(store, frame, ref, observed):
    store.db.execute(
        "INSERT INTO memory_provenance(frame_id,source_type,source_ref,observed_at) "
        "VALUES (?,'document',?,?)",
        (frame.id, ref, observed),
    )
    store.db.commit()


def search_for(store, cfg, monkeypatch):
    monkeypatch.setattr(store.db.chunk_vec, "search", lambda *a, **k: [])
    monkeypatch.setattr(store.db.frame_vec, "search", lambda *a, **k: [])
    return HybridSearch(
        store.db,
        store,
        cfg.with_overrides({"search.abstain.enabled": False}),
        embedder=store.embedder,
        temporal_policy=TemporalPolicy(allow_historical=True),
    )


@pytest.mark.parametrize("observation_timing", ["before", "during-index"])
def test_compaction_preserves_early_base_without_late_qualifiers(
    frame_store, cfg, monkeypatch, observation_timing
):
    store = frame_store
    base = store.create_i_frame("topic", "Atlas root statement.")
    temporal = TemporalStore(store)
    observation = TemporalObservation("2025-01-01", "2025-01-01")
    if observation_timing == "before":
        temporal.record_observation(base.id, observation)
    support(store, base, "shared", "2025-01-01")
    deltas = [
        store.create_p_frame("topic", text, base.id)
        for text in ("Late secret qualifier.", "Another late qualifier.")
    ]
    for delta in deltas:
        support(store, delta, "shared", "2025-06-01")
    search = search_for(store, cfg, monkeypatch)
    if observation_timing == "before":
        before = search.recall("Atlas", limit=1, known_at="2025-02-01")
        assert [h.frame.id for h in before.hits] == [base.id]
    else:
        prepare_index = store._prepare_index

        def observe_during_preparation(content):
            temporal.record_observation(base.id, observation)
            return prepare_index(content)

        monkeypatch.setattr(store, "_prepare_index", observe_during_preparation)
    compact(
        store,
        "topic",
        cfg.with_overrides(
            {
                "write.compaction.p_frame_trigger": 1,
                "write.compaction.keep_recent": 0,
            }
        ),
    )
    after = search.recall("Atlas", limit=1, known_at="2025-02-01")
    assert [h.frame.content for h in after.hits] == [base.content]
    metadata = temporal.metadata(base.id)
    assert metadata is not None and metadata.observation == observation
    assert all(store.get_frame(p.id).content == p.content for p in deltas)
    current = search.recall("Atlas", limit=1)
    assert {h.frame.id for h in current.hits} == {base.id, *(p.id for p in deltas)}
