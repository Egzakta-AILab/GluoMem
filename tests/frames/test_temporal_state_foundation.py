"""Portable critical contracts; authored inputs and deterministic embeddings only."""

from __future__ import annotations

from importlib.util import find_spec

import pytest


def _api():
    assert find_spec("gluomem.frames.temporal") is not None, "temporal foundation is missing"
    from gluomem.frames import temporal

    return temporal


def test_transition_links_independent_i_versions_without_destructive_correction(frame_store):
    temporal = _api()
    store = temporal.TemporalStore(frame_store)
    successor = frame_store.create_i_frame("new-session", "Lives in Lyon.")
    predecessor = frame_store.create_i_frame("old-session", "Lives in Oslo.")
    store.record_observation(
        predecessor.id,
        temporal.TemporalObservation(effective_at="2025-01-01", observed_at="2025-01-02"),
    )
    metadata = store.record_transition(
        store.revision(predecessor.id),
        successor.id,
        temporal.TemporalObservation(effective_at="2025-06-01", observed_at="2025-05-01"),
    )
    assert metadata.operation == "transition"
    assert metadata.previous_frame_id == predecessor.id
    assert successor.id < predecessor.id
    assert {f.status for f in frame_store.all_frames()} == {"active"}
    assert {f.frame_type for f in frame_store.all_frames()} == {"I"}
    assert frame_store.get_frame(predecessor.id).content == "Lives in Oslo."
    assert frame_store.get_frame(predecessor.id).superseded_by is None
    assert frame_store.get_frame(successor.id).supersedes is None
    assert frame_store.get_frame(successor.id).gop_id == "new-session"


def _timeline(frame_store):
    temporal = _api()
    store = temporal.TemporalStore(frame_store)
    new = frame_store.create_i_frame("home", "Lives in Lyon.")
    old = frame_store.create_i_frame("home", "Lives in Oslo.")
    store.record_observation(
        old.id,
        temporal.TemporalObservation(effective_at="2025-01-01", observed_at="2025-01-02"),
    )
    store.record_transition(
        store.revision(old.id),
        new.id,
        temporal.TemporalObservation(effective_at="2025-06-01", observed_at="2025-05-01"),
    )
    return temporal, store, old, new


@pytest.mark.parametrize(
    "as_of, known_at, expected",
    [
        (None, None, "new"),
        ("2025-05-31T23:59:59Z", None, "old"),
        ("2025-06-01", None, "new"),
        ("2025-07-01", "2025-04-01", "old"),
        ("2025-07-01", "2025-05-15", "new"),
        ("2024-12-31", None, None),
        ("2025-07-01", "2025-01-01", None),
    ],
)
def test_resolve_current_as_of_and_known_at_uses_explicit_chronology(
    frame_store,
    as_of,
    known_at,
    expected,
):
    temporal, store, old, new = _timeline(frame_store)
    query = temporal.TemporalQuery(as_of=as_of, known_at=known_at, now="2025-07-01")
    for seed in (old.id, new.id):
        result = store.resolve(
            seed, query=query, policy=temporal.TemporalPolicy(allow_historical=True)
        )
        assert result.status == ("resolved" if expected else "unknown")
        assert result.frame_id == {"old": old.id, "new": new.id, None: None}[expected]
        assert result.path == (old.id, new.id)
        assert "Lives in" not in repr(result)


@pytest.mark.parametrize("argument", ["as_of", "known_at"])
def test_temporal_views_require_host_history_authorization(frame_store, argument):
    temporal, store, old, new = _timeline(frame_store)
    policy = temporal.TemporalPolicy()
    current = store.resolve(old.id, policy=policy, query=temporal.TemporalQuery(now="2025-07-01"))
    assert current.frame_id == new.id
    with pytest.raises(PermissionError, match="Historical temporal views"):
        store.resolve(
            old.id,
            policy=policy,
            query=temporal.TemporalQuery(**{argument: "2025-03-01"}, now="2025-07-01"),
        )


def test_undated_observation_does_not_fabricate_calendar_history(frame_store):
    temporal = _api()
    store = temporal.TemporalStore(frame_store)
    frame = frame_store.create_i_frame(
        "unknown-date", "An undated assertion.", event_time="2099-01-01"
    )
    store.record_observation(frame.id, temporal.TemporalObservation(observed_at="2025-01-01"))
    policy = temporal.TemporalPolicy(allow_historical=True)
    assert store.resolve(frame.id, policy=policy).frame_id == frame.id
    result = store.resolve(
        frame.id,
        policy=policy,
        query=temporal.TemporalQuery(as_of="1900-01-01"),
    )
    assert result.status == "unknown"
    assert result.frame_id is None


def test_unlinked_i_roots_in_one_gop_do_not_acquire_precedence(frame_store):
    temporal, store, old, new = _timeline(frame_store)
    unrelated = frame_store.create_i_frame("home", "Independent state in the same topic.")
    store.record_observation(unrelated.id, temporal.TemporalObservation(observed_at="2025-01-01"))
    assert store.resolve(unrelated.id, policy=temporal.TemporalPolicy()).frame_id == unrelated.id
    assert store.resolve(old.id, policy=temporal.TemporalPolicy()).frame_id == new.id
    legacy = frame_store.create_i_frame("home", "Legacy undated state with no metadata.")
    assert store.resolve(legacy.id, policy=temporal.TemporalPolicy()).status == "unmanaged"


def test_explicit_hard_forget_closes_temporal_and_typed_version_dependencies(frame_store):
    from gluomem.frames.lifecycle import correct_memory, forget_memory

    temporal, store, old, new = _timeline(frame_store)
    unrelated = frame_store.create_i_frame("home", "Unrelated authorized workshop fact.")
    corrected = correct_memory(frame_store, new.id, "The corrected later state.").replacement_frame
    store.record_observation(
        corrected.id,
        temporal.TemporalObservation(effective_at="2025-06-01", observed_at="2025-06-02"),
    )
    final = frame_store.create_i_frame("later-session", "The final independent dated state.")
    store.record_transition(
        store.revision(corrected.id),
        final.id,
        temporal.TemporalObservation(effective_at="2025-09-01", observed_at="2025-08-01"),
    )
    detail = frame_store.create_p_frame(final.gop_id, "A private final-state detail.", final.id)
    forget_memory(frame_store, old.id)
    for frame in (old, new, corrected, final, detail):
        erased = frame_store.get_frame(frame.id)
        assert erased.status == "forgotten"
        assert erased.content == "[forgotten]"
        assert frame.id not in frame_store.db.frame_vec.ids()
    assert frame_store.get_frame(unrelated.id).content == unrelated.content
    metadata = store.metadata(final.id)
    assert metadata is not None
    assert metadata.previous_frame_id == corrected.id
