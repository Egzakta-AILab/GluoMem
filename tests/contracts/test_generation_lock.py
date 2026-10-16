"""Generation lock acquisition must not act as a lifecycle invalidation."""

import pytest

from gluomem.frames.expiry import sweep_expired_episodes
from gluomem.frames.generation import (
    bump_scope_generation,
    current_scope_generation,
    lock_scope_generation,
)
from gluomem.frames.lifecycle import forget_scope, retract_source
from gluomem.frames.source_fences import lock_source
from gluomem.storage.scope import MemoryScope, scope_sql


def test_source_lock_initializes_zero_without_bumping(mind_db, frame_store):
    scope = frame_store.scope
    assert mind_db.scalar("SELECT COUNT(*) FROM scope_generations") == 0
    with mind_db.transaction():
        lock_source(mind_db.backend, scope, "doc:new")
        assert (
            mind_db.scalar(
                f"SELECT generation FROM scope_generations WHERE {scope_sql()}", scope.as_tuple()
            )
            == 0
        )
        timestamp = mind_db.scalar("SELECT updated_at FROM scope_generations")
        lock_source(mind_db.backend, scope, "doc:new")
        lock_scope_generation(mind_db.backend, scope)
        assert current_scope_generation(mind_db.backend, scope) == 0
        assert mind_db.scalar("SELECT updated_at FROM scope_generations") == timestamp
        assert bump_scope_generation(mind_db.backend, scope) == 1
    with mind_db.transaction():
        lock_scope_generation(mind_db.backend, scope)
        assert current_scope_generation(mind_db.backend, scope) == 1
    assert mind_db.scalar("SELECT COUNT(*) FROM scope_generations") == 1


def test_generation_lock_and_bump_rollback_together(mind_db, frame_store):
    scope = frame_store.scope
    with pytest.raises(RuntimeError, match="rollback"):
        with mind_db.transaction():
            lock_scope_generation(mind_db.backend, scope)
            assert bump_scope_generation(mind_db.backend, scope) == 1
            raise RuntimeError("rollback")
    assert mind_db.scalar("SELECT COUNT(*) FROM scope_generations") == 0
    with mind_db.transaction():
        lock_scope_generation(mind_db.backend, scope)
        assert bump_scope_generation(mind_db.backend, scope) == 1
    with pytest.raises(RuntimeError, match="rollback"):
        with mind_db.transaction():
            lock_scope_generation(mind_db.backend, scope)
            assert bump_scope_generation(mind_db.backend, scope) == 2
            raise RuntimeError("rollback")
    assert current_scope_generation(mind_db.backend, scope) == 1


def test_empty_lifecycle_counts_and_scoped_lock_preserve_semantics(mind_db, frame_store):
    other = MemoryScope("other", "principal", "namespace")
    with mind_db.transaction():
        lock_scope_generation(mind_db.backend, other)
    assert sweep_expired_episodes(frame_store).scope_generation == 0
    assert forget_scope(frame_store).scope_generation == 1
    assert retract_source(frame_store, "doc:empty").scope_generation == 2
    assert retract_source(frame_store, "doc:empty").scope_generation == 3
    assert forget_scope(frame_store).scope_generation == 4
    assert sweep_expired_episodes(frame_store).scope_generation == 4
    assert current_scope_generation(mind_db.backend, other) == 0
    assert mind_db.scalar("SELECT COUNT(*) FROM source_retractions") == 1
