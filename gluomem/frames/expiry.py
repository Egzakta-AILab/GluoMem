"""Configured episode expiration and index sweep."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from gluomem.config.app_config import LifecycleConfig
from gluomem.frames.generation import bump_scope_generation, get_scope_generation
from gluomem.storage.scope import MemoryScope, resolve_scope, scope_sql
from gluomem.timeutil import now_utc_iso, parse_utc, to_storage, utc_key

if TYPE_CHECKING:
    from gluomem.frames.store import FrameStore


@dataclass(frozen=True, slots=True)
class SweepResult:
    """Result of sweeping expired frames and reclaiming their search indexes."""

    scope: MemoryScope
    swept_frame_ids: list[int]
    retired_entity_ids: list[int]
    retired_relation_ids: list[int]
    scope_generation: int


def episode_expiry(anchor: str, cfg: LifecycleConfig) -> str:
    """Return the episode expiry anchored at ``anchor`` in UTC."""
    expiry = to_storage(parse_utc(anchor) + timedelta(days=cfg.episode_ttl_days))
    assert expiry is not None
    return expiry


def sweep_expired_episodes(
    store: FrameStore,
    *,
    scope: MemoryScope | None = None,
    now_iso: str | None = None,
) -> SweepResult:
    """Transition expired non-pinned frames to ``expired`` and reclaim their search indexes."""
    from gluomem.frames.lifecycle import _retire_orphaned_kg_for_frame

    eff_scope = resolve_scope(scope, default=store.scope)
    now_s = utc_key(now_iso or now_utc_iso())
    backend = store._backend
    with backend.transaction():
        rows = backend.fetchall(
            f"SELECT id FROM memory_frames WHERE {scope_sql()} "
            "AND status IN ('active', 'candidate') AND pinned = 0 "
            "AND expires_at IS NOT NULL AND REPLACE(expires_at, ' ', 'T') <= ? "
            "ORDER BY id",
            (*eff_scope.as_tuple(), now_s),
        )
        swept_ids = [int(r["id"]) for r in rows]
        if not swept_ids:
            return SweepResult(
                scope=eff_scope,
                swept_frame_ids=[],
                retired_entity_ids=[],
                retired_relation_ids=[],
                scope_generation=get_scope_generation(backend, eff_scope),
            )
        for fid in swept_ids:
            store._delete_index_rows(fid, commit=False)
        ph = ",".join("?" for _ in swept_ids)
        backend.execute(
            f"UPDATE memory_frames SET status = 'expired', last_accessed = ? "
            f"WHERE {scope_sql()} AND id IN ({ph})",
            (now_s, *eff_scope.as_tuple(), *swept_ids),
        )
        eids, rids = _retire_orphaned_kg_for_frame(backend, swept_ids, scope=eff_scope, now_s=now_s)
        gen = bump_scope_generation(backend, eff_scope)
    return SweepResult(
        scope=eff_scope,
        swept_frame_ids=swept_ids,
        retired_entity_ids=eids,
        retired_relation_ids=rids,
        scope_generation=gen,
    )
