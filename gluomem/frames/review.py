"""Candidate review, pinning and episode promotion."""

from __future__ import annotations

from typing import Literal

from gluomem.frames.content_hash import content_hash, suppression_key
from gluomem.frames.expiry import episode_expiry
from gluomem.frames.generation import bump_scope_generation
from gluomem.frames.provenance import is_suppressed_by_tombstone
from gluomem.frames.store import Frame, FrameStore
from gluomem.search.multilingual import fold_multilingual_text
from gluomem.storage.scope import MemoryScope, resolve_scope, scope_sql
from gluomem.timeutil import now_utc_iso

ReviewAction = Literal["accept", "edit", "dismiss"]


def list_review_queue(
    store: FrameStore,
    *,
    scope: MemoryScope | None = None,
    limit: int = 100,
) -> list[Frame]:
    """List ``candidate`` frames awaiting review in ``scope``."""
    eff_scope = resolve_scope(scope, default=store.scope)
    rows = store._backend.fetchall(
        "SELECT id FROM memory_frames "
        f"WHERE {scope_sql()} "
        "AND status = 'candidate' "
        "ORDER BY created_at ASC, id ASC LIMIT ?",
        (
            *eff_scope.as_tuple(),
            int(limit),
        ),
    )
    ids = [int(r["id"]) for r in rows]
    fmap = store.get_frames(ids, scope=eff_scope)
    return [fmap[fid] for fid in ids if fid in fmap]


def review_memory(
    store: FrameStore,
    frame_id: int,
    *,
    action: ReviewAction,
    edited_content: str | None = None,
    scope: MemoryScope | None = None,
) -> Frame:
    """Accept, edit or scrub a candidate memory."""
    from gluomem.frames.lifecycle import _scrub_frames

    scope = resolve_scope(scope, default=store.scope)
    frame = store.get_frame(frame_id, scope=scope)
    if frame.status != "candidate":
        raise ValueError("Only candidate memories can be reviewed")
    if action not in ("accept", "edit", "dismiss"):
        raise ValueError(f"Unsupported review action: {action!r}")
    text = (edited_content or "").strip() if action == "edit" else frame.indexable_text
    if action == "edit" and not text:
        raise ValueError("edited_content must be non-empty when action='edit'")
    if action == "edit" and is_suppressed_by_tombstone(
        store._backend,
        scope=scope,
        content_hash_val=content_hash(text),
        suppression_key_val=suppression_key(text),
    ):
        raise ValueError("Edited content is suppressed by a tombstone")
    prepared = store._prepare_index(text) if action != "dismiss" else None
    with store._backend.transaction():
        frame = store.get_frame(frame_id, scope=scope)
        if frame.status != "candidate":
            raise ValueError("Only candidate memories can be reviewed")
        if action == "dismiss":
            row = store._backend.fetchone(
                f"SELECT se.event_id, p.source_ref FROM memory_provenance p LEFT JOIN "
                f"source_events se ON se.id = p.source_event_id WHERE {scope_sql('p')} "
                f"AND p.frame_id = ? ORDER BY p.id LIMIT 1",
                (*scope.as_tuple(), frame.id),
            )
            _scrub_frames(
                store,
                [frame],
                scope=scope,
                status="tombstoned",
                reason="dismissed",
                source_ref=row["source_ref"] if row else None,
                source_event_id=row["event_id"] if row else None,
            )
        else:
            store._update(
                frame.id,
                scope,
                content=text,
                content_hash=content_hash(text),
                suppression_key=suppression_key(text),
                folded_content=fold_multilingual_text(text),
                status="active",
                verified_by_user=1,
                last_accessed=now_utc_iso(),
            )
            assert prepared is not None
            store._write_index(frame.id, prepared)
        bump_scope_generation(store._backend, scope)
    return store.get_frame(frame.id, scope=scope)


def pin_memory(
    store: FrameStore, frame_id: int, *, pinned: bool = True, scope: MemoryScope | None = None
) -> Frame:
    """Pin an active/candidate memory, clearing TTL while pinned."""
    scope = resolve_scope(scope, default=store.scope)
    frame = store.get_frame(frame_id, scope=scope)
    if frame.status not in ("active", "candidate"):
        raise ValueError("Only active or candidate memories can be pinned")
    expiry = (
        None
        if pinned
        else (
            episode_expiry(frame.created_at, store._lifecycle)
            if frame.kind == "episode"
            else frame.expires_at
        )
    )
    with store._backend.transaction():
        store._update(frame.id, scope, pinned=int(pinned), expires_at=expiry)
        bump_scope_generation(store._backend, scope)
    return store.get_frame(frame.id, scope=scope)


def promote_episode_to_fact(
    store: FrameStore,
    frame_id: int,
    *,
    new_kind: Literal["fact", "procedure"] = "fact",
    scope: MemoryScope | None = None,
) -> Frame:
    """Make an active/candidate episode durable by clearing its TTL."""
    scope = resolve_scope(scope, default=store.scope)
    frame = store.get_frame(frame_id, scope=scope)
    if frame.status not in ("active", "candidate") or frame.kind != "episode":
        raise ValueError("Only active or candidate episodes can be promoted")
    with store._backend.transaction():
        store._update(frame.id, scope, kind=new_kind, expires_at=None)
        bump_scope_generation(store._backend, scope)
    return store.get_frame(frame.id, scope=scope)
