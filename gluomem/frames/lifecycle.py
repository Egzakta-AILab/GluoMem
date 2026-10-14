"""Hard lifecycle operations: correct, forget, retract_source, list, and export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from gluomem.frames.content_hash import content_hash
from gluomem.frames.eligibility import is_recall_eligible
from gluomem.storage.scope import (
    DEFAULT_SCOPE,
    MEMORY_KINDS,
    MemoryKind,
    MemoryScope,
    resolve_scope,
)
from gluomem.timeutil import now_utc_iso
from gluomem.write.events import (
    ProvenanceRecord,
    TombstoneReason,
    get_frame_provenance,
    list_review_queue,
)

if TYPE_CHECKING:
    from gluomem.frames.store import Frame, FrameStore, LineageEntry
    from gluomem.kg.graph import KnowledgeGraph
    from gluomem.storage.backend import DatabaseBackend


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    """Result of correcting an existing memory frame."""

    superseded_frame: Frame
    replacement_frame: Frame
    scope_generation: int


@dataclass(frozen=True, slots=True)
class ForgetResult:
    """Result of forgetting a memory frame."""

    frame_id: int
    content_hash: str
    reason: TombstoneReason
    retired_entity_ids: list[int]
    retired_relation_ids: list[int]
    scope_generation: int


@dataclass(frozen=True, slots=True)
class RetractSourceResult:
    """Result of retracting all memories exclusively derived from a source_ref."""

    source_ref: str
    forgotten_frame_ids: list[int]
    retained_frame_ids: list[int]
    scope_generation: int


@dataclass(slots=True)
class GroupedMemories:
    """Active memories grouped by kind and source_ref, plus review queue."""

    scope: MemoryScope
    by_kind: dict[str, list[Frame]] = field(default_factory=dict)
    by_source_ref: dict[str, list[Frame]] = field(default_factory=dict)
    active_frames: list[Frame] = field(default_factory=list)
    review_queue: list[Frame] = field(default_factory=list)


def get_scope_generation(
    backend: DatabaseBackend,
    scope: MemoryScope = DEFAULT_SCOPE,
) -> int:
    """Return the current monotonic generation counter for ``scope``."""
    eff_scope = resolve_scope(scope)
    val = backend.scalar(
        "SELECT generation FROM scope_generations "
        "WHERE tenant_id = ? AND principal_id = ? AND namespace = ?",
        (eff_scope.tenant_id, eff_scope.principal_id, eff_scope.namespace),
    )
    return int(val) if val is not None else 0


def bump_scope_generation(
    backend: DatabaseBackend,
    scope: MemoryScope = DEFAULT_SCOPE,
    *,
    commit: bool = False,
) -> int:
    """Atomically increment and return the generation counter for ``scope``."""
    eff_scope = resolve_scope(scope)
    now_s = now_utc_iso()
    current = get_scope_generation(backend, eff_scope)
    next_gen = current + 1
    backend.execute(
        "INSERT INTO scope_generations "
        "(tenant_id, principal_id, namespace, generation, updated_at) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT (tenant_id, principal_id, namespace) DO UPDATE SET "
        "generation = EXCLUDED.generation, updated_at = EXCLUDED.updated_at",
        (
            eff_scope.tenant_id,
            eff_scope.principal_id,
            eff_scope.namespace,
            next_gen,
            now_s,
        ),
    )
    if commit:
        backend.commit()
    return next_gen


def _retire_orphaned_kg_for_frame(
    backend: DatabaseBackend,
    frame_id: int,
    *,
    scope: MemoryScope,
    now_s: str,
) -> tuple[list[int], list[int]]:
    """Unlink ``frame_id`` from ``kg_entity_frames`` and retire orphaned KG entities/relations."""
    eff_scope = resolve_scope(scope)
    ent_rows = backend.fetchall(
        "SELECT ef.entity_id FROM kg_entity_frames ef "
        "JOIN knowledge_entities e ON e.id = ef.entity_id "
        "WHERE ef.frame_id = ? AND e.tenant_id = ? AND e.principal_id = ? AND e.namespace = ?",
        (
            frame_id,
            eff_scope.tenant_id,
            eff_scope.principal_id,
            eff_scope.namespace,
        ),
    )
    affected_eids = [int(r["entity_id"]) for r in ent_rows]
    backend.execute(
        "DELETE FROM kg_entity_frames WHERE frame_id = ?",
        (frame_id,),
    )

    retired_eids: list[int] = []
    retired_rids: list[int] = []
    for eid in affected_eids:
        remaining = backend.scalar(
            "SELECT COUNT(*) FROM kg_entity_frames ef "
            "JOIN memory_frames m ON m.id = ef.frame_id "
            "WHERE ef.entity_id = ? "
            "AND m.tenant_id = ? AND m.principal_id = ? AND m.namespace = ? "
            "AND m.status = 'active' AND m.trust != 'quarantined'",
            (
                eid,
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
            ),
        )
        if not remaining or int(remaining) == 0:
            backend.execute(
                "UPDATE knowledge_entities SET valid_to = ? "
                "WHERE id = ? AND tenant_id = ? AND principal_id = ? AND namespace = ? "
                "AND valid_to IS NULL",
                (
                    now_s,
                    eid,
                    eff_scope.tenant_id,
                    eff_scope.principal_id,
                    eff_scope.namespace,
                ),
            )
            retired_eids.append(eid)
            rel_rows = backend.fetchall(
                "SELECT id FROM knowledge_relations "
                "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
                "AND valid_to IS NULL AND (source_id = ? OR target_id = ?)",
                (
                    eff_scope.tenant_id,
                    eff_scope.principal_id,
                    eff_scope.namespace,
                    eid,
                    eid,
                ),
            )
            for rr in rel_rows:
                rid = int(rr["id"])
                backend.execute(
                    "UPDATE knowledge_relations SET valid_to = ? WHERE id = ?",
                    (now_s, rid),
                )
                retired_rids.append(rid)

    return retired_eids, sorted(set(retired_rids))


def correct_memory(
    store: FrameStore,
    frame_id: int,
    new_content: str,
    *,
    kg: KnowledgeGraph | None = None,
    scope: MemoryScope | None = None,
    created_at: str | None = None,
    event_time: str | None = None,
) -> CorrectionResult:
    """Supersede ``frame_id`` with ``new_content`` and record version lineage."""
    del kg
    if not new_content or not new_content.strip():
        raise ValueError("new_content must be a non-empty string")
    eff_scope = resolve_scope(scope, default=store.scope)
    old_frame = store.get_frame(frame_id, scope=eff_scope)
    backend = store._backend
    now_s = created_at or now_utc_iso()
    cleaned_new = new_content.strip()

    old_hash_row = backend.fetchone(
        "SELECT content_hash FROM memory_frames WHERE id = ?",
        (old_frame.id,),
    )
    old_hash = (
        str(old_hash_row["content_hash"])
        if old_hash_row is not None and old_hash_row["content_hash"]
        else content_hash(old_frame.content)
    )

    new_frame = store.create_i_frame(
        old_frame.gop_id,
        cleaned_new,
        importance=old_frame.importance,
        source="user_stated",
        trust="trusted",
        event_time=event_time or old_frame.event_time,
        created_at=now_s,
        scope=eff_scope,
        kind=old_frame.kind,
        status="active",
        confidence=1.0,
        expires_at=old_frame.expires_at,
        pinned=old_frame.pinned,
        verified_by_user=True,
    )

    with store._db.transaction():
        store._delete_index_rows(old_frame.id, commit=False)
        backend.execute(
            "UPDATE memory_frames "
            "SET status = 'superseded', superseded_by = ?, last_accessed = ? "
            "WHERE id = ? AND tenant_id = ? AND principal_id = ? AND namespace = ?",
            (
                new_frame.id,
                now_s,
                old_frame.id,
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
            ),
        )
        backend.execute(
            "UPDATE memory_frames SET supersedes = ? "
            "WHERE id = ? AND tenant_id = ? AND principal_id = ? AND namespace = ?",
            (
                old_frame.id,
                new_frame.id,
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
            ),
        )
        backend.execute(
            "INSERT INTO memory_frame_lineage "
            "(derived_frame_id, source_frame_id, source_frame_type, "
            "source_content_hash, source_source, source_trust, "
            "source_created_at, source_event_time, operation) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'correct')",
            (
                new_frame.id,
                old_frame.id,
                old_frame.frame_type,
                old_hash,
                old_frame.source,
                old_frame.trust,
                old_frame.created_at,
                old_frame.event_time,
            ),
        )
        backend.execute(
            "INSERT INTO memory_tombstones "
            "(tenant_id, principal_id, namespace, content_hash, reason, "
            "source_event_id, source_ref, frame_id, created_at) "
            "VALUES (?, ?, ?, ?, 'forgotten', NULL, NULL, ?, ?)",
            (
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
                old_hash,
                old_frame.id,
                now_s,
            ),
        )
        _retire_orphaned_kg_for_frame(
            backend, old_frame.id, scope=eff_scope, now_s=now_s
        )
        gen = bump_scope_generation(backend, eff_scope, commit=False)

    return CorrectionResult(
        superseded_frame=store.get_frame(old_frame.id, scope=eff_scope),
        replacement_frame=store.get_frame(new_frame.id, scope=eff_scope),
        scope_generation=gen,
    )


def forget_memory(
    store: FrameStore,
    frame_id: int,
    *,
    scope: MemoryScope | None = None,
    reason: TombstoneReason = "forgotten",
    source_ref: str | None = None,
) -> ForgetResult:
    """Hard-forget ``frame_id`` across vector, FTS, chunks, KG, and record a replay tombstone."""
    eff_scope = resolve_scope(scope, default=store.scope)
    frame = store.get_frame(frame_id, scope=eff_scope)
    backend = store._backend
    now_s = now_utc_iso()

    hash_row = backend.fetchone(
        "SELECT content_hash FROM memory_frames WHERE id = ?",
        (frame.id,),
    )
    orig_hash = (
        str(hash_row["content_hash"])
        if hash_row is not None and hash_row["content_hash"]
        else content_hash(frame.content)
    )

    with store._db.transaction():
        store._delete_index_rows(frame.id, commit=False)
        retired_eids, retired_rids = _retire_orphaned_kg_for_frame(
            backend, frame.id, scope=eff_scope, now_s=now_s
        )
        backend.execute(
            "UPDATE memory_provenance SET verbatim_span = NULL "
            "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? AND frame_id = ?",
            (
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
                frame.id,
            ),
        )
        backend.execute(
            "UPDATE memory_frames "
            "SET content = '[forgotten]', folded_content = '', status = 'forgotten', "
            "pinned = 0, last_accessed = ? "
            "WHERE id = ? AND tenant_id = ? AND principal_id = ? AND namespace = ?",
            (
                now_s,
                frame.id,
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
            ),
        )
        backend.execute(
            "INSERT INTO memory_tombstones "
            "(tenant_id, principal_id, namespace, content_hash, reason, "
            "source_event_id, source_ref, frame_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?)",
            (
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
                orig_hash,
                reason,
                source_ref,
                frame.id,
                now_s,
            ),
        )
        gen = bump_scope_generation(backend, eff_scope, commit=False)

    return ForgetResult(
        frame_id=frame.id,
        content_hash=orig_hash,
        reason=reason,
        retired_entity_ids=retired_eids,
        retired_relation_ids=retired_rids,
        scope_generation=gen,
    )


def retract_source(
    store: FrameStore,
    source_ref: str,
    *,
    scope: MemoryScope | None = None,
) -> RetractSourceResult:
    """Retract ``source_ref`` in ``scope``; forget frames with no remaining corroborating source."""
    eff_scope = resolve_scope(scope, default=store.scope)
    backend = store._backend

    prov_rows = backend.fetchall(
        "SELECT DISTINCT frame_id FROM memory_provenance "
        "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
        "AND source_ref = ? ORDER BY frame_id ASC",
        (
            eff_scope.tenant_id,
            eff_scope.principal_id,
            eff_scope.namespace,
            source_ref,
        ),
    )
    candidate_fids = [int(r["frame_id"]) for r in prov_rows]

    with store._db.transaction():
        backend.execute(
            "DELETE FROM memory_provenance "
            "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
            "AND source_ref = ?",
            (
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
                source_ref,
            ),
        )
        backend.execute(
            "DELETE FROM source_events "
            "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
            "AND source_ref = ?",
            (
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
                source_ref,
            ),
        )

    forgotten_fids: list[int] = []
    retained_fids: list[int] = []

    for fid in candidate_fids:
        remaining = backend.scalar(
            "SELECT COUNT(*) FROM memory_provenance "
            "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
            "AND frame_id = ?",
            (
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
                fid,
            ),
        )
        if not remaining or int(remaining) == 0:
            forget_memory(
                store,
                fid,
                scope=eff_scope,
                reason="retracted",
                source_ref=source_ref,
            )
            forgotten_fids.append(fid)
        else:
            retained_fids.append(fid)

    gen = bump_scope_generation(backend, eff_scope, commit=True)
    return RetractSourceResult(
        source_ref=source_ref,
        forgotten_frame_ids=forgotten_fids,
        retained_frame_ids=retained_fids,
        scope_generation=gen,
    )


def version_lineage(
    store: FrameStore,
    frame_id: int,
    *,
    scope: MemoryScope | None = None,
) -> list[Frame]:
    """Return the chronological supersession chain (oldest -> newest) containing ``frame_id``."""
    eff_scope = resolve_scope(scope, default=store.scope)
    cur = store.get_frame(frame_id, scope=eff_scope)
    seen: set[int] = {cur.id}

    # Walk backward to earliest ancestor
    while cur.supersedes is not None and cur.supersedes not in seen:
        seen.add(cur.supersedes)
        cur = store.get_frame(cur.supersedes, scope=eff_scope)

    # Walk forward from earliest ancestor to newest descendant
    chain: list[Frame] = [cur]
    visited_fwd: set[int] = {cur.id}
    while cur.superseded_by is not None and cur.superseded_by not in visited_fwd:
        visited_fwd.add(cur.superseded_by)
        cur = store.get_frame(cur.superseded_by, scope=eff_scope)
        chain.append(cur)
    return chain


def list_memories(
    store: FrameStore,
    *,
    scope: MemoryScope | None = None,
    kind: MemoryKind | None = None,
    source_ref: str | None = None,
    now_iso: str | None = None,
) -> GroupedMemories:
    """Return recall-eligible active memories grouped by kind and source_ref plus review queue."""
    eff_scope = resolve_scope(scope, default=store.scope)
    backend = store._backend
    all_frames = store.all_frames(scope=eff_scope)

    active: list[Frame] = []
    for f in all_frames:
        if not is_recall_eligible(f, now_iso=now_iso):
            continue
        if kind is not None and f.kind != kind:
            continue
        active.append(f)

    by_kind: dict[str, list[Frame]] = {k: [] for k in MEMORY_KINDS}
    by_source_ref: dict[str, list[Frame]] = {}
    filtered_active: list[Frame] = []

    for f in active:
        provs: list[ProvenanceRecord] = get_frame_provenance(
            backend, f.id, scope=eff_scope
        )
        refs = sorted({p.source_ref for p in provs}) if provs else [f"gop:{f.gop_id}"]
        if source_ref is not None and source_ref not in refs:
            continue
        filtered_active.append(f)
        by_kind.setdefault(f.kind, []).append(f)
        for r in refs:
            by_source_ref.setdefault(r, []).append(f)

    by_kind = {k: v for k, v in by_kind.items() if v}
    queue = list_review_queue(store, scope=eff_scope)
    return GroupedMemories(
        scope=eff_scope,
        by_kind=by_kind,
        by_source_ref=by_source_ref,
        active_frames=filtered_active,
        review_queue=queue,
    )


def export_memories(
    store: FrameStore,
    *,
    scope: MemoryScope | None = None,
    now_iso: str | None = None,
) -> dict[str, Any]:
    """Export active memories, provenance, review queue, and lineage for ``scope``."""
    eff_scope = resolve_scope(scope, default=store.scope)
    grouped = list_memories(store, scope=eff_scope, now_iso=now_iso)
    backend = store._backend

    def _serialize_frame(f: Frame) -> dict[str, Any]:
        provs = get_frame_provenance(backend, f.id, scope=eff_scope)
        lineage_entries: list[LineageEntry] = store.lineage(f.id)
        return {
            "id": f.id,
            "kind": f.kind,
            "status": f.status,
            "content": f.content,
            "importance": f.importance,
            "source": f.source,
            "trust": f.trust,
            "confidence": f.confidence,
            "pinned": f.pinned,
            "verified_by_user": f.verified_by_user,
            "expires_at": f.expires_at,
            "supersedes": f.supersedes,
            "superseded_by": f.superseded_by,
            "created_at": f.created_at,
            "event_time": f.event_time,
            "provenance": [
                {
                    "source_type": p.source_type,
                    "source_ref": p.source_ref,
                    "message_id": p.message_id,
                    "verbatim_span": p.verbatim_span,
                    "created_at": p.created_at,
                }
                for p in provs
            ],
            "lineage": [
                {
                    "source_frame_id": le.source_frame_id,
                    "source_frame_type": le.source_frame_type,
                    "source_content_hash": le.source_content_hash,
                    "operation": le.operation,
                }
                for le in lineage_entries
            ],
        }

    return {
        "scope": eff_scope.as_dict(),
        "scope_generation": get_scope_generation(backend, eff_scope),
        "active_memories": [_serialize_frame(f) for f in grouped.active_frames],
        "by_kind": {
            k: [_serialize_frame(f) for f in frames]
            for k, frames in grouped.by_kind.items()
        },
        "by_source_ref": {
            ref: [f.id for f in frames]
            for ref, frames in grouped.by_source_ref.items()
        },
        "review_queue": [_serialize_frame(f) for f in grouped.review_queue],
    }
