"""Hard lifecycle operations: correct, forget, retract_source, list, and export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from gluomem.frames.content_hash import content_hash, suppression_key
from gluomem.frames.generation import (
    bump_scope_generation,
    get_scope_generation,
)
from gluomem.frames.generation import current_scope_generation as current_scope_generation
from gluomem.frames.provenance import (
    ProvenanceRecord,
    TombstoneReason,
    get_frames_provenance,
)
from gluomem.frames.review import list_review_queue
from gluomem.storage.scope import (
    MEMORY_KINDS,
    MemoryKind,
    MemoryScope,
    eligible_sql,
    resolve_scope,
    scope_sql,
)
from gluomem.timeutil import now_utc_iso, utc_key

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
class ScopeForgetResult:
    """Result of erasing all memories, provenance, and KG edges in a scope."""

    scope: MemoryScope
    forgotten_frame_ids: list[int]
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


def _retire_orphaned_kg_for_frame(
    backend: DatabaseBackend, frame_ids: list[int], *, scope: MemoryScope, now_s: str
) -> tuple[list[int], list[int]]:
    """Remove frame bridges and erase orphan entities with set-based queries."""
    if not frame_ids:
        return [], []
    ph = ",".join("?" for _ in frame_ids)
    eids = [
        int(r["entity_id"])
        for r in backend.fetchall(
            f"SELECT DISTINCT ef.entity_id FROM kg_entity_frames ef JOIN "
            f"knowledge_entities e ON e.id = ef.entity_id "
            f"WHERE ef.frame_id IN ({ph}) AND {scope_sql('e')}",
            (*frame_ids, *scope.as_tuple()),
        )
    ]
    rids = [
        int(r["relation_id"])
        for r in backend.fetchall(
            f"SELECT DISTINCT rf.relation_id FROM kg_relation_frames rf JOIN "
            f"knowledge_relations r ON r.id = rf.relation_id "
            f"WHERE rf.frame_id IN ({ph}) AND {scope_sql('r')}",
            (*frame_ids, *scope.as_tuple()),
        )
    ]
    backend.execute(f"DELETE FROM kg_entity_frames WHERE frame_id IN ({ph})", frame_ids)
    backend.execute(f"DELETE FROM kg_relation_frames WHERE frame_id IN ({ph})", frame_ids)
    orphan_eids: list[int] = []
    if eids:
        eph = ",".join("?" for _ in eids)
        orphan_eids = [
            int(r["id"])
            for r in backend.fetchall(
                f"SELECT e.id FROM knowledge_entities e WHERE {scope_sql('e')} AND e.id "
                f"IN ({eph}) AND NOT EXISTS "
                f"(SELECT 1 FROM kg_entity_frames ef JOIN memory_frames m ON m.id = ef.frame_id "
                f"WHERE ef.entity_id = e.id AND {scope_sql('m')} AND {eligible_sql('m')})",
                (*scope.as_tuple(), *eids, *scope.as_tuple(), utc_key(now_s)),
            )
        ]
    retired_rids: list[int] = []
    if rids:
        rph = ",".join("?" for _ in rids)
        retired_rids = [
            int(r["id"])
            for r in backend.fetchall(
                f"UPDATE knowledge_relations SET valid_to = ?, properties = '{{}}' WHERE "
                f"{scope_sql()} AND id IN ({rph}) "
                f"AND NOT EXISTS (SELECT 1 FROM kg_relation_frames rf JOIN memory_frames "
                f"m ON m.id = rf.frame_id "
                f"WHERE rf.relation_id = knowledge_relations.id AND {scope_sql('m')} AND "
                f"{eligible_sql('m')}) RETURNING id",
                (now_s, *scope.as_tuple(), *rids, *scope.as_tuple(), utc_key(now_s)),
            )
        ]
    if orphan_eids:
        eph = ",".join("?" for _ in orphan_eids)
        # Entity names must not be recoverable through historical snapshots.
        deleted = backend.fetchall(
            f"DELETE FROM knowledge_relations WHERE {scope_sql()} "
            f"AND (source_id IN ({eph}) OR target_id IN ({eph})) RETURNING id",
            (*scope.as_tuple(), *orphan_eids, *orphan_eids),
        )
        retired_rids.extend(int(r["id"]) for r in deleted)
        backend.execute(
            f"DELETE FROM knowledge_entities WHERE {scope_sql()} AND id IN ({eph})",
            (*scope.as_tuple(), *orphan_eids),
        )
    return orphan_eids, sorted(set(retired_rids))


def _redact_payloads(store: FrameStore, frames: list[Frame], *, scope: MemoryScope) -> None:
    """Redact affected envelopes while retaining their idempotency hashes."""
    import json

    if not frames:
        return
    backend = store._backend
    ids = [f.id for f in frames]
    ph = ",".join("?" for _ in ids)
    events = backend.fetchall(
        f"SELECT DISTINCT se.id, se.event_id, se.revision FROM source_events se "
        f"JOIN memory_provenance p ON p.source_event_id = se.id WHERE {scope_sql('se')} "
        f"AND p.frame_id IN ({ph})",
        (*scope.as_tuple(), *ids),
    )
    if events:
        event_ids = [int(r["id"]) for r in events]
        eph = ",".join("?" for _ in event_ids)
        backend.execute(
            f"UPDATE source_events SET payload_json = '{{}}' WHERE {scope_sql()} AND id IN ({eph})",
            (*scope.as_tuple(), *event_ids),
        )
    dedup_keys = {f"{r['event_id']}:{r['revision']}" for r in events}

    def contains_claim(value: Any) -> bool:
        if isinstance(value, dict):
            return any(contains_claim(v) for v in value.values())
        if isinstance(value, list):
            return any(contains_claim(v) for v in value)
        return isinstance(value, str) and any(f.indexable_text in value for f in frames)

    job_ids: list[int] = []
    for row in backend.fetchall(
        f"SELECT id, dedup_key, payload FROM memory_jobs WHERE {scope_sql()}", scope.as_tuple()
    ):
        payload = json.loads(str(row["payload"]))
        if (
            str(row["dedup_key"]) in dedup_keys
            or payload.get("frame_id") in ids
            or contains_claim(payload)
        ):
            job_ids.append(int(row["id"]))
    if job_ids:
        jph = ",".join("?" for _ in job_ids)
        backend.execute(
            f"UPDATE memory_jobs SET payload = '{{}}', last_error = NULL WHERE "
            f"{scope_sql()} AND id IN ({jph})",
            (*scope.as_tuple(), *job_ids),
        )
    backend.execute(
        f"UPDATE memory_provenance SET verbatim_span = NULL WHERE {scope_sql()} AND "
        f"frame_id IN ({ph})",
        (*scope.as_tuple(), *ids),
    )


def _scrub_frames(
    store: FrameStore,
    frames: list[Frame],
    *,
    scope: MemoryScope,
    status: str,
    reason: TombstoneReason | None = None,
    source_ref: str | None = None,
    source_event_id: str | None = None,
) -> tuple[list[int], list[int]]:
    if not frames:
        return [], []
    now_s = now_utc_iso()
    _redact_payloads(store, frames, scope=scope)
    for frame in frames:
        store._backend.execute(
            "INSERT INTO memory_frame_keys(frame_id, suppression_key) VALUES (?, ?) "
            "ON CONFLICT (frame_id) DO NOTHING",
            (frame.id, suppression_key(frame.indexable_text)),
        )
        store._delete_index_rows(frame.id, commit=False)
        if reason is not None and frame.status not in ("forgotten", "tombstoned"):
            store._insert_tombstone(
                frame, reason=reason, source_ref=source_ref, source_event_id=source_event_id
            )
    ids = [f.id for f in frames]
    ph = ",".join("?" for _ in ids)
    store._backend.execute(
        f"UPDATE memory_frames SET content = '[forgotten]', folded_content = '', status "
        f"= ?, pinned = 0, "
        f"last_accessed = ? WHERE {scope_sql()} AND id IN ({ph})",
        (status, now_s, *scope.as_tuple(), *ids),
    )
    return _retire_orphaned_kg_for_frame(store._backend, ids, scope=scope, now_s=now_s)


def _forget_chain(store: FrameStore, frame_id: int, scope: MemoryScope) -> list[Frame]:
    store.get_frame(frame_id, scope=scope)
    rows = store._backend.fetchall(
        f"WITH RECURSIVE affected(id) AS ("
        f"SELECT id FROM memory_frames WHERE id = ? AND {scope_sql()} UNION "
        "SELECT m.id FROM memory_frames m "
        "LEFT JOIN memory_frame_lineage l ON l.derived_frame_id = m.id "
        "JOIN affected a ON (m.supersedes = a.id OR m.superseded_by = a.id "
        f"OR m.base_frame_id = a.id OR l.source_frame_id = a.id) WHERE {scope_sql('m')}) "
        f"SELECT m.id FROM memory_frames m JOIN affected a ON m.id = a.id WHERE "
        f"{scope_sql('m')} ORDER BY m.id",
        (frame_id, *scope.as_tuple(), *scope.as_tuple(), *scope.as_tuple()),
    )
    ids = [int(r["id"]) for r in rows]
    frames = store.get_frames(ids, scope=scope)
    return [frames[i] for i in ids]


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
    """Replace an active/candidate value and scrub the superseded plaintext."""
    scope = resolve_scope(scope, default=store.scope)
    old = store.get_frame(frame_id, scope=scope)
    if old.status not in ("active", "candidate"):
        raise ValueError("Only active or candidate memories can be corrected")
    text = new_content.strip()
    if not text or content_hash(text) == (old.content_hash or content_hash(old.indexable_text)):
        raise ValueError("Correction must change the normalized content")
    prepared = store._prepare_index(text)
    with store._backend.transaction():
        old = store.get_frame(frame_id, scope=scope)
        if old.status not in ("active", "candidate"):
            raise ValueError("Only active or candidate memories can be corrected")
        affected = _forget_chain(store, old.id, scope)
        replacement = store.create_i_frame(
            old.gop_id,
            text,
            importance=old.importance,
            source="user_stated",
            trust="trusted",
            event_time=event_time or old.event_time,
            created_at=created_at or now_utc_iso(),
            scope=scope,
            kind=old.kind,
            confidence=1.0,
            expires_at=old.expires_at,
            pinned=old.pinned,
            verified_by_user=True,
            deduplicate=False,
            prepared=prepared,
        )
        store._record_lineage(replacement.id, old, old.content_hash, "correct")
        _scrub_frames(
            store,
            affected,
            scope=scope,
            status="superseded",
        )
        store._update(old.id, scope, superseded_by=replacement.id)
        store._update(replacement.id, scope, supersedes=old.id)
        gen = bump_scope_generation(store._backend, scope)
    return CorrectionResult(
        store.get_frame(old.id, scope=scope), store.get_frame(replacement.id, scope=scope), gen
    )


def forget_memory(
    store: FrameStore,
    frame_id: int,
    *,
    scope: MemoryScope | None = None,
    reason: TombstoneReason = "forgotten",
    source_ref: str | None = None,
) -> ForgetResult:
    """Scrub the version chain, dependent frames and payloads, recording replay suppression."""
    scope = resolve_scope(scope, default=store.scope)
    frame = store.get_frame(frame_id, scope=scope)
    with store._backend.transaction():
        frames = _forget_chain(store, frame_id, scope)
        eids, rids = _scrub_frames(
            store, frames, scope=scope, status="forgotten", reason=reason, source_ref=source_ref
        )
        gen = bump_scope_generation(store._backend, scope)
    return ForgetResult(
        frame.id, frame.content_hash or content_hash(frame.content), reason, eids, rids, gen
    )


def forget_scope(
    store: FrameStore,
    *,
    scope: MemoryScope | None = None,
    reason: TombstoneReason = "forgotten",
) -> ScopeForgetResult:
    """Erase all memories, index projections, KG entities/relations, and payloads for ``scope``."""
    eff_scope = resolve_scope(scope, default=store.scope)
    backend = store._backend
    with backend.transaction():
        rows = backend.fetchall(
            f"SELECT id FROM memory_frames WHERE {scope_sql()} AND status != 'forgotten' "
            "ORDER BY id",
            eff_scope.as_tuple(),
        )
        frame_ids = [int(r["id"]) for r in rows]
        frames_map = store.get_frames(frame_ids, scope=eff_scope)
        frames = [frames_map[i] for i in frame_ids]
        eids, rids = _scrub_frames(
            store, frames, scope=eff_scope, status="forgotten", reason=reason
        )
        backend.execute(
            f"UPDATE source_events SET payload_json = '{{}}' WHERE {scope_sql()}",
            eff_scope.as_tuple(),
        )
        backend.execute(
            f"UPDATE memory_jobs SET payload = '{{}}', last_error = NULL WHERE {scope_sql()}",
            eff_scope.as_tuple(),
        )
        backend.execute(
            f"UPDATE memory_provenance SET verbatim_span = NULL WHERE {scope_sql()}",
            eff_scope.as_tuple(),
        )
        extra_rids = [
            int(r["id"])
            for r in backend.fetchall(
                f"DELETE FROM knowledge_relations WHERE {scope_sql()} RETURNING id",
                eff_scope.as_tuple(),
            )
        ]
        extra_eids = [
            int(r["id"])
            for r in backend.fetchall(
                f"DELETE FROM knowledge_entities WHERE {scope_sql()} RETURNING id",
                eff_scope.as_tuple(),
            )
        ]
        gen = bump_scope_generation(backend, eff_scope)
    return ScopeForgetResult(
        scope=eff_scope,
        forgotten_frame_ids=frame_ids,
        retired_entity_ids=sorted(set(eids) | set(extra_eids)),
        retired_relation_ids=sorted(set(rids) | set(extra_rids)),
        scope_generation=gen,
    )


def retract_source(
    store: FrameStore, source_ref: str, *, scope: MemoryScope | None = None
) -> RetractSourceResult:
    """Retract one source and scrub its exclusively supported frames in one transaction."""
    scope = resolve_scope(scope, default=store.scope)
    backend = store._backend
    with backend.transaction():
        rows = backend.fetchall(
            f"SELECT DISTINCT frame_id FROM memory_provenance WHERE {scope_sql()} AND "
            f"source_ref = ? ORDER BY frame_id",
            (*scope.as_tuple(), source_ref),
        )
        ids = [int(r["frame_id"]) for r in rows]
        backend.execute(
            f"UPDATE source_events SET payload_json = '{{}}' WHERE {scope_sql()} AND "
            f"source_ref = ?",
            (*scope.as_tuple(), source_ref),
        )
        events = backend.fetchall(
            f"SELECT event_id, revision FROM source_events WHERE {scope_sql()} AND source_ref = ?",
            (*scope.as_tuple(), source_ref),
        )
        keys = [f"{r['event_id']}:{r['revision']}" for r in events]
        if keys:
            ph = ",".join("?" for _ in keys)
            backend.execute(
                f"UPDATE memory_jobs SET payload = '{{}}', last_error = NULL WHERE "
                f"{scope_sql()} AND dedup_key IN ({ph})",
                (*scope.as_tuple(), *keys),
            )
        backend.execute(
            f"DELETE FROM memory_provenance WHERE {scope_sql()} AND source_ref = ?",
            (*scope.as_tuple(), source_ref),
        )
        retained = {
            int(r["frame_id"])
            for r in backend.fetchall(
                f"SELECT DISTINCT frame_id FROM memory_provenance WHERE {scope_sql()}",
                scope.as_tuple(),
            )
        }
        forgotten = [i for i in ids if i not in retained]
        chains = {f.id: f for i in forgotten for f in _forget_chain(store, i, scope)}
        _scrub_frames(
            store,
            list(chains.values()),
            scope=scope,
            status="forgotten",
            reason="retracted",
            source_ref=source_ref,
        )
        gen = bump_scope_generation(backend, scope)
    return RetractSourceResult(source_ref, forgotten, [i for i in ids if i in retained], gen)


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

    while cur.supersedes is not None and cur.supersedes not in seen:
        seen.add(cur.supersedes)
        cur = store.get_frame(cur.supersedes, scope=eff_scope)

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
    from gluomem.frames.store import _COLUMNS

    eff_scope = resolve_scope(scope, default=store.scope)
    backend = store._backend
    rows = backend.fetchall(
        f"SELECT {_COLUMNS} FROM memory_frames WHERE {scope_sql()} AND {eligible_sql()}"
        + (" AND kind = ?" if kind else "")
        + " ORDER BY id",
        (*eff_scope.as_tuple(), utc_key(now_iso or now_utc_iso()), *((kind,) if kind else ())),
    )
    ids = [int(r["id"]) for r in rows]
    active = [store._row_to_frame(row) for row in rows]
    provenance: dict[int, list[ProvenanceRecord]] = {}
    for p in get_frames_provenance(backend, ids, scope=eff_scope):
        provenance.setdefault(p.frame_id, []).append(p)

    by_kind: dict[str, list[Frame]] = {k: [] for k in MEMORY_KINDS}
    by_source_ref: dict[str, list[Frame]] = {}
    filtered_active: list[Frame] = []

    for f in active:
        provs = provenance.get(f.id, [])
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
    provenance: dict[int, list[ProvenanceRecord]] = {}
    for p in get_frames_provenance(
        backend, [f.id for f in (*grouped.active_frames, *grouped.review_queue)], scope=eff_scope
    ):
        provenance.setdefault(p.frame_id, []).append(p)

    def _serialize_frame(f: Frame) -> dict[str, Any]:
        provs = provenance.get(f.id, [])
        lineage_entries: list[LineageEntry] = store.lineage(f.id, scope=eff_scope)
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
        "by_kind": {k: [f.id for f in frames] for k, frames in grouped.by_kind.items()},
        "by_source_ref": {
            ref: [f.id for f in frames] for ref, frames in grouped.by_source_ref.items()
        },
        "review_queue": [_serialize_frame(f) for f in grouped.review_queue],
    }
