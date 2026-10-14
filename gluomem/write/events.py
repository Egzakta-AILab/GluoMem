"""First-class SourceEvent ingestion, provenance spans, review queue, and episode TTL."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from gluomem.frames.content_hash import content_hash
from gluomem.search.multilingual import fold_multilingual_text, fold_to_ascii
from gluomem.storage.scope import (
    DEFAULT_SCOPE,
    FrameStatus,
    MemoryKind,
    MemoryScope,
    resolve_scope,
)
from gluomem.timeutil import now_utc_iso, parse_utc, to_storage

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig
    from gluomem.frames.store import Frame, FrameSource, FrameStore, Trust
    from gluomem.kg.graph import KnowledgeGraph
    from gluomem.storage.backend import DatabaseBackend

SourceType = Literal[
    "chat_turn",
    "document",
    "explicit_memory",
    "tool_result",
    "import",
]
ReviewAction = Literal["accept", "edit", "dismiss"]
TombstoneReason = Literal["dismissed", "forgotten", "retracted"]

_SOURCE_TYPE_TO_FRAME_SOURCE: dict[str, str] = {
    "chat_turn": "user_stated",
    "document": "import",
    "explicit_memory": "user_stated",
    "tool_result": "tool_verified",
    "import": "import",
}


class SourceEventMessage(BaseModel):
    """A single conversational or document segment inside a SourceEvent."""

    model_config = ConfigDict(extra="forbid")

    message_id: str
    role: str = "user"
    content: str
    timestamp: str | None = None


class SourceEvent(BaseModel):
    """Canonical idempotent ingestion envelope for GluoMem."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    event_id: str
    revision: int = Field(default=1, ge=1)
    source_type: SourceType = "chat_turn"
    source_ref: str
    scope: MemoryScope = DEFAULT_SCOPE
    messages: list[SourceEventMessage] = Field(default_factory=list)
    content: str | None = None
    kind: MemoryKind | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    require_review: bool = False
    pinned: bool = False
    event_time: str | None = None
    created_at: str | None = None
    force_reassert: bool = False


@dataclass(frozen=True, slots=True)
class ExtractedClaim:
    """An atomic memory candidate extracted from a SourceEvent with verbatim span."""

    content: str
    verbatim_span: str
    message_id: str | None
    kind: MemoryKind
    confidence: float
    event_time: str | None = None


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Link from a stored memory frame to its originating SourceEvent and span."""

    id: int
    scope: MemoryScope
    frame_id: int
    source_event_id: int | None
    source_type: str
    source_ref: str
    message_id: str | None
    verbatim_span: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class SuppressionTombstone:
    """Tombstone suppressing re-proposal or replay of dismissed/forgotten/retracted content."""

    id: int
    scope: MemoryScope
    content_hash: str
    reason: TombstoneReason
    source_event_id: str | None
    source_ref: str | None
    frame_id: int | None
    created_at: str


@dataclass(slots=True)
class IngestEventResult:
    """Outcome of ingesting a SourceEvent."""

    event_row_id: int
    event_id: str
    revision: int
    deduplicated: bool
    frames: list[Frame] = field(default_factory=list)
    provenance: list[ProvenanceRecord] = field(default_factory=list)
    suppressed_hashes: list[str] = field(default_factory=list)


_PREFERENCE_CUES = (
    "prefer",
    "prefers",
    "preference",
    "always use",
    "never use",
    "favorite",
    "favourite",
    "like to",
    "likes to",
    "volim",
    "preferiram",
    "uvek koristim",
    "nikada ne koristim",
    "omiljen",
)
_DECISION_CUES = (
    "decided",
    "decision",
    "agreed to",
    "we chose",
    "selected",
    "odlucili",
    "odluka",
    "dogovorili",
    "izabrali smo",
    "usvojili",
)
_PROCEDURE_CUES = (
    "procedure",
    "runbook",
    "workflow",
    "step 1",
    "step 2",
    "first,",
    "how to",
    "procedura",
    "uputstvo",
    "korak 1",
    "proces",
)
_EPISODE_CUES = (
    "yesterday",
    "today",
    "last week",
    "meeting",
    "incident",
    "happened",
    "called",
    "visited",
    "juce",
    "danas",
    "prosle nedelje",
    "sastanak",
    "desilo se",
)
_PROFILE_CUES = (
    "my name is",
    "i work at",
    "works at",
    "my role is",
    "role is",
    "title is",
    "lives in",
    "based in",
    "zovem se",
    "radim u",
    "moja uloga",
    "inzenjer",
    "direktor",
    "zivi u",
)


def classify_memory_kind(
    text: str,
    *,
    default: MemoryKind = "fact",
) -> MemoryKind:
    """Classify a statement into one of GluoMem's memory kinds (EN + SR)."""
    folded = fold_to_ascii(text).lower()
    if any(cue in folded for cue in _PROCEDURE_CUES):
        return "procedure"
    if any(cue in folded for cue in _DECISION_CUES):
        return "decision"
    if any(cue in folded for cue in _PREFERENCE_CUES):
        return "preference"
    if any(cue in folded for cue in _PROFILE_CUES):
        return "profile"
    if any(cue in folded for cue in _EPISODE_CUES):
        return "episode"
    return default


def extract_candidate_claims(event: SourceEvent) -> list[ExtractedClaim]:
    """Extract candidate memory claims and verbatim spans from a SourceEvent."""
    claims: list[ExtractedClaim] = []
    if event.messages:
        for msg in event.messages:
            if msg.role not in ("user", "system", "tool"):
                continue
            raw = msg.content.strip()
            if not raw:
                continue
            kind = event.kind or classify_memory_kind(raw)
            claims.append(
                ExtractedClaim(
                    content=raw,
                    verbatim_span=raw,
                    message_id=msg.message_id,
                    kind=kind,
                    confidence=float(event.confidence),
                    event_time=msg.timestamp or event.event_time,
                )
            )
    elif event.content and event.content.strip():
        raw = event.content.strip()
        kind = event.kind or classify_memory_kind(raw)
        claims.append(
            ExtractedClaim(
                content=raw,
                verbatim_span=raw,
                message_id=None,
                kind=kind,
                confidence=float(event.confidence),
                event_time=event.event_time,
            )
        )
    return claims


def _canonical_event_payload(event: SourceEvent) -> tuple[str, str]:
    payload = {
        "event_id": event.event_id,
        "revision": event.revision,
        "source_type": event.source_type,
        "source_ref": event.source_ref,
        "scope": event.scope.as_dict(),
        "messages": [m.model_dump() for m in event.messages],
        "content": event.content,
        "kind": event.kind,
        "confidence": event.confidence,
        "require_review": event.require_review,
        "pinned": event.pinned,
        "event_time": event.event_time,
    }
    payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    return payload_json, payload_hash


def is_suppressed_by_tombstone(
    backend: DatabaseBackend,
    *,
    scope: MemoryScope,
    content_hash_val: str,
    event_id: str | None = None,
    force_reassert: bool = False,
) -> bool:
    """Return True if ``content_hash_val`` is suppressed in ``scope``."""
    eff_scope = resolve_scope(scope)
    rows = backend.fetchall(
        "SELECT reason, source_event_id FROM memory_tombstones "
        "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
        "AND content_hash = ?",
        (
            eff_scope.tenant_id,
            eff_scope.principal_id,
            eff_scope.namespace,
            content_hash_val,
        ),
    )
    for row in rows:
        reason = str(row["reason"])
        src_evt = (
            str(row["source_event_id"])
            if row["source_event_id"] is not None
            else None
        )
        if reason in ("forgotten", "retracted") and not force_reassert:
            return True
        if reason == "dismissed" and (src_evt is None or src_evt == event_id):
            return True
    return False


def get_frame_provenance(
    backend: DatabaseBackend,
    frame_id: int,
    *,
    scope: MemoryScope = DEFAULT_SCOPE,
) -> list[ProvenanceRecord]:
    """Return all provenance records attached to ``frame_id`` in ``scope``."""
    eff_scope = resolve_scope(scope)
    rows = backend.fetchall(
        "SELECT id, tenant_id, principal_id, namespace, frame_id, source_event_id, "
        "source_type, source_ref, message_id, verbatim_span, created_at "
        "FROM memory_provenance "
        "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? AND frame_id = ? "
        "ORDER BY id ASC",
        (
            eff_scope.tenant_id,
            eff_scope.principal_id,
            eff_scope.namespace,
            int(frame_id),
        ),
    )
    return [
        ProvenanceRecord(
            id=int(r["id"]),
            scope=MemoryScope(
                str(r["tenant_id"]), str(r["principal_id"]), str(r["namespace"])
            ),
            frame_id=int(r["frame_id"]),
            source_event_id=(
                int(r["source_event_id"])
                if r["source_event_id"] is not None
                else None
            ),
            source_type=str(r["source_type"]),
            source_ref=str(r["source_ref"]),
            message_id=str(r["message_id"]) if r["message_id"] is not None else None,
            verbatim_span=(
                str(r["verbatim_span"]) if r["verbatim_span"] is not None else None
            ),
            created_at=str(r["created_at"]),
        )
        for r in rows
    ]


def _gop_for_source_ref(source_ref: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", source_ref.strip())
    return f"src:{cleaned or 'default'}"


def ingest_source_event(
    event: SourceEvent,
    *,
    store: FrameStore,
    kg: KnowledgeGraph | None = None,
    cfg: AppConfig | None = None,
    review_threshold: float = 0.75,
    episode_ttl_days: int = 90,
) -> IngestEventResult:
    """Idempotently ingest a ``SourceEvent``, creating provenance-linked frames."""
    backend = store._backend
    eff_scope = resolve_scope(event.scope, default=store.scope)
    payload_json, payload_hash = _canonical_event_payload(event)

    existing = backend.fetchone(
        "SELECT id, payload_hash FROM source_events "
        "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
        "AND event_id = ? AND revision = ?",
        (
            eff_scope.tenant_id,
            eff_scope.principal_id,
            eff_scope.namespace,
            event.event_id,
            int(event.revision),
        ),
    )
    if existing is not None:
        existing_id = int(existing["id"])
        existing_hash = str(existing["payload_hash"])
        if existing_hash != payload_hash:
            raise ValueError(
                f"SourceEvent ({event.event_id!r}, revision={event.revision}) already "
                "ingested with a different payload"
            )
        prov_rows = backend.fetchall(
            "SELECT frame_id FROM memory_provenance "
            "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
            "AND source_event_id = ? ORDER BY id ASC",
            (
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
                existing_id,
            ),
        )
        frame_ids = list(dict.fromkeys(int(r["frame_id"]) for r in prov_rows))
        frames_map = store.get_frames(frame_ids, scope=eff_scope)
        frames = [frames_map[fid] for fid in frame_ids if fid in frames_map]
        all_prov: list[ProvenanceRecord] = []
        for fid in frame_ids:
            all_prov.extend(get_frame_provenance(backend, fid, scope=eff_scope))
        return IngestEventResult(
            event_row_id=existing_id,
            event_id=event.event_id,
            revision=event.revision,
            deduplicated=True,
            frames=frames,
            provenance=all_prov,
            suppressed_hashes=[],
        )

    now_s = to_storage(event.created_at) or now_utc_iso()
    event_row_id = backend.insert_returning_id(
        "INSERT INTO source_events "
        "(tenant_id, principal_id, namespace, event_id, revision, source_type, "
        "source_ref, payload_json, payload_hash, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            eff_scope.tenant_id,
            eff_scope.principal_id,
            eff_scope.namespace,
            event.event_id,
            int(event.revision),
            event.source_type,
            event.source_ref,
            payload_json,
            payload_hash,
            now_s,
        ),
    )
    backend.commit()

    claims = extract_candidate_claims(event)
    gop = _gop_for_source_ref(event.source_ref)
    frame_source: FrameSource = _SOURCE_TYPE_TO_FRAME_SOURCE.get(  # type: ignore[assignment]
        event.source_type, "user_stated"
    )
    trust: Trust = (
        "trusted" if event.source_type == "explicit_memory" else "normal"
    )

    from gluomem.config.app_config import load_config
    from gluomem.write.gate import InjectionScanner, admit

    eff_cfg = cfg if cfg is not None else load_config()
    scanner = InjectionScanner(eff_cfg.safety)

    created_frames: list[Frame] = []
    created_prov: list[ProvenanceRecord] = []
    suppressed_hashes: list[str] = []

    for claim in claims:
        chash = content_hash(claim.content)
        if is_suppressed_by_tombstone(
            backend,
            scope=eff_scope,
            content_hash_val=chash,
            event_id=event.event_id,
            force_reassert=event.force_reassert,
        ):
            suppressed_hashes.append(chash)
            continue

        needs_review = event.require_review or (
            event.source_type != "explicit_memory"
            and claim.confidence < review_threshold
        )
        status: FrameStatus = "candidate" if needs_review else "active"
        verified = event.source_type == "explicit_memory"

        expires_at: str | None = None
        if claim.kind == "episode" and not event.pinned:
            anchor_dt = parse_utc(now_s)
            expires_at = to_storage(anchor_dt + timedelta(days=episode_ttl_days))

        admit_res = admit(
            claim.content,
            gop=gop,
            scanner=scanner,
            store=store,
            kg=kg,
            cfg=eff_cfg,
            source=frame_source,
            trust=trust,
            event_time=claim.event_time,
            created_at=now_s,
            scope=eff_scope,
            kind=claim.kind,
            status=status,
            confidence=claim.confidence,
            expires_at=expires_at,
            pinned=event.pinned,
            verified_by_user=verified,
        )
        if admit_res.frame is None:
            continue
        frame = admit_res.frame
        prov_id = backend.insert_returning_id(
            "INSERT INTO memory_provenance "
            "(tenant_id, principal_id, namespace, frame_id, source_event_id, "
            "source_type, source_ref, message_id, verbatim_span, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
                frame.id,
                event_row_id,
                event.source_type,
                event.source_ref,
                claim.message_id,
                claim.verbatim_span,
                now_s,
            ),
        )
        backend.commit()
        created_frames.append(frame)
        created_prov.append(
            ProvenanceRecord(
                id=prov_id,
                scope=eff_scope,
                frame_id=frame.id,
                source_event_id=event_row_id,
                source_type=event.source_type,
                source_ref=event.source_ref,
                message_id=claim.message_id,
                verbatim_span=claim.verbatim_span,
                created_at=now_s,
            )
        )

    return IngestEventResult(
        event_row_id=event_row_id,
        event_id=event.event_id,
        revision=event.revision,
        deduplicated=False,
        frames=created_frames,
        provenance=created_prov,
        suppressed_hashes=suppressed_hashes,
    )


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
        "WHERE tenant_id = ? AND principal_id = ? AND namespace = ? "
        "AND status = 'candidate' "
        "ORDER BY created_at ASC, id ASC LIMIT ?",
        (
            eff_scope.tenant_id,
            eff_scope.principal_id,
            eff_scope.namespace,
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
    """Accept, edit, or dismiss a candidate memory frame in ``scope``."""
    eff_scope = resolve_scope(scope, default=store.scope)
    frame = store.get_frame(frame_id, scope=eff_scope)
    backend = store._backend
    now_s = now_utc_iso()

    if action == "accept":
        with store._db.transaction():
            backend.execute(
                "UPDATE memory_frames "
                "SET status = 'active', verified_by_user = 1, last_accessed = ? "
                "WHERE id = ? AND tenant_id = ? AND principal_id = ? AND namespace = ?",
                (
                    now_s,
                    frame.id,
                    eff_scope.tenant_id,
                    eff_scope.principal_id,
                    eff_scope.namespace,
                ),
            )
        updated = store.get_frame(frame.id, scope=eff_scope)
        store.ensure_indexed(updated)
        return updated

    if action == "edit":
        if edited_content is None or not edited_content.strip():
            raise ValueError("edited_content must be non-empty when action='edit'")
        new_text = edited_content.strip()
        new_hash = content_hash(new_text)
        folded = fold_multilingual_text(new_text)
        prepared = store._prepare_index(new_text)
        with store._db.transaction():
            backend.execute(
                "UPDATE memory_frames "
                "SET content = ?, folded_content = ?, content_hash = ?, "
                "status = 'active', verified_by_user = 1, last_accessed = ? "
                "WHERE id = ? AND tenant_id = ? AND principal_id = ? AND namespace = ?",
                (
                    new_text,
                    folded,
                    new_hash,
                    now_s,
                    frame.id,
                    eff_scope.tenant_id,
                    eff_scope.principal_id,
                    eff_scope.namespace,
                ),
            )
            store._delete_index_rows(frame.id, commit=False)
            store._write_index(frame.id, prepared)
        return store.get_frame(frame.id, scope=eff_scope)

    if action == "dismiss":
        chash = content_hash(frame.content)
        prov_row = backend.fetchone(
            "SELECT p.source_ref, se.event_id "
            "FROM memory_provenance p "
            "LEFT JOIN source_events se ON se.id = p.source_event_id "
            "WHERE p.tenant_id = ? AND p.principal_id = ? AND p.namespace = ? "
            "AND p.frame_id = ? ORDER BY p.id ASC LIMIT 1",
            (
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
                frame.id,
            ),
        )
        src_event_id = (
            str(prov_row["event_id"])
            if prov_row is not None and prov_row["event_id"] is not None
            else None
        )
        src_ref = (
            str(prov_row["source_ref"])
            if prov_row is not None and prov_row["source_ref"] is not None
            else None
        )
        with store._db.transaction():
            store._delete_index_rows(frame.id, commit=False)
            backend.execute(
                "UPDATE memory_frames "
                "SET status = 'tombstoned', last_accessed = ? "
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
                "VALUES (?, ?, ?, ?, 'dismissed', ?, ?, ?, ?)",
                (
                    eff_scope.tenant_id,
                    eff_scope.principal_id,
                    eff_scope.namespace,
                    chash,
                    src_event_id,
                    src_ref,
                    frame.id,
                    now_s,
                ),
            )
        return store.get_frame(frame.id, scope=eff_scope)

    raise ValueError(f"Unsupported review action: {action!r}")


def pin_memory(
    store: FrameStore,
    frame_id: int,
    *,
    pinned: bool = True,
    scope: MemoryScope | None = None,
) -> Frame:
    """Pin or unpin a memory frame; pinning clears its TTL expiration."""
    eff_scope = resolve_scope(scope, default=store.scope)
    frame = store.get_frame(frame_id, scope=eff_scope)
    with store._db.transaction():
        if pinned:
            store._backend.execute(
                "UPDATE memory_frames SET pinned = 1, expires_at = NULL "
                "WHERE id = ? AND tenant_id = ? AND principal_id = ? AND namespace = ?",
                (
                    frame.id,
                    eff_scope.tenant_id,
                    eff_scope.principal_id,
                    eff_scope.namespace,
                ),
            )
        else:
            expires_at_s: str | None = frame.expires_at
            if frame.kind == "episode" and expires_at_s is None:
                anchor = parse_utc(frame.created_at)
                expires_at_s = to_storage(anchor + timedelta(days=90))
            store._backend.execute(
                "UPDATE memory_frames SET pinned = 0, expires_at = ? "
                "WHERE id = ? AND tenant_id = ? AND principal_id = ? AND namespace = ?",
                (
                    expires_at_s,
                    frame.id,
                    eff_scope.tenant_id,
                    eff_scope.principal_id,
                    eff_scope.namespace,
                ),
            )
    return store.get_frame(frame.id, scope=eff_scope)


def promote_episode_to_fact(
    store: FrameStore,
    frame_id: int,
    *,
    new_kind: Literal["fact", "procedure"] = "fact",
    scope: MemoryScope | None = None,
) -> Frame:
    """Promote an expiring episode frame into a durable fact or procedure (clearing TTL)."""
    eff_scope = resolve_scope(scope, default=store.scope)
    frame = store.get_frame(frame_id, scope=eff_scope)
    with store._db.transaction():
        store._backend.execute(
            "UPDATE memory_frames SET kind = ?, expires_at = NULL "
            "WHERE id = ? AND tenant_id = ? AND principal_id = ? AND namespace = ?",
            (
                new_kind,
                frame.id,
                eff_scope.tenant_id,
                eff_scope.principal_id,
                eff_scope.namespace,
            ),
        )
    return store.get_frame(frame.id, scope=eff_scope)
