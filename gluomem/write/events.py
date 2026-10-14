"""Scoped SourceEvent ingestion with replay receipts and provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from gluomem.frames.content_hash import content_hash
from gluomem.frames.provenance import (
    ProvenanceRecord,
    is_suppressed_by_tombstone,
    redact_event_payload,
)
from gluomem.frames.provenance import (
    TombstoneReason as TombstoneReason,
)
from gluomem.frames.provenance import (
    get_frame_provenance as get_frame_provenance,
)
from gluomem.frames.review import (
    ReviewAction as ReviewAction,
)
from gluomem.frames.review import (
    list_review_queue as list_review_queue,
)
from gluomem.frames.review import (
    pin_memory as pin_memory,
)
from gluomem.frames.review import (
    promote_episode_to_fact as promote_episode_to_fact,
)
from gluomem.frames.review import (
    review_memory as review_memory,
)
from gluomem.frames.store import FrameSource
from gluomem.storage.scope import (
    FrameStatus,
    MemoryKind,
    MemoryScope,
    resolve_scope,
    scope_sql,
)
from gluomem.timeutil import now_utc_iso, to_storage
from gluomem.write.classify import classify_memory_kind
from gluomem.write.gate import InjectionScanner

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig
    from gluomem.frames.store import Frame, FrameStore
    from gluomem.kg.graph import KnowledgeGraph

SourceType = Literal[
    "chat_turn",
    "document",
    "explicit_memory",
    "tool_result",
    "import",
]

_SOURCE_TYPE_TO_FRAME_SOURCE: dict[str, FrameSource] = {
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
    """Scoped, idempotent ingestion envelope."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    event_id: str
    revision: int = Field(default=1, ge=1)
    source_type: SourceType = "chat_turn"
    source_ref: str
    scope: MemoryScope | None = None
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
    """One candidate claim extracted from a source event."""

    content: str
    message_id: str | None
    kind: MemoryKind
    confidence: float
    event_time: str | None = None


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


def extract_candidate_claims(event: SourceEvent) -> list[ExtractedClaim]:
    """Extract non-empty claims, excluding assistant messages."""
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
        "scope": event.scope.as_dict() if event.scope is not None else None,
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


class EventConflictError(ValueError):
    """An event revision was already ingested with different payload bytes."""


def _gop_for_source_ref(source_ref: str) -> str:
    return "src:" + hashlib.sha256(source_ref.encode("utf-8")).hexdigest()


def ingest_source_event(
    event: SourceEvent,
    *,
    store: FrameStore,
    kg: KnowledgeGraph | None = None,
    cfg: AppConfig | None = None,
    scanner: InjectionScanner | None = None,
    write_guard: Callable[[], None] | None = None,
) -> IngestEventResult:
    """Embed first, then commit an event, its claims and provenance together."""
    from gluomem.frames.content_hash import suppression_key
    from gluomem.frames.expiry import episode_expiry
    from gluomem.frames.provenance import get_frames_provenance
    from gluomem.write.gate import admit

    backend = store._backend
    scope = resolve_scope(event.scope, default=store.scope)
    event = event.model_copy(update={"scope": scope})
    payload_json, payload_hash = _canonical_event_payload(event)
    lookup_sql = (
        f"SELECT id, payload_hash FROM source_events WHERE {scope_sql()} "
        "AND event_id = ? AND revision = ?"
    )
    lookup_params = (*scope.as_tuple(), event.event_id, event.revision)

    def duplicate(existing: Mapping[str, Any]) -> IngestEventResult:
        if str(existing["payload_hash"]) != payload_hash:
            raise EventConflictError(
                f"SourceEvent ({event.event_id!r}, revision={event.revision}) already "
                f"ingested with a different payload"
            )
        row_id = int(existing["id"])
        rows = backend.fetchall(
            f"SELECT DISTINCT frame_id FROM memory_provenance WHERE {scope_sql()} AND "
            f"source_event_id = ? ORDER BY frame_id",
            (*scope.as_tuple(), row_id),
        )
        ids = [int(r["frame_id"]) for r in rows]
        frames = store.get_frames(ids, scope=scope)
        return IngestEventResult(
            row_id,
            event.event_id,
            event.revision,
            True,
            [frames[i] for i in ids if i in frames and frames[i].status in ("active", "candidate")],
            get_frames_provenance(backend, ids, scope=scope),
        )

    existing = backend.fetchone(lookup_sql, lookup_params)
    if existing is not None:
        return duplicate(existing)
    eff_cfg = cfg or _default_config()
    scanner = scanner or (_default_scanner() if cfg is None else InjectionScanner(eff_cfg.safety))
    claims = extract_candidate_claims(event)
    prepared = [store._prepare_index(claim.content) for claim in claims]
    now_s = to_storage(event.created_at) or now_utc_iso()
    with backend.transaction():
        row = backend.fetchone(
            "INSERT INTO source_events (tenant_id, principal_id, namespace, event_id, "
            "revision, source_type, "
            "source_ref, payload_json, payload_hash, created_at) VALUES (?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?) "
            "ON CONFLICT (tenant_id, principal_id, namespace, event_id, revision) DO "
            "NOTHING RETURNING id",
            (
                *scope.as_tuple(),
                event.event_id,
                event.revision,
                event.source_type,
                event.source_ref,
                payload_json,
                payload_hash,
                now_s,
            ),
        )
        if write_guard is not None:
            write_guard()
        if row is None:
            existing = backend.fetchone(lookup_sql, lookup_params)
            assert existing is not None
            return duplicate(existing)
        result = IngestEventResult(int(row["id"]), event.event_id, event.revision, False)
        for claim, index in zip(claims, prepared, strict=True):
            chash = content_hash(claim.content)
            if is_suppressed_by_tombstone(
                backend,
                scope=scope,
                content_hash_val=chash,
                suppression_key_val=suppression_key(claim.content),
                event_id=event.event_id,
                force_reassert=event.force_reassert,
            ):
                result.suppressed_hashes.append(chash)
                continue
            needs_review = event.require_review or (
                event.source_type != "explicit_memory"
                and claim.confidence < eff_cfg.lifecycle.review_threshold
            )
            status: FrameStatus = "candidate" if needs_review else "active"
            admitted = admit(
                claim.content,
                gop=_gop_for_source_ref(event.source_ref),
                scanner=scanner,
                store=store,
                kg=kg,
                cfg=eff_cfg,
                source=_SOURCE_TYPE_TO_FRAME_SOURCE[event.source_type],
                trust="trusted" if event.source_type == "explicit_memory" else "normal",
                event_time=claim.event_time,
                created_at=now_s,
                scope=scope,
                kind=claim.kind,
                status=status,
                confidence=claim.confidence,
                expires_at=episode_expiry(now_s, eff_cfg.lifecycle)
                if claim.kind == "episode" and not event.pinned
                else None,
                pinned=event.pinned,
                verified_by_user=event.source_type == "explicit_memory",
                scope_wide=True,
                prepared=index,
            )
            if admitted.frame is None:
                continue
            frame = admitted.frame
            backend.execute(
                "INSERT INTO memory_provenance (tenant_id, principal_id, namespace, "
                "frame_id, source_event_id, "
                "source_type, source_ref, message_id, verbatim_span, created_at) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    *scope.as_tuple(),
                    frame.id,
                    result.event_row_id,
                    event.source_type,
                    event.source_ref,
                    claim.message_id,
                    claim.content,
                    now_s,
                ),
            )
            if frame.id not in {f.id for f in result.frames}:
                result.frames.append(frame)
        # Suppressed/rejected envelopes are receipts, not a second plaintext store.
        if result.suppressed_hashes or len(result.frames) < len(claims):
            redact_event_payload(
                backend,
                result.event_row_id,
                scope=scope,
                event_id=event.event_id,
                revision=event.revision,
            )
        result.provenance = get_frames_provenance(
            backend, [f.id for f in result.frames], scope=scope
        )
    return result


@lru_cache(maxsize=1)
def _default_config() -> AppConfig:
    from gluomem.config.app_config import load_config

    return load_config()


@lru_cache(maxsize=1)
def _default_scanner() -> InjectionScanner:
    return InjectionScanner(_default_config().safety)
