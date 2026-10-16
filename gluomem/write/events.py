"""Scoped SourceEvent ingestion with single-pass extraction, consolidation, and provenance."""

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
from gluomem.frames.source_fences import assert_source_not_retracted
from gluomem.frames.store import FrameSource
from gluomem.llm.fake import NullExtractors
from gluomem.llm.outputs import (
    CognifyResult,
    ConsolidationDecision,
    Extractors,
    KGEntity,
    KGRelation,
    StoredCandidate,
)
from gluomem.storage.scope import (
    FrameStatus,
    MemoryKind,
    MemoryScope,
    resolve_scope,
    scope_sql,
)
from gluomem.timeutil import now_utc_iso, parse_utc, to_storage
from gluomem.write.gate import Scanner, make_scanner

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig
    from gluomem.frames.store import Frame, FrameStore, _PreparedIndex
    from gluomem.kg.graph import KnowledgeGraph

SourceType = Literal[
    "chat_turn",
    "document",
    "doc_chunk",
    "explicit_memory",
    "tool_result",
    "tool_output",
    "import",
]

_SOURCE_TYPE_TO_FRAME_SOURCE: dict[str, FrameSource] = {
    "chat_turn": "user_stated",
    "document": "import",
    "doc_chunk": "import",
    "explicit_memory": "user_stated",
    "tool_result": "tool_verified",
    "tool_output": "tool_verified",
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
    """One atomic claim extracted from a source event."""

    content: str
    message_id: str | None
    kind: MemoryKind
    confidence: float
    event_time: str | None = None
    verbatim_span: str = ""
    entities: tuple[KGEntity, ...] = ()
    relations: tuple[KGRelation, ...] = ()


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


def _direct_claims(event: SourceEvent) -> list[ExtractedClaim]:
    """Direct claim pass-through when kind is explicitly provided or no extractor is active."""
    effective_kind: MemoryKind = event.kind or "fact"
    claims: list[ExtractedClaim] = []
    if event.messages:
        for msg in event.messages:
            if msg.role not in ("user", "system", "tool"):
                continue
            raw = msg.content.strip()
            if not raw:
                continue
            claims.append(
                ExtractedClaim(
                    content=raw,
                    message_id=msg.message_id,
                    kind=effective_kind,
                    confidence=float(event.confidence),
                    event_time=msg.timestamp or event.event_time,
                    verbatim_span=raw,
                )
            )
    elif event.content and event.content.strip():
        raw = event.content.strip()
        claims.append(
            ExtractedClaim(
                content=raw,
                message_id=None,
                kind=effective_kind,
                confidence=float(event.confidence),
                event_time=event.event_time,
                verbatim_span=raw,
            )
        )
    return claims


def _format_event_source(event: SourceEvent) -> tuple[str, dict[str, str | None], str | None]:
    """Format eligible event segments into a single source string for LLM extraction."""
    msg_timestamps: dict[str, str | None] = {}
    default_mid: str | None = None
    if event.messages:
        lines: list[str] = []
        eligible_mids: list[str] = []
        for msg in event.messages:
            if msg.role not in ("user", "system", "tool"):
                continue
            raw = msg.content.strip()
            if not raw:
                continue
            msg_timestamps[msg.message_id] = msg.timestamp or event.event_time
            eligible_mids.append(msg.message_id)
            lines.append(f"[{msg.message_id}] {msg.role}: {raw}")
        if len(eligible_mids) == 1:
            default_mid = eligible_mids[0]
        return "\n".join(lines), msg_timestamps, default_mid
    if event.content and event.content.strip():
        return event.content.strip(), msg_timestamps, None
    return "", msg_timestamps, None


def _active_extractors(extractors: Extractors | None) -> Extractors | None:
    if extractors is None or isinstance(extractors, NullExtractors):
        return None
    return extractors


def extract_event_claims_from_envelope(
    event: SourceEvent,
    *,
    extractors: Extractors | None = None,
) -> list[ExtractedClaim]:
    """Extract atomic claims from a SourceEvent using single-pass Extractors when configured."""
    active_ext = _active_extractors(extractors)
    if (event.source_type == "explicit_memory" and event.kind is not None) or active_ext is None:
        return _direct_claims(event)

    source_text, msg_timestamps, default_mid = _format_event_source(event)
    if not source_text:
        return []

    ref_iso = event.event_time or event.created_at
    ref_dt = None
    if ref_iso:
        try:
            ref_dt = parse_utc(ref_iso)
        except ValueError:
            ref_dt = None

    specs = active_ext.extract_event_claims(source_text, reference_date=ref_dt)
    claims: list[ExtractedClaim] = []
    for spec in specs:
        text = spec.text.strip()
        if not text:
            continue
        mid = spec.message_id if spec.message_id in msg_timestamps else default_mid
        ev_time = (
            to_storage(spec.event_date)
            if spec.event_date is not None
            else (msg_timestamps.get(mid) if mid is not None else event.event_time)
            or event.event_time
        )
        claims.append(
            ExtractedClaim(
                content=text,
                message_id=mid,
                kind=event.kind or spec.kind,
                confidence=min(float(event.confidence), float(spec.confidence)),
                event_time=ev_time,
                verbatim_span=spec.verbatim_span.strip() or text,
                entities=tuple(spec.entities),
                relations=tuple(spec.relations),
            )
        )
    return claims


def _retrieve_consolidation_candidates(
    store: FrameStore,
    prepared: _PreparedIndex,
    *,
    scope: MemoryScope,
    limit: int = 5,
) -> list[StoredCandidate]:
    """Retrieve top-k recall-eligible active frames in scope for claim consolidation."""
    from gluomem.frames.eligibility import is_recall_eligible

    seen_ids: list[int] = []
    for fid in store._fts.search(prepared.text, limit, scope=scope, match_any=True):
        if fid not in seen_ids:
            seen_ids.append(fid)
    if prepared.frame_vec is not None:
        for fid, _ in store._db.frame_vec.search(prepared.frame_vec, limit, scope=scope):
            if fid not in seen_ids:
                seen_ids.append(fid)
    if not seen_ids:
        return []
    frames_map = store.get_frames(seen_ids[:limit], scope=scope)
    candidates: list[StoredCandidate] = []
    for fid in seen_ids[:limit]:
        frame = frames_map.get(fid)
        if frame is not None and frame.status == "active" and is_recall_eligible(frame):
            candidates.append(
                StoredCandidate(
                    frame_id=frame.id,
                    content=frame.content,
                    kind=frame.kind,
                    source=frame.source,
                    event_time=frame.event_time,
                )
            )
    return candidates


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


@dataclass(slots=True)
class PreparedSourceEvent:
    """Phase 1 output of SourceEvent ingestion (computed outside any DB write lock)."""

    event: SourceEvent
    scope: MemoryScope
    payload_json: str
    payload_hash: str
    eff_cfg: AppConfig
    scanner: Scanner
    frame_source: FrameSource
    claims: list[ExtractedClaim] = field(default_factory=list)
    prepared: list[_PreparedIndex] = field(default_factory=list)
    verdicts: list[Any] = field(default_factory=list)
    cognified: list[CognifyResult | None] = field(default_factory=list)
    decisions: list[ConsolidationDecision] = field(default_factory=list)
    early_duplicate: IngestEventResult | None = None


def _duplicate_result(
    existing: Mapping[str, Any],
    *,
    event: SourceEvent,
    scope: MemoryScope,
    payload_hash: str,
    store: FrameStore,
) -> IngestEventResult:
    from gluomem.frames.provenance import get_frames_provenance

    backend = store._backend
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


def prepare_source_event(
    event: SourceEvent,
    *,
    store: FrameStore,
    kg: KnowledgeGraph | None = None,
    cfg: AppConfig | None = None,
    scanner: Scanner | None = None,
    extractors: Extractors | None = None,
    on_progress: Callable[[], None] | None = None,
) -> PreparedSourceEvent:
    """Phase 1: Read + LLM/embedding compute outside any DB write lock."""
    backend = store._backend
    scope = resolve_scope(event.scope, default=store.scope)
    scoped_event = event.model_copy(update={"scope": scope})
    assert_source_not_retracted(backend, scope, scoped_event.source_ref)
    payload_json, payload_hash = _canonical_event_payload(scoped_event)
    eff_cfg = cfg or _default_config()
    eff_scanner = scanner or (_default_scanner() if cfg is None else make_scanner(eff_cfg))
    frame_source = _SOURCE_TYPE_TO_FRAME_SOURCE[scoped_event.source_type]

    lookup_sql = (
        f"SELECT id, payload_hash FROM source_events WHERE {scope_sql()} "
        "AND event_id = ? AND revision = ?"
    )
    lookup_params = (*scope.as_tuple(), scoped_event.event_id, scoped_event.revision)
    existing = backend.fetchone(lookup_sql, lookup_params)
    if existing is not None:
        return PreparedSourceEvent(
            event=scoped_event,
            scope=scope,
            payload_json=payload_json,
            payload_hash=payload_hash,
            eff_cfg=eff_cfg,
            scanner=eff_scanner,
            frame_source=frame_source,
            early_duplicate=_duplicate_result(
                existing,
                event=scoped_event,
                scope=scope,
                payload_hash=payload_hash,
                store=store,
            ),
        )

    if on_progress is not None:
        on_progress()
    active_ext = _active_extractors(extractors)
    claims = extract_event_claims_from_envelope(scoped_event, extractors=active_ext)
    prepared = [store._prepare_index(claim.content) for claim in claims]
    verdicts = [eff_scanner.scan(claim.content, context=frame_source) for claim in claims]

    cognified: list[CognifyResult | None] = []
    used_single_pass = not (
        scoped_event.source_type == "explicit_memory" and scoped_event.kind is not None
    )
    for claim in claims:
        if kg is not None and eff_cfg.write.cognify.enabled:
            if claim.entities or claim.relations:
                cognified.append(
                    CognifyResult(
                        entities=list(claim.entities),
                        relations=list(claim.relations),
                    )
                )
            elif active_ext is not None and not used_single_pass:
                if on_progress is not None:
                    on_progress()
                cognified.append(active_ext.cognify(claim.content))
            else:
                cognified.append(None)
        else:
            cognified.append(None)

    decisions: list[ConsolidationDecision] = []
    for claim, index in zip(claims, prepared, strict=True):
        if active_ext is not None:
            candidates = _retrieve_consolidation_candidates(store, index, scope=scope)
            if candidates:
                if on_progress is not None:
                    on_progress()
                decisions.append(active_ext.consolidate_claim(claim.content, candidates))
            else:
                decisions.append(ConsolidationDecision(action="ADD"))
        else:
            decisions.append(ConsolidationDecision(action="ADD"))

    return PreparedSourceEvent(
        event=scoped_event,
        scope=scope,
        payload_json=payload_json,
        payload_hash=payload_hash,
        eff_cfg=eff_cfg,
        scanner=eff_scanner,
        frame_source=frame_source,
        claims=claims,
        prepared=prepared,
        verdicts=verdicts,
        cognified=cognified,
        decisions=decisions,
    )


def commit_source_event(
    prepared_event: PreparedSourceEvent,
    *,
    store: FrameStore,
    kg: KnowledgeGraph | None = None,
    write_guard: Callable[[], None] | None = None,
) -> IngestEventResult:
    """Phase 2: Short atomic DB write transaction for a prepared SourceEvent."""
    from gluomem.frames.content_hash import suppression_key
    from gluomem.frames.eligibility import is_recall_eligible
    from gluomem.frames.expiry import episode_expiry
    from gluomem.frames.generation import bump_scope_generation
    from gluomem.frames.lifecycle import _scrub_frames
    from gluomem.frames.provenance import get_frames_provenance
    from gluomem.write.cognify import link_and_retire_resolved, resolve_cognify_result
    from gluomem.write.gate import admit

    backend = store._backend
    event = prepared_event.event
    scope = prepared_event.scope
    payload_json = prepared_event.payload_json
    payload_hash = prepared_event.payload_hash
    eff_cfg = prepared_event.eff_cfg
    scanner = prepared_event.scanner
    frame_source = prepared_event.frame_source
    claims = prepared_event.claims
    prepared = prepared_event.prepared
    verdicts = prepared_event.verdicts
    cognified = prepared_event.cognified
    decisions = prepared_event.decisions

    lookup_sql = (
        f"SELECT id, payload_hash FROM source_events WHERE {scope_sql()} "
        "AND event_id = ? AND revision = ?"
    )
    lookup_params = (*scope.as_tuple(), event.event_id, event.revision)
    now_s = to_storage(event.created_at) or now_utc_iso()
    untrusted_sources = set(eff_cfg.safety.untrusted_sources)
    with backend.transaction():
        assert_source_not_retracted(backend, scope, event.source_ref, lock=True)
        if prepared_event.early_duplicate is not None:
            if write_guard is not None:
                write_guard()
            return prepared_event.early_duplicate
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
            return _duplicate_result(
                existing,
                event=event,
                scope=scope,
                payload_hash=payload_hash,
                store=store,
            )
        result = IngestEventResult(int(row["id"]), event.event_id, event.revision, False)
        for claim, index, verdict, cognify_res, decision in zip(
            claims, prepared, verdicts, cognified, decisions, strict=True
        ):
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

            target_frame: Frame | None = None
            if decision.target_frame_id is not None:
                try:
                    cand_frame = store.get_frame(decision.target_frame_id, scope=scope)
                    if cand_frame.status == "active":
                        target_frame = cand_frame
                except KeyError:
                    target_frame = None

            if decision.action == "NOOP" and target_frame is not None:
                backend.execute(
                    "INSERT INTO memory_provenance (tenant_id, principal_id, namespace, "
                    "frame_id, source_event_id, "
                    "source_type, source_ref, message_id, verbatim_span, created_at) VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        *scope.as_tuple(),
                        target_frame.id,
                        result.event_row_id,
                        event.source_type,
                        event.source_ref,
                        claim.message_id,
                        claim.verbatim_span or claim.content,
                        now_s,
                    ),
                )
                if target_frame.id not in {f.id for f in result.frames}:
                    result.frames.append(target_frame)
                continue

            needs_review = (
                event.require_review
                or decision.action == "CANDIDATE"
                or (
                    event.source_type != "explicit_memory"
                    and claim.confidence < eff_cfg.lifecycle.review_threshold
                )
            )
            untrusted_conflict = (
                decision.action == "SUPERSEDE"
                and target_frame is not None
                and frame_source in untrusted_sources
                and target_frame.source not in untrusted_sources
                and not (event.pinned or event.source_type == "explicit_memory")
            )
            if untrusted_conflict:
                needs_review = True

            status: FrameStatus = "candidate" if needs_review else "active"
            admitted = admit(
                claim.content,
                gop=_gop_for_source_ref(event.source_ref),
                scanner=scanner,
                store=store,
                kg=kg,
                cfg=eff_cfg,
                source=frame_source,
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
                verdict=verdict,
            )
            if admitted.frame is None:
                continue
            frame = admitted.frame

            if (
                decision.action == "SUPERSEDE"
                and target_frame is not None
                and target_frame.id != frame.id
                and frame.status == "active"
                and is_recall_eligible(frame)
            ):
                store._record_lineage(
                    frame.id, target_frame, target_frame.content_hash, "supersede"
                )
                _scrub_frames(store, [target_frame], scope=scope, status="superseded")
                store._update(target_frame.id, scope, superseded_by=frame.id)
                store._update(frame.id, scope, supersedes=target_frame.id)
                bump_scope_generation(backend, scope)
                frame = store.get_frame(frame.id, scope=scope)

            if cognify_res is not None and kg is not None and is_recall_eligible(frame):
                resolved = resolve_cognify_result(
                    cognify_res,
                    kg=kg,
                    cfg=eff_cfg,
                    valid_from=frame.event_time,
                    scope=scope,
                )
                link_and_retire_resolved(resolved, frame.id, kg=kg, cfg=eff_cfg, scope=scope)
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
                    claim.verbatim_span or claim.content,
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


def ingest_source_event(
    event: SourceEvent,
    *,
    store: FrameStore,
    kg: KnowledgeGraph | None = None,
    cfg: AppConfig | None = None,
    scanner: Scanner | None = None,
    extractors: Extractors | None = None,
    write_guard: Callable[[], None] | None = None,
) -> IngestEventResult:
    """Two-phase SourceEvent ingestion: Phase 1 outside DB lock, Phase 2 atomic commit."""
    prepared_event = prepare_source_event(
        event,
        store=store,
        kg=kg,
        cfg=cfg,
        scanner=scanner,
        extractors=extractors,
    )
    return commit_source_event(
        prepared_event,
        store=store,
        kg=kg,
        write_guard=write_guard,
    )


@lru_cache(maxsize=1)
def _default_config() -> AppConfig:
    from gluomem.config.app_config import load_config

    return load_config()


@lru_cache(maxsize=1)
def _default_scanner() -> Scanner:
    return make_scanner(_default_config())
