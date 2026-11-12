"""Scoped SourceEvent ingestion with single-pass extraction, consolidation, and provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

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
from gluomem.frames.temporal import TemporalObservation, TemporalRevision, TemporalStore
from gluomem.llm.fake import NullExtractors
from gluomem.llm.outputs import (
    TARGET_REQUIRED_ACTIONS,
    ClaimConsolidationContext,
    CognifyResult,
    ConsolidationDecision,
    ConsolidationProjection,
    ContextualConsolidation,
    Extractors,
    KGEntity,
    KGRelation,
    ProjectedConsolidation,
    ProjectedConsolidationDecision,
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
from gluomem.write.event_outcome import EventCommitOutcome
from gluomem.write.gate import Scanner, make_scanner
from gluomem.write.source_binding import CapturePolicy, bind_source_claim

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig
    from gluomem.frames.store import Frame, FrameStore, _PreparedIndex
    from gluomem.kg.graph import KnowledgeGraph
    from gluomem.llm.semantic import ClaimMeaning, SemanticSnapshot, SemanticSourceWitness

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
    capture_policy: CapturePolicy | None = None


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
    source_role: str | None = None
    frame_source: FrameSource | None = None
    observed_at: str | None = None
    # Explicit world date only; event_time retains the legacy source-time fallback.
    effective_at: str | None = None
    # V4 proposals only. Unknown legacy meaning is never inferred or backfilled.
    meaning: ClaimMeaning | None = None
    semantic_witness: SemanticSourceWitness | None = None


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
            if msg.role not in _capture_roles(event):
                continue
            raw = msg.content if event.capture_policy else msg.content.strip()
            if not raw.strip():
                continue
            binding = (
                bind_source_claim(
                    event, message_id=msg.message_id, quote=raw, capture_policy=event.capture_policy
                )
                if event.capture_policy
                else None
            )
            claims.append(
                ExtractedClaim(
                    content=raw,
                    message_id=msg.message_id,
                    kind=effective_kind,
                    confidence=float(event.confidence),
                    event_time=msg.timestamp or event.event_time,
                    effective_at=event.event_time,
                    verbatim_span=raw,
                    source_role=binding.role if binding else None,
                    frame_source=binding.frame_source if binding else None,
                    observed_at=binding.observed_at if binding else None,
                )
            )
    elif event.content and event.content.strip():
        raw = event.content if event.capture_policy else event.content.strip()
        binding = (
            bind_source_claim(
                event, message_id=None, quote=raw, capture_policy=event.capture_policy
            )
            if event.capture_policy
            else None
        )
        claims.append(
            ExtractedClaim(
                content=raw,
                message_id=None,
                kind=effective_kind,
                confidence=float(event.confidence),
                event_time=event.event_time,
                effective_at=event.event_time,
                verbatim_span=raw,
                source_role=binding.role if binding else None,
                frame_source=binding.frame_source if binding else None,
                observed_at=binding.observed_at if binding else None,
            )
        )
    return claims


def _capture_roles(event: SourceEvent) -> tuple[str, ...]:
    roles = ("user", "system", "tool")
    return (*roles, "assistant") if event.capture_policy == "attributed-interaction" else roles


def _format_event_source(event: SourceEvent) -> tuple[str, dict[str, str | None], str | None]:
    """Format eligible event segments into a single source string for LLM extraction."""
    msg_timestamps: dict[str, str | None] = {}
    default_mid: str | None = None
    if event.messages:
        lines: list[str] = []
        eligible_mids: list[str] = []
        for msg in event.messages:
            if msg.role not in _capture_roles(event):
                continue
            raw = msg.content if event.capture_policy else msg.content.strip()
            if not raw.strip():
                continue
            msg_timestamps[msg.message_id] = msg.timestamp or event.event_time
            eligible_mids.append(msg.message_id)
            lines.append(f"[{msg.message_id}] {msg.role}: {raw}")
        if len(eligible_mids) == 1:
            default_mid = eligible_mids[0]
        return "\n".join(lines), msg_timestamps, default_mid
    if event.content and event.content.strip():
        raw = event.content if event.capture_policy else event.content.strip()
        return raw, msg_timestamps, None
    return "", msg_timestamps, None


def _active_extractors(extractors: Extractors | None) -> Extractors | None:
    if extractors is None or isinstance(extractors, NullExtractors):
        return None
    return extractors


def extract_event_claims_from_envelope(
    event: SourceEvent,
    *,
    extractors: Extractors | None = None,
    writer_contract: str | None = None,
    semantic_producer_contract: str | None = None,
) -> list[ExtractedClaim]:
    """Extract atomic claims from a SourceEvent using single-pass Extractors when configured."""
    active_ext = _active_extractors(extractors)
    if writer_contract == "semantic-state-writer-v4" and (
        active_ext is None
        or getattr(active_ext, "writer_capability", None) != writer_contract
        or event.capture_policy is None
    ):
        raise ValueError(
            "Semantic writer requires its explicit native capability and source policy"
        )
    if semantic_producer_contract == "observed-assertion-scope-v4" and (
        getattr(active_ext, "semantic_producer_capability", None) != semantic_producer_contract
    ):
        raise ValueError("Assertion scope requires its explicit producer capability")
    if (
        (event.source_type == "explicit_memory" and event.kind is not None) or active_ext is None
    ) and writer_contract != "semantic-state-writer-v4":
        return _direct_claims(event)

    assert active_ext is not None  # Both non-extraction paths returned or refused above.
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

    if writer_contract:
        from gluomem.write.source_binding import ACTIVE_SOURCE_ENVELOPE

        token = ACTIVE_SOURCE_ENVELOPE.set(
            {"writer_contract": writer_contract, "source_event": event.model_dump(mode="json")}
        )
        try:
            specs = active_ext.extract_event_claims(source_text, reference_date=ref_dt)
        finally:
            ACTIVE_SOURCE_ENVELOPE.reset(token)
    else:
        specs = active_ext.extract_event_claims(source_text, reference_date=ref_dt)
    claims: list[ExtractedClaim] = []
    for spec in specs:
        semantic_witness = None
        semantic_meaning = None
        if writer_contract == "semantic-state-writer-v4":
            from gluomem.llm.semantic import SemanticClaimSpec, source_witness

            parser = SemanticClaimSpec
            if semantic_producer_contract == "observed-assertion-scope-v4":
                from gluomem.llm.assertion_scope import scoped_claim_type

                parser = scoped_claim_type()
            semantic_spec = parser.model_validate(spec.model_dump())
            semantic_meaning = semantic_spec.meaning
            spec = semantic_spec
            semantic_witness = source_witness(event, spec)
        text = spec.text.strip()
        if not text:
            if writer_contract == "semantic-state-writer-v4":
                raise ValueError("Blank semantic claim cannot be silently discarded")
            continue
        if event.capture_policy:
            mid = spec.message_id if spec.message_id is not None else default_mid
            binding = bind_source_claim(
                event,
                message_id=mid,
                quote=spec.verbatim_span,
                capture_policy=event.capture_policy,
                writer_contract=writer_contract,
            )
        else:
            mid = spec.message_id if spec.message_id in msg_timestamps else default_mid
            binding = None
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
                effective_at=(
                    to_storage(spec.event_date)
                    if spec.event_date is not None
                    else None
                    if event.capture_policy
                    else event.event_time
                ),
                verbatim_span=(
                    binding.verbatim_span if binding else spec.verbatim_span.strip() or text
                ),
                entities=tuple(spec.entities),
                relations=tuple(spec.relations),
                source_role=binding.role if binding else None,
                frame_source=binding.frame_source if binding else None,
                observed_at=binding.observed_at if binding else None,
                meaning=semantic_meaning,
                semantic_witness=semantic_witness,
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
            metadata = TemporalStore(store).metadata(frame.id, scope=scope)
            candidates.append(
                StoredCandidate(
                    frame_id=frame.id,
                    content=frame.content,
                    frame_type=frame.frame_type,
                    kind=frame.kind,
                    source=frame.source,
                    event_time=frame.event_time,
                    temporal_operation=metadata.operation if metadata else None,
                    previous_frame_id=metadata.previous_frame_id if metadata else None,
                    observed_at=metadata.observation.observed_at if metadata else None,
                    effective_at=metadata.observation.effective_at if metadata else None,
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
    if event.capture_policy is not None:
        payload["capture_policy"] = event.capture_policy
        payload["created_at"] = event.created_at
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
    decisions: list[ConsolidationDecision | ProjectedConsolidationDecision] = field(
        default_factory=list
    )
    projections: list[ConsolidationProjection | None] = field(default_factory=list)
    early_duplicate: IngestEventResult | None = None
    candidate_revisions: list[dict[int, str]] = field(default_factory=list)
    temporal_revisions: list[dict[int, TemporalRevision]] = field(default_factory=list)
    projection_generation: int | None = None
    semantic_inputs: tuple[ExtractedClaim, ...] | None = None
    semantic_revisions: dict[int, SemanticSnapshot | None] = field(default_factory=dict)
    producer_inputs: str | None = None
    receipt_policy: str | None = None
    receipt_support: tuple = ()


def _frame_revision(frame: Frame) -> str:
    """Bind semantic, scope and mutable admission policy, not read bookkeeping."""
    values = asdict(frame)
    for key in ("access_count", "last_accessed"):
        values.pop(key)
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _transition_revision(
    decision: ConsolidationDecision,
    claim: ExtractedClaim,
    revisions: Mapping[int, TemporalRevision],
    *,
    temporal: TemporalStore,
    scope: MemoryScope,
    owned_generation_bumps: int = 0,
) -> TemporalRevision:
    target_id = decision.target_frame_id
    expected = revisions.get(target_id) if target_id is not None else None
    if expected is None or expected.metadata is None:
        raise ValueError("TRANSITION requires an observed I-frame with temporal metadata")
    if claim.observed_at is None or claim.effective_at is None:
        raise ValueError("TRANSITION requires trusted observed_at and explicit effective time")
    # Account only for named writes below, never replace a frozen digest/metadata
    # or infer an arbitrary observed generation delta after a callback.
    expected = replace(expected, generation=expected.generation + owned_generation_bumps)
    if expected != temporal.revision(expected.frame_id, scope=scope):
        raise ValueError("Stale temporal transition target; prepare ingestion again")
    return expected


def _consolidation_target(
    decision: ConsolidationDecision,
    revisions: Mapping[int, str],
    *,
    store: FrameStore,
    scope: MemoryScope,
) -> Frame | None:
    from gluomem.frames.eligibility import is_recall_eligible

    if decision.action in TARGET_REQUIRED_ACTIONS and decision.target_frame_id is None:
        raise ValueError("Consolidation action requires an observed candidate target")
    # Custom facades and model_construct must not bypass the typed final defense.
    checked = ConsolidationDecision.model_validate(decision.model_dump())
    target_id = checked.target_frame_id
    if target_id is None:
        return None
    if target_id not in revisions:
        raise ValueError("Consolidation target is outside the prepared candidate set")
    try:
        current = store.get_frame(target_id, scope=scope)
    except KeyError:
        raise ValueError("Stale consolidation target; prepare ingestion again") from None
    if _frame_revision(current) != revisions[target_id] or not is_recall_eligible(current):
        raise ValueError("Stale consolidation target changed; prepare ingestion again")
    if checked.action == "REFINE" and current.frame_type != "I":
        raise ValueError("REFINE requires an observed eligible I-frame target")
    return current


@contextmanager
def _event_write_transaction(
    store: FrameStore, scope: MemoryScope, source_ref: str
) -> Iterator[None]:
    """Roll back the event, never a successful reentrant forget/retraction.

    The savepoint contains only event publication. On refusal, retain *erasure
    intent*, not plaintext snapshots, and apply it through the lifecycle API to
    the restored pre-event state while the outer generation/source lock is held.
    Raise after this transaction exits. If joined to a caller-owned transaction,
    only that owner can commit the replay: use ``attempt_commit_source_event``
    and raise its refusal AFTER the outer transaction exits normally. An
    arbitrary caller rolling its own outer transaction back can undo erasure.
    This is not a command planner/rebaser.
    """
    from gluomem.frames.lifecycle import forget_memory, retract_source

    backend = store._backend
    failure: Exception | None = None
    with backend.transaction():
        assert_source_not_retracted(backend, scope, source_ref, lock=True)
        watermark = int(backend.scalar("SELECT COALESCE(MAX(id), 0) FROM memory_tombstones"))
        old_sources = {
            str(row["source_ref"])
            for row in backend.fetchall(
                f"SELECT source_ref FROM source_retractions WHERE {scope_sql()}", scope.as_tuple()
            )
        }
        savepoint = "event_" + uuid4().hex
        backend.execute(f"SAVEPOINT {savepoint}")
        try:
            yield
        except Exception as exc:
            erased = backend.fetchall(
                f"SELECT t.*, k.suppression_key FROM memory_tombstones t "
                "LEFT JOIN memory_tombstone_keys k ON k.tombstone_id = t.id "
                f"WHERE {scope_sql('t')} AND t.id > ? ORDER BY t.id",
                (*scope.as_tuple(), watermark),
            )
            removed_sources = {
                str(row["source_ref"])
                for row in backend.fetchall(
                    f"SELECT source_ref FROM source_retractions WHERE {scope_sql()}",
                    scope.as_tuple(),
                )
            } - old_sources
            backend.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            backend.execute(f"RELEASE SAVEPOINT {savepoint}")
            if not erased and not removed_sources:
                raise
            for removed in sorted(removed_sources):
                retract_source(store, removed, scope=scope)
            for tombstone in erased:
                fid = tombstone["frame_id"]
                try:
                    frame = store.get_frame(int(fid), scope=scope) if fid is not None else None
                except KeyError:
                    frame = None  # An event-local publication was rolled back.
                if frame is not None and frame.status not in ("forgotten", "tombstoned"):
                    forget_memory(
                        store,
                        frame.id,
                        scope=scope,
                        reason=tombstone["reason"],
                        source_ref=tombstone["source_ref"],
                    )
                # Preserve even a rolled-back publication's hash suppression and
                # source-event binding without recreating its frame/plaintext.
                row = backend.fetchone(
                    "INSERT INTO memory_tombstones (tenant_id, principal_id, namespace, "
                    "content_hash, reason, source_event_id, source_ref, frame_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
                    (
                        *scope.as_tuple(),
                        tombstone["content_hash"],
                        tombstone["reason"],
                        tombstone["source_event_id"],
                        tombstone["source_ref"],
                        frame.id if frame is not None else None,
                        tombstone["created_at"],
                    ),
                )
                assert row is not None
                if tombstone["suppression_key"] is not None:
                    backend.execute(
                        "INSERT INTO memory_tombstone_keys(tombstone_id, suppression_key) "
                        "VALUES (?, ?)",
                        (int(row["id"]), tombstone["suppression_key"]),
                    )
            failure = exc
        else:
            backend.execute(f"RELEASE SAVEPOINT {savepoint}")
    if failure is not None:
        raise failure


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
    ids = [i for i in ids if i in frames and frames[i].status in ("active", "candidate")]
    return IngestEventResult(
        row_id,
        event.event_id,
        event.revision,
        True,
        [frames[i] for i in ids],
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
    eff_cfg = cfg or _default_config()
    capture_policy = event.capture_policy or eff_cfg.write.source_capture_policy
    scoped_event = event.model_copy(update={"scope": scope, "capture_policy": capture_policy})
    assert_source_not_retracted(backend, scope, scoped_event.source_ref)
    payload_json, payload_hash = _canonical_event_payload(scoped_event)
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
    from gluomem.frames.generation import current_scope_generation

    projection_generation = (
        current_scope_generation(backend, scope)
        if isinstance(active_ext, ProjectedConsolidation)
        else None
    )
    claims = extract_event_claims_from_envelope(
        scoped_event,
        extractors=active_ext,
        writer_contract=eff_cfg.write.writer_contract,
        semantic_producer_contract=eff_cfg.write.semantic_producer_contract,
    )
    prepared = [store._prepare_index(claim.content) for claim in claims]
    verdicts = [
        eff_scanner.scan(claim.content, context=claim.frame_source or frame_source)
        for claim in claims
    ]

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

    from gluomem.frames.semantics import SemanticStore
    from gluomem.llm.semantic import SemanticClaimContext, SemanticStoredCandidate
    from gluomem.write.consolidation_projection import ClaimProjectionState
    from gluomem.write.semantic_policy import SemanticProjectionState

    semantic_mode = eff_cfg.write.writer_contract == "semantic-state-writer-v4"
    if semantic_mode:
        if eff_cfg.write.semantic_producer_contract == "observed-assertion-scope-v4":
            from gluomem.llm.assertion_scope import (
                AssertionScopedCandidate as SemanticStoredCandidate,
            )
            from gluomem.llm.assertion_scope import (
                AssertionScopedContext as SemanticClaimContext,
            )

        if not isinstance(active_ext, ProjectedConsolidation):
            raise ValueError("Semantic writer requires the projected consolidation capability")
    semantic_projection_state = SemanticProjectionState() if semantic_mode else None
    projection_state = semantic_projection_state or ClaimProjectionState()
    if eff_cfg.write.semantic_producer_contract is not None:
        from gluomem.write.observed_identity import ObservedProjectionState

        semantic_projection_state = ObservedProjectionState(
            active_ext,
            producer_contract=eff_cfg.write.semantic_producer_contract,
            event=scoped_event,
            cfg=eff_cfg,
            verdicts=verdicts,
            frame_source=frame_source,
        )
        projection_state = semantic_projection_state
    projections: list[ConsolidationProjection | None] = []
    decisions: list[ConsolidationDecision | ProjectedConsolidationDecision] = []
    candidate_revisions: list[dict[int, str]] = []
    temporal_revisions: list[dict[int, TemporalRevision]] = []
    for claim_index, (claim, index) in enumerate(zip(claims, prepared, strict=True)):
        candidates = (
            _retrieve_consolidation_candidates(store, index, scope=scope)
            if active_ext is not None
            else []
        )
        observations = {}
        temporal_observations = {}
        if semantic_mode:
            candidates = [
                SemanticStoredCandidate(
                    **c.model_dump(),
                    semantic=SemanticStore(store).snapshot(
                        c.frame_id,
                        scope=scope,
                        producer_contract=eff_cfg.write.semantic_producer_contract,
                    ),
                )
                for c in candidates
            ]
        for candidate in candidates:
            observed = store.get_frame(candidate.frame_id, scope=scope)
            if (
                observed.content != candidate.content
                or observed.frame_type != candidate.frame_type
                or observed.kind != candidate.kind
                or observed.source != candidate.source
                or observed.event_time != candidate.event_time
            ):
                raise ValueError("Consolidation candidate changed; prepare ingestion again")
            observations[observed.id] = _frame_revision(observed)
            if observed.frame_type == "I":
                revision = TemporalStore(store).revision(observed.id, scope=scope)
                if revision.digest != observations[observed.id]:
                    raise ValueError("Consolidation candidate changed; prepare ingestion again")
                metadata = revision.metadata
                if (
                    candidate.temporal_operation != (metadata.operation if metadata else None)
                    or candidate.previous_frame_id
                    != (metadata.previous_frame_id if metadata else None)
                    or candidate.observed_at
                    != (metadata.observation.observed_at if metadata else None)
                    or candidate.effective_at
                    != (metadata.observation.effective_at if metadata else None)
                ):
                    raise ValueError(
                        "Consolidation candidate metadata changed; prepare ingestion again"
                    )
                temporal_observations[observed.id] = revision
        candidate_revisions.append(observations)
        temporal_revisions.append(temporal_observations)
        if isinstance(active_ext, ProjectedConsolidation):
            projection = projection_state.project(claim_index, candidates, claim, store, scope)
            if projection.targets and on_progress is not None:
                on_progress()
            decision = active_ext.consolidate_claim_with_projection(
                claim.content,
                candidates,
                context=(SemanticClaimContext if semantic_mode else ClaimConsolidationContext)(
                    kind=claim.kind,
                    source_role=claim.source_role,
                    observed_at=claim.observed_at,
                    effective_at=claim.effective_at,
                    **(
                        {"meaning": claim.meaning, "witness": claim.semantic_witness}
                        if semantic_mode
                        else {}
                    ),
                ),
                projection=projection,
            )
            projection.validate_decision(decision)
            decisions.append(decision)
            projections.append(projection)
            projection_state.apply(
                claim_index,
                claim,
                decision,
                claim.frame_source or frame_source,
                event=scoped_event,
                cfg=eff_cfg,
                verdict=verdicts[claim_index],
                store=store,
                scope=scope,
            )
            continue
        projections.append(None)
        if active_ext is not None:
            if candidates:
                if on_progress is not None:
                    on_progress()
                if isinstance(active_ext, ContextualConsolidation):
                    decisions.append(
                        active_ext.consolidate_claim_with_context(
                            claim.content,
                            candidates,
                            context=ClaimConsolidationContext(
                                kind=claim.kind,
                                source_role=claim.source_role,
                                observed_at=claim.observed_at,
                                effective_at=claim.effective_at,
                            ),
                        )
                    )
                else:
                    decisions.append(active_ext.consolidate_claim(claim.content, candidates))
            else:
                decisions.append(ConsolidationDecision(action="ADD"))
        else:
            decisions.append(ConsolidationDecision(action="ADD"))

    plan = PreparedSourceEvent(
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
        projections=projections,
        candidate_revisions=candidate_revisions,
        temporal_revisions=temporal_revisions,
        projection_generation=projection_generation,
        semantic_inputs=tuple(claims) if semantic_mode else None,
        semantic_revisions=(
            dict(semantic_projection_state.revisions)
            if semantic_projection_state is not None
            else {}
        ),
    )
    if getattr(eff_cfg.write, "semantic_reuse_policy", None) is not None:
        from gluomem.write.receipt_coherence import finish_support

        plan.receipt_policy = eff_cfg.write.semantic_reuse_policy
        plan.receipt_support = finish_support(
            projection_state,
            claims,
            event=scoped_event,
            cfg=eff_cfg,
            verdicts=verdicts,
            store=store,
            scope=scope,
        )
    if eff_cfg.write.semantic_producer_contract is not None:
        from gluomem.write.observed_identity import producer_binding

        plan.producer_inputs = producer_binding(plan)
    return plan


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
    from gluomem.frames.generation import bump_scope_generation, current_scope_generation
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
    temporal = TemporalStore(store)
    owned_generation_bumps = 0
    publications: dict[int, Frame] = {}
    publication_temporal: dict[int, TemporalRevision] = {}
    frozen_frames: dict[int, str] = {}
    owned_frames: dict[int, Frame] = {}
    publication_expected: dict[int, Frame] = {}
    deferred_payloads: list[Frame] = []
    semantic_guard = None
    if prepared_event.semantic_inputs is not None:
        from gluomem.write.semantic_commit import SemanticCommitGuard

        if eff_cfg.write.semantic_producer_contract is not None:
            from gluomem.write.observed_identity import ObservedCommitGuard

            semantic_guard = ObservedCommitGuard(prepared_event, store)
        else:
            semantic_guard = SemanticCommitGuard(prepared_event, store)
    elif (
        eff_cfg.write.writer_contract == "semantic-state-writer-v4"
        and prepared_event.early_duplicate is None
    ):
        raise ValueError("Semantic input binding changed or is missing")

    def check_projection_state():
        if semantic_guard is not None:
            semantic_guard.check()
        generation = prepared_event.projection_generation
        if generation is None:
            return
        if current_scope_generation(backend, scope) != generation + owned_generation_bumps:
            raise ValueError("Unowned consolidation projection generation change")
        for fid, expected in publication_expected.items():
            if _frame_revision(store.get_frame(fid, scope=scope)) != _frame_revision(expected):
                raise ValueError("Consolidation publication changed after an earlier claim")
        for expected in publication_temporal.values():
            current = temporal.revision(expected.frame_id, scope=scope)
            if current.metadata != expected.metadata:
                raise ValueError("Consolidation publication changed temporal metadata")

    def checked_target(decision, revisions, claim):
        if semantic_guard is not None:
            semantic_guard.require_target(decision, claim)
        return _consolidation_target(decision, revisions, store=store, scope=scope)

    def resolve_decision(position, decision, revisions, temporal_revisions):
        if not isinstance(decision, ProjectedConsolidationDecision):
            return decision, revisions, temporal_revisions
        projection = prepared_event.projections[position]
        if projection is None:
            raise ValueError("Projected decision requires its observed projection")
        projection.validate_decision(decision)
        target_id = decision.target_frame_id
        if target_id in owned_frames and revisions.get(target_id) == frozen_frames[target_id]:
            # Only explicitly named gate flags/supersedes writes below may
            # advance an external observation. The frozen map is never changed.
            digest = _frame_revision(owned_frames[target_id])
            revisions = {**revisions, target_id: digest}
            if target_id in temporal_revisions:
                temporal_revisions = {
                    **temporal_revisions,
                    target_id: replace(
                        temporal_revisions[target_id],
                        digest=digest,
                    ),
                }
        if decision.target_claim_index is not None:
            publication = publications.get(decision.target_claim_index)
            if publication is None:
                raise ValueError("Publication reference did not produce a new eligible I-frame")
            target_id = publication.id
            revisions = {target_id: _frame_revision(publication)}
            temporal_revisions = (
                {target_id: publication_temporal[decision.target_claim_index]}
                if decision.target_claim_index in publication_temporal
                else {}
            )
        return (
            ConsolidationDecision(
                action=decision.action,
                target_frame_id=target_id,
                rationale=decision.rationale,
            ),
            revisions,
            temporal_revisions,
        )

    with _event_write_transaction(store, scope, event.source_ref):
        assert_source_not_retracted(backend, scope, event.source_ref, lock=True)
        if prepared_event.early_duplicate is not None:
            if write_guard is not None:
                write_guard()
            assert_source_not_retracted(backend, scope, event.source_ref)
            existing = backend.fetchone(lookup_sql, lookup_params)
            if existing is None:
                raise EventConflictError("Prepared duplicate receipt no longer exists")
            return _duplicate_result(
                existing, event=event, scope=scope, payload_hash=payload_hash, store=store
            )
        if any(isinstance(d, ProjectedConsolidationDecision) for d in decisions):
            check_projection_state()
            for observations in prepared_event.candidate_revisions:
                for fid, digest in observations.items():
                    current = store.get_frame(fid, scope=scope)
                    if _frame_revision(current) != digest:
                        raise ValueError(
                            "Stale consolidation target changed; prepare ingestion again"
                        )
                    frozen_frames[fid] = digest
                    owned_frames[fid] = current
        for position, (claim, decision, revisions, temporal_revisions) in enumerate(
            zip(
                claims,
                decisions,
                prepared_event.candidate_revisions,
                prepared_event.temporal_revisions,
                strict=True,
            )
        ):
            if isinstance(decision, ProjectedConsolidationDecision):
                projection = prepared_event.projections[position]
                if projection is None:
                    raise ValueError("Projected decision requires its observed projection")
                projection.validate_decision(decision)
                if decision.target_claim_index is not None:
                    continue
                decision, revisions, temporal_revisions = resolve_decision(
                    position,
                    decision,
                    revisions,
                    temporal_revisions,
                )
            checked_target(decision, revisions, claim)
            if decision.action == "TRANSITION":
                _transition_revision(
                    decision,
                    claim,
                    temporal_revisions,
                    temporal=temporal,
                    scope=scope,
                )
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
        assert_source_not_retracted(backend, scope, event.source_ref)
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
        staged_support = {}
        consumed_support = {}
        if prepared_event.receipt_policy is not None:
            from gluomem.write.receipt_coherence import attach_prelude

            staged_support = attach_prelude(
                prepared_event, store, semantic_guard, result.event_row_id, now_s
            )
            for command in prepared_event.receipt_support:
                supported = store.get_frame(command.frame_id, scope=scope)
                if _frame_revision(supported) != command.frame_revision:
                    raise ValueError("Current-source corroboration frame changed before admission")
                # Freeze implicit reuse too, even when retrieval omitted this row.
                # This revision shadow creates NO publication handle.
                frozen_frames[supported.id] = command.frame_revision
                owned_frames[supported.id] = supported
                publication_expected[supported.id] = supported
        admission_rejected = False
        for position, (
            claim,
            index,
            verdict,
            cognify_res,
            decision,
            revisions,
            temporal_revisions,
        ) in enumerate(
            zip(
                claims,
                prepared,
                verdicts,
                cognified,
                decisions,
                prepared_event.candidate_revisions,
                prepared_event.temporal_revisions,
                strict=True,
            )
        ):
            check_projection_state()
            decision, revisions, temporal_revisions = resolve_decision(
                position,
                decision,
                revisions,
                temporal_revisions,
            )
            frame_source = claim.frame_source or prepared_event.frame_source
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

            target_frame = checked_target(decision, revisions, claim)

            if decision.action == "NOOP" and target_frame is not None:
                provenance_row = backend.fetchone(
                    "INSERT INTO memory_provenance (tenant_id, principal_id, namespace, "
                    "frame_id, source_event_id, "
                    "source_type, source_ref, message_id, message_role, observed_at, "
                    "verbatim_span, created_at) VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
                    (
                        *scope.as_tuple(),
                        target_frame.id,
                        result.event_row_id,
                        event.source_type,
                        event.source_ref,
                        claim.message_id,
                        claim.source_role,
                        claim.observed_at,
                        claim.verbatim_span or claim.content,
                        now_s,
                    ),
                )
                if eff_cfg.write.writer_contract == "semantic-state-writer-v4":
                    assert semantic_guard is not None and provenance_row is not None
                    semantic_guard.attach(int(provenance_row["id"]), target_frame.id, claim)
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
            admission_gop = (
                target_frame.gop_id
                if decision.action == "REFINE" and target_frame
                else _gop_for_source_ref(event.source_ref)
            )
            before_admission = store.find_frame_by_hash(
                admission_gop if decision.action == "REFINE" else None,
                claim.content,
                scope=scope,
            )
            if before_admission is not None:
                expected_before = publication_expected.get(before_admission.id) or owned_frames.get(
                    before_admission.id
                )
                if expected_before is not None and _frame_revision(
                    before_admission
                ) != _frame_revision(expected_before):
                    raise ValueError("Consolidation publication changed before reuse")
            admitted = admit(
                claim.content,
                gop=admission_gop,
                scanner=scanner,
                store=store,
                kg=kg,
                cfg=eff_cfg,
                frame_type="P" if decision.action == "REFINE" else "I",
                base_id=target_frame.id if decision.action == "REFINE" and target_frame else None,
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
                scope_wide=decision.action != "REFINE",
                prepared=index,
                verdict=verdict,
                **({"semantic_meaning": claim.meaning} if semantic_guard is not None else {}),
            )
            assert_source_not_retracted(backend, scope, event.source_ref)
            if semantic_guard is not None:
                semantic_guard.check()
            if admitted.frame is None:
                checked_target(decision, revisions, claim)
                check_projection_state()
                admission_rejected = True
                continue
            frame = admitted.frame
            if admitted.deduped and before_admission is not None:
                expected_reuse = replace(before_admission, **admitted.policy_updates)
                if frame.id != expected_reuse.id or _frame_revision(frame) != _frame_revision(
                    expected_reuse
                ):
                    raise ValueError("Consolidation reused publication changed during admission")
            if admitted.policy_updates:
                owned_generation_bumps += 1  # Gate's explicit authorized flag update.

            if decision.action == "TRANSITION" and (
                frame.frame_type != "I"
                or frame.base_frame_id is not None
                or admitted.deduped
                or admitted.semantic_deduped
                or not is_recall_eligible(frame)
            ):
                raise ValueError("TRANSITION requires a new eligible independent I-frame")

            if decision.action == "REFINE" and (
                target_frame is None
                or frame.frame_type != "P"
                or frame.base_frame_id != target_frame.id
                or frame.gop_id != target_frame.gop_id
            ):
                raise ValueError("REFINE admission did not preserve the selected I target")

            if target_frame is not None:
                expected = target_frame
                if frame.id == target_frame.id and admitted.deduped:
                    expected = replace(target_frame, **admitted.policy_updates)
                current_target = store.get_frame(target_frame.id, scope=scope)
                if _frame_revision(current_target) != _frame_revision(
                    expected
                ) or not is_recall_eligible(current_target):
                    raise ValueError("Stale consolidation target changed during admission")
                target_frame = current_target

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
                checked_target(decision, revisions, claim)
                binding = semantic_guard.expected.get(target_frame.id) if semantic_guard else None
                semantic_event_ids = (
                    tuple(r.source_event_row_id for r in binding.receipts) if binding else ()
                )
                selected = decisions[position]
                earlier = (
                    publications.get(selected.target_claim_index)
                    if isinstance(selected, ProjectedConsolidationDecision)
                    and selected.target_claim_index is not None
                    else None
                )
                provenance = backend.fetchall(
                    f"SELECT source_event_id FROM memory_provenance WHERE {scope_sql()} "
                    "AND frame_id=?",
                    (*scope.as_tuple(), target_frame.id),
                )
                defer_current_source = (
                    semantic_guard is not None
                    and earlier is not None
                    and earlier.id == target_frame.id
                    and bool(semantic_event_ids)
                    and all(eid == result.event_row_id for eid in semantic_event_ids)
                    and bool(provenance)
                    and all(p["source_event_id"] == result.event_row_id for p in provenance)
                )
                _scrub_frames(
                    store,
                    [target_frame],
                    scope=scope,
                    status="superseded",
                    deferred_payloads=deferred_payloads if defer_current_source else None,
                )
                if semantic_guard is not None:
                    semantic_guard.scrubbed(
                        (target_frame.id,), () if defer_current_source else semantic_event_ids
                    )
                store._update(target_frame.id, scope, superseded_by=frame.id)
                store._update(frame.id, scope, supersedes=target_frame.id)
                bump_scope_generation(backend, scope)
                owned_generation_bumps += 1
                frame = replace(frame, supersedes=target_frame.id)
                if target_frame.id in publication_expected:
                    publication_expected[target_frame.id] = replace(
                        publication_expected[target_frame.id],
                        content="[forgotten]",
                        status="superseded",
                        pinned=False,
                        superseded_by=frame.id,
                    )

            if cognify_res is not None and kg is not None and is_recall_eligible(frame):
                resolved = resolve_cognify_result(
                    cognify_res,
                    kg=kg,
                    cfg=eff_cfg,
                    valid_from=frame.event_time,
                    scope=scope,
                )
                link_and_retire_resolved(resolved, frame.id, kg=kg, cfg=eff_cfg, scope=scope)
            assert_source_not_retracted(backend, scope, event.source_ref)
            if decision.action == "REFINE":
                checked_target(decision, revisions, claim)
            published_metadata = None
            if decision.action == "TRANSITION":
                expected = _transition_revision(
                    decision,
                    claim,
                    temporal_revisions,
                    temporal=temporal,
                    scope=scope,
                    owned_generation_bumps=owned_generation_bumps,
                )
                assert claim.observed_at is not None  # Validated against the trusted claim above.
                published_metadata = temporal.record_transition(
                    expected,
                    frame.id,
                    TemporalObservation(
                        observed_at=claim.observed_at,
                        effective_at=claim.effective_at,
                    ),
                    scope=scope,
                )
                owned_generation_bumps += 1
            elif (
                event.capture_policy is not None
                and claim.observed_at is not None
                and frame.frame_type == "I"
                and is_recall_eligible(frame)
                and not (admitted.deduped or admitted.semantic_deduped)
            ):
                published_metadata = temporal.record_observation(
                    frame.id,
                    TemporalObservation(
                        observed_at=claim.observed_at,
                        effective_at=claim.effective_at,
                    ),
                    scope=scope,
                )
                owned_generation_bumps += 1
            current_publication = store.get_frame(frame.id, scope=scope)
            if _frame_revision(current_publication) != _frame_revision(frame):
                raise ValueError("Consolidation publication changed before provenance")
            if position in staged_support:
                if (
                    frame.id != staged_support[position]
                    or not admitted.deduped
                    or decision.action != "ADD"
                ):
                    raise ValueError("Current-source corroboration admission changed")
                consumed_support[position] = frame.id
            else:
                from gluomem.write.receipt_coherence import insert_provenance

                provenance_id = insert_provenance(
                    backend,
                    scope=scope,
                    frame_id=frame.id,
                    event_row_id=result.event_row_id,
                    event=event,
                    claim=claim,
                    now_s=now_s,
                )
                if eff_cfg.write.writer_contract == "semantic-state-writer-v4":
                    assert semantic_guard is not None
                    semantic_guard.attach(provenance_id, frame.id, claim)
            if frame.id not in {f.id for f in result.frames}:
                result.frames.append(frame)
            if frame.id in owned_frames:
                # Exact returned frame checked above; no arbitrary revision reread.
                owned_frames[frame.id] = frame
            if frame.id in publication_expected:
                publication_expected[frame.id] = frame
                for origin, publication in publications.items():
                    if publication.id == frame.id:
                        publications[origin] = frame
                        publication_temporal[origin] = replace(
                            publication_temporal[origin],
                            digest=_frame_revision(frame),
                        )
            if (
                frame.frame_type == "I"
                and frame.base_frame_id is None
                and not (admitted.deduped or admitted.semantic_deduped)
                and is_recall_eligible(frame)
                and (semantic_guard is None or claim.meaning.independently_versioned)
            ):
                publications[position] = frame
                revision = temporal.revision(frame.id, scope=scope)
                if (
                    revision.digest != _frame_revision(frame)
                    or revision.metadata != published_metadata
                ):
                    raise ValueError("Consolidation publication changed before reference admission")
                publication_temporal[position] = replace(
                    revision,
                    generation=(
                        prepared_event.projection_generation
                        if prepared_event.projection_generation is not None
                        else revision.generation - owned_generation_bumps
                    ),
                )
                publication_expected[frame.id] = frame
            check_projection_state()
        if deferred_payloads:
            from gluomem.frames.lifecycle import _redact_payloads

            # Validate BEFORE revocation; deleting receipts must not hide an
            # unowned callback mutation. Explicit erasure keeps its immediate path.
            check_projection_state()
            assert_source_not_retracted(backend, scope, event.source_ref)
            for pending in deferred_payloads:
                provenance = backend.fetchall(
                    f"SELECT source_event_id FROM memory_provenance WHERE {scope_sql()} "
                    "AND frame_id=?",
                    (*scope.as_tuple(), pending.id),
                )
                if not provenance or any(
                    p["source_event_id"] != result.event_row_id for p in provenance
                ):
                    raise ValueError("Deferred supersession source ownership changed")
            _redact_payloads(store, deferred_payloads, scope=scope, preserve_siblings=True)
            assert semantic_guard is not None  # Only semantic supersession defers its payload.
            semantic_guard.scrubbed(event_ids=(result.event_row_id,))
        # Suppressed/rejected envelopes are receipts, not a second plaintext store.
        # Accepted claims may share a frame through NOOP or admission reuse.
        if result.suppressed_hashes or admission_rejected:
            redact_event_payload(
                backend,
                result.event_row_id,
                scope=scope,
                event_id=event.event_id,
                revision=event.revision,
            )
            if semantic_guard is not None:
                semantic_guard.scrubbed(event_ids=(result.event_row_id,))
        for position, (claim, decision, revisions, temporal_revisions) in enumerate(
            zip(
                claims,
                decisions,
                prepared_event.candidate_revisions,
                prepared_event.temporal_revisions,
                strict=True,
            )
        ):
            decision, revisions, temporal_revisions = resolve_decision(
                position,
                decision,
                revisions,
                temporal_revisions,
            )
            if decision.action == "TRANSITION":
                if semantic_guard is not None:
                    semantic_guard.require_target(decision, claim)
                _transition_revision(
                    decision,
                    claim,
                    temporal_revisions,
                    temporal=temporal,
                    scope=scope,
                    owned_generation_bumps=owned_generation_bumps,
                )
        check_projection_state()
        if prepared_event.receipt_policy is not None:
            from gluomem.write.receipt_coherence import require_staged

            require_staged(prepared_event, consumed_support, semantic_guard)
        current_frames = store.get_frames([frame.id for frame in result.frames], scope=scope)
        result.frames = [
            current_frames[frame.id]
            for frame in result.frames
            if frame.id in current_frames
            and current_frames[frame.id].status in ("active", "candidate")
        ]
        result.provenance = get_frames_provenance(
            backend, [f.id for f in result.frames], scope=scope
        )
    return result


def attempt_commit_source_event(
    prepared_event: PreparedSourceEvent,
    *,
    store: FrameStore,
    kg: KnowledgeGraph | None = None,
    write_guard: Callable[[], None] | None = None,
) -> EventCommitOutcome:
    """Return a nonthrowing commit outcome for a transaction-owning caller.

    Event publication still rolls back on refusal. Successful reentrant erasure
    replay remains in the owner's transaction; the owner must commit that
    transaction normally, skip success side effects, then raise/handle the
    original outcome error. This does not protect a generic enclosing rollback.
    Process-control BaseExceptions and transaction commit failures are not
    converted into successful publications or erasure durability guarantees.
    """
    try:
        return EventCommitOutcome(
            result=commit_source_event(
                prepared_event,
                store=store,
                kg=kg,
                write_guard=write_guard,
            )
        )
    except Exception as exc:
        return EventCommitOutcome(error=exc)


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
