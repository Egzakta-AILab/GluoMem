"""Pydantic v2 request and response schemas for the GluoMem HTTP service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from gluomem.frames.store import Frame
from gluomem.storage.scope import FrameStatus, MemoryKind, MemoryScope
from gluomem.write.events import ProvenanceRecord, SourceType


class ScopeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    principal_id: str
    namespace: str


class ProvenanceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    source_event_id: int | None
    source_type: str
    source_ref: str
    message_id: str | None
    verbatim_span: str | None
    created_at: str

    @classmethod
    def from_record(cls, rec: ProvenanceRecord) -> ProvenanceModel:
        return cls(
            id=rec.id,
            source_event_id=rec.source_event_id,
            source_type=rec.source_type,
            source_ref=rec.source_ref,
            message_id=rec.message_id,
            verbatim_span=rec.verbatim_span,
            created_at=rec.created_at,
        )


class MemoryFrameModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    frame_type: Literal["I", "P", "B"]
    gop_id: str
    t: int
    base_frame_id: int | None
    content: str
    importance: str
    source: str
    trust: str
    access_count: int
    created_at: str
    event_time: str | None
    last_accessed: str
    scope: ScopeModel
    kind: MemoryKind
    status: FrameStatus
    expires_at: str | None
    pinned: bool
    verified_by_user: bool
    confidence: float
    supersedes: int | None
    superseded_by: int | None
    provenance: list[ProvenanceModel] = Field(default_factory=list)

    @classmethod
    def from_frame(
        cls,
        frame: Frame,
        provenance: list[ProvenanceRecord] | None = None,
    ) -> MemoryFrameModel:
        return cls(
            id=frame.id,
            frame_type=frame.frame_type,
            gop_id=frame.gop_id,
            t=frame.t,
            base_frame_id=frame.base_frame_id,
            content=frame.content,
            importance=frame.importance,
            source=frame.source,
            trust=frame.trust,
            access_count=frame.access_count,
            created_at=frame.created_at,
            event_time=frame.event_time,
            last_accessed=frame.last_accessed,
            scope=ScopeModel(
                tenant_id=frame.scope.tenant_id,
                principal_id=frame.scope.principal_id,
                namespace=frame.scope.namespace,
            ),
            kind=frame.kind,
            status=frame.status,
            expires_at=frame.expires_at,
            pinned=frame.pinned,
            verified_by_user=frame.verified_by_user,
            confidence=frame.confidence,
            supersedes=frame.supersedes,
            superseded_by=frame.superseded_by,
            provenance=[
                ProvenanceModel.from_record(p) for p in (provenance or [])
            ],
        )


class EventMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    role: str = "user"
    content: str
    timestamp: str | None = None


class IngestEventRequest(BaseModel):
    """Request body for ``POST /v1/events`` (strictly forbids body-supplied scope)."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    revision: int = Field(default=1, ge=1)
    source_type: SourceType = "chat_turn"
    source_ref: str
    messages: list[EventMessageRequest] = Field(default_factory=list)
    content: str | None = None
    kind: MemoryKind | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    require_review: bool = False
    pinned: bool = False
    event_time: str | None = None
    created_at: str | None = None
    force_reassert: bool = False
    async_mode: bool = False


class IngestEventResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_event_row_id: int
    event_id: str
    revision: int
    deduped_event: bool
    frames: list[MemoryFrameModel]
    suppressed_hashes: list[str]


class RecallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    k: int = Field(default=10, ge=1, le=100)
    profile: str = "balanced"
    since: str | None = None
    until: str | None = None
    now_iso: str | None = None


class RecallHitModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame: MemoryFrameModel
    rrf: float
    relevance: float
    fusion: float
    rerank: float | None
    final: float


class RecallResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hits: list[RecallHitModel]
    abstained: bool
    abstention: dict[str, Any]
    latency_ms: float


class ListMemoriesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: ScopeModel
    active_frames: list[MemoryFrameModel]
    by_kind: dict[str, list[MemoryFrameModel]]
    by_source_ref: dict[str, list[MemoryFrameModel]]
    review_queue: list[MemoryFrameModel]


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["accept", "edit", "dismiss"]
    edited_content: str | None = None
    kind: MemoryKind | None = None


class ReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["accept", "edit", "dismiss"]
    frame: MemoryFrameModel | None
    tombstoned_hash: str | None = None


class CorrectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new_content: str
    kind: MemoryKind | None = None


class CorrectResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    superseded_frame: MemoryFrameModel
    replacement_frame: MemoryFrameModel
    b_frame: MemoryFrameModel | None = None
    scope_generation: int


class ForgetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_id: int
    content_hash: str
    reason: Literal["forgotten", "retracted", "dismissed"]
    retired_entity_ids: list[int]
    retired_relation_ids: list[int]
    scope_generation: int


class RetractSourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ref: str


class RetractSourceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ref: str
    forgotten_frame_ids: list[int]
    retained_frame_ids: list[int]
    scope_generation: int


class JobResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    scope: ScopeModel
    job_type: str
    dedup_key: str
    status: str
    attempt_count: int
    max_attempts: int
    scope_generation: int
    llm_calls_used: int
    max_llm_calls: int
    tokens_used: int
    max_tokens: int
    last_error: str | None
    created_at: str
    updated_at: str


def scope_to_model(scope: MemoryScope) -> ScopeModel:
    return ScopeModel(
        tenant_id=scope.tenant_id,
        principal_id=scope.principal_id,
        namespace=scope.namespace,
    )
