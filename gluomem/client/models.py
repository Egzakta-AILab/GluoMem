"""Pydantic-only wire types shared by HTTP transports and the service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

HEADER_TENANT = "X-GluoMem-Tenant"
HEADER_PRINCIPAL = "X-GluoMem-Principal"
HEADER_NAMESPACE = "X-GluoMem-Namespace"
MemoryKind = Literal[
    "fact", "profile", "preference", "decision", "procedure", "episode", "correction", "summary"
]
FrameStatus = Literal["candidate", "active", "superseded", "expired", "forgotten", "tombstoned"]
SourceType = Literal[
    "chat_turn",
    "document",
    "doc_chunk",
    "explicit_memory",
    "tool_result",
    "tool_output",
    "import",
]


class _Resp(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)


class ScopeModel(_Resp):
    tenant_id: str = "local"
    principal_id: str = "default"
    namespace: str = "personal"


class ProvenanceModel(_Resp):
    id: int
    source_event_id: int | None
    source_type: str
    source_ref: str
    message_id: str | None
    verbatim_span: str | None
    created_at: str


class MemoryFrameModel(_Resp):
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


class IngestEventResponse(_Resp):
    source_event_row_id: int
    event_id: str
    revision: int
    deduped_event: bool
    frames: list[MemoryFrameModel]
    suppressed_hashes: list[str]


class RecallHitModel(_Resp):
    frame: MemoryFrameModel
    rrf: float
    relevance: float
    fusion: float
    rerank: float | None
    final: float


class RecallResponse(_Resp):
    hits: list[RecallHitModel]
    abstained: bool
    abstention: dict[str, Any]
    latency_ms: float


class ListMemoriesResponse(_Resp):
    scope: ScopeModel
    active_frames: list[MemoryFrameModel]
    by_kind: dict[str, list[int]]
    by_source_ref: dict[str, list[int]]
    review_queue: list[MemoryFrameModel]
    next_cursor: int | None = None


class ReviewResponse(_Resp):
    action: Literal["accept", "edit", "dismiss"]
    frame: MemoryFrameModel | None


class CorrectResponse(_Resp):
    superseded_frame: MemoryFrameModel
    replacement_frame: MemoryFrameModel
    scope_generation: int


class ForgetResponse(_Resp):
    frame_id: int
    content_hash: str
    reason: Literal["forgotten", "retracted", "dismissed"]
    retired_entity_ids: list[int]
    retired_relation_ids: list[int]
    scope_generation: int


class RetractSourceResponse(_Resp):
    source_ref: str
    forgotten_frame_ids: list[int]
    retained_frame_ids: list[int]
    scope_generation: int


class SweepExpiredResponse(_Resp):
    scope: ScopeModel
    swept_frame_ids: list[int]
    retired_entity_ids: list[int]
    retired_relation_ids: list[int]
    scope_generation: int


class JobResponse(_Resp):
    id: int
    scope: ScopeModel
    job_type: str
    dedup_key: str
    status: str
    attempt_count: int
    max_attempts: int
    scope_generation: int
    last_error: str | None
    created_at: str
    updated_at: str
