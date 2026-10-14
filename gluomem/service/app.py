"""Authenticated FastAPI service exposing GluoMem v1 HTTP endpoints."""

import time
from dataclasses import asdict
from typing import Annotated, Any

from gluomem.config.app_config import AppConfig
from gluomem.embedding.compat import stored_fingerprint
from gluomem.extras import require
from gluomem.frames.lifecycle import (
    correct_memory,
    export_memories,
    forget_memory,
    list_memories,
    retract_source,
    version_lineage,
)
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.hybrid import HybridSearch
from gluomem.service.auth import (
    HEADER_NAMESPACE,
    HEADER_PRINCIPAL,
    HEADER_TENANT,
    resolve_service_key,
    scope_from_headers,
    verify_bearer_header,
)
from gluomem.service.jobs import BackpressureError, JobQueue, MemoryJob
from gluomem.service.schemas import (
    CorrectRequest,
    CorrectResponse,
    ForgetResponse,
    IngestEventRequest,
    IngestEventResponse,
    JobResponse,
    ListMemoriesResponse,
    MemoryFrameModel,
    RecallHitModel,
    RecallRequest,
    RecallResponse,
    RetractSourceRequest,
    RetractSourceResponse,
    ReviewRequest,
    ReviewResponse,
    scope_to_model,
)
from gluomem.storage.db import MindDB
from gluomem.storage.scope import MemoryScope
from gluomem.write.events import (
    SourceEvent,
    SourceEventMessage,
    get_frame_provenance,
    ingest_source_event,
    review_memory,
)


def _job_to_response(job: MemoryJob) -> JobResponse:
    return JobResponse(
        id=job.id,
        scope=scope_to_model(job.scope),
        job_type=job.job_type,
        dedup_key=job.dedup_key,
        status=job.status,
        attempt_count=job.attempt_count,
        max_attempts=job.max_attempts,
        scope_generation=job.scope_generation,
        llm_calls_used=job.llm_calls_used,
        max_llm_calls=job.max_llm_calls,
        tokens_used=job.tokens_used,
        max_tokens=job.max_tokens,
        last_error=job.last_error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def create_app(
    db: MindDB,
    store: FrameStore,
    *,
    cfg: AppConfig | None = None,
    service_key: str | None = None,
    search: HybridSearch | None = None,
    kg: KnowledgeGraph | None = None,
    queue: JobQueue | None = None,
) -> Any:
    """Build the authenticated FastAPI application for GluoMem."""
    require("service", "GluoMem HTTP service")
    from fastapi import Depends, FastAPI, Header, HTTPException, Response
    from fastapi.responses import JSONResponse

    eff_cfg = cfg if cfg is not None else AppConfig()
    eff_key = resolve_service_key(service_key)
    eff_kg = kg if kg is not None else KnowledgeGraph(db, scope=store.scope)
    eff_search = (
        search
        if search is not None
        else HybridSearch(
            db,
            store,
            eff_cfg,
            embedder=store._embedder,
            kg=eff_kg,
            scope=store.scope,
        )
    )
    eff_queue = queue if queue is not None else JobQueue(db.backend)

    app = FastAPI(
        title="GluoMem Service",
        version="0.1.0",
        description="Scoped, provenance-first memory service for KVARK and agent runtimes.",
    )

    def _require_scope(
        authorization: Annotated[str | None, Header()] = None,
        x_gluomem_tenant: Annotated[
            str | None, Header(alias=HEADER_TENANT)
        ] = None,
        x_gluomem_principal: Annotated[
            str | None, Header(alias=HEADER_PRINCIPAL)
        ] = None,
        x_gluomem_namespace: Annotated[
            str | None, Header(alias=HEADER_NAMESPACE)
        ] = None,
    ) -> MemoryScope:
        verify_bearer_header(authorization, eff_key)
        return scope_from_headers(
            x_gluomem_tenant, x_gluomem_principal, x_gluomem_namespace
        )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        row = db.backend.fetchone("SELECT 1 AS ok")
        if row is None or int(row["ok"]) != 1:
            raise HTTPException(status_code=503, detail="Database probe failed")
        fp = stored_fingerprint(db)
        return {
            "status": "ready",
            "dialect": db.backend.dialect,
            "embedder_fingerprint": fp.to_json() if fp is not None else None,
            "queue": eff_queue.queue_stats(),
        }

    @app.post("/v1/events", response_model=None)
    def post_event(
        req: IngestEventRequest,
        response: Response,
        scope: Annotated[MemoryScope, Depends(_require_scope)],
    ) -> Any:
        if req.async_mode:
            dedup_key = f"{req.event_id}:{req.revision}"
            payload = req.model_dump(exclude={"async_mode"})
            try:
                job = eff_queue.enqueue(
                    scope=scope,
                    job_type="ingest_event",
                    dedup_key=dedup_key,
                    payload=payload,
                )
            except BackpressureError as exc:
                raise HTTPException(status_code=429, detail=str(exc)) from exc
            response.status_code = 202
            return _job_to_response(job)

        evt = SourceEvent(
            event_id=req.event_id,
            revision=req.revision,
            source_type=req.source_type,
            source_ref=req.source_ref,
            scope=scope,
            messages=[
                SourceEventMessage(
                    message_id=m.message_id,
                    role=m.role,
                    content=m.content,
                    timestamp=m.timestamp,
                )
                for m in req.messages
            ],
            content=req.content,
            kind=req.kind,
            confidence=req.confidence,
            require_review=req.require_review,
            pinned=req.pinned,
            event_time=req.event_time,
            created_at=req.created_at,
            force_reassert=req.force_reassert,
        )
        try:
            res = ingest_source_event(evt, store=store, kg=eff_kg, cfg=eff_cfg)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        frame_models = [
            MemoryFrameModel.from_frame(
                f, get_frame_provenance(db.backend, f.id, scope=scope)
            )
            for f in res.frames
        ]
        return IngestEventResponse(
            source_event_row_id=res.event_row_id,
            event_id=res.event_id,
            revision=res.revision,
            deduped_event=res.deduplicated,
            frames=frame_models,
            suppressed_hashes=res.suppressed_hashes,
        )

    @app.post("/v1/memories:recall", response_model=RecallResponse)
    def post_recall(
        req: RecallRequest,
        scope: Annotated[MemoryScope, Depends(_require_scope)],
    ) -> RecallResponse:
        t0 = time.perf_counter()
        res = eff_search.search(
            req.query,
            limit=req.k,
            profile=req.profile,
            since=req.since,
            until=req.until,
            scope=scope,
            now_iso=req.now_iso,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        hit_models = [
            RecallHitModel(
                frame=MemoryFrameModel.from_frame(
                    h.frame,
                    get_frame_provenance(db.backend, h.frame.id, scope=scope),
                ),
                rrf=h.rrf,
                relevance=h.relevance,
                fusion=h.fusion,
                rerank=h.rerank,
                final=h.final,
            )
            for h in res.hits
        ]
        return RecallResponse(
            hits=hit_models,
            abstained=res.abstained,
            abstention=asdict(res.abstention) if res.abstention is not None else {},
            latency_ms=elapsed_ms,
        )

    @app.get("/v1/memories", response_model=ListMemoriesResponse)
    def get_memories(
        scope: Annotated[MemoryScope, Depends(_require_scope)],
        include_review_queue: bool = True,
    ) -> ListMemoriesResponse:
        grouped = list_memories(store, scope=scope)

        def _convert(f: Any) -> MemoryFrameModel:
            return MemoryFrameModel.from_frame(
                f, get_frame_provenance(db.backend, f.id, scope=scope)
            )

        return ListMemoriesResponse(
            scope=scope_to_model(scope),
            active_frames=[_convert(f) for f in grouped.active_frames],
            by_kind={
                k: [_convert(f) for f in frames]
                for k, frames in grouped.by_kind.items()
            },
            by_source_ref={
                s: [_convert(f) for f in frames]
                for s, frames in grouped.by_source_ref.items()
            },
            review_queue=(
                [_convert(f) for f in grouped.review_queue]
                if include_review_queue
                else []
            ),
        )

    @app.get("/v1/memories/{frame_id}/lineage")
    def get_memory_lineage(
        frame_id: int,
        scope: Annotated[MemoryScope, Depends(_require_scope)],
    ) -> list[MemoryFrameModel]:
        try:
            chain = version_lineage(store, frame_id, scope=scope)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return [
            MemoryFrameModel.from_frame(
                f, get_frame_provenance(db.backend, f.id, scope=scope)
            )
            for f in chain
        ]

    @app.post("/v1/memories/{frame_id}:review", response_model=ReviewResponse)
    def post_review_memory(
        frame_id: int,
        req: ReviewRequest,
        scope: Annotated[MemoryScope, Depends(_require_scope)],
    ) -> ReviewResponse:
        try:
            updated = review_memory(
                store,
                frame_id,
                action=req.action,
                edited_content=req.edited_content,
                scope=scope,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return ReviewResponse(
            action=req.action,
            frame=MemoryFrameModel.from_frame(
                updated, get_frame_provenance(db.backend, updated.id, scope=scope)
            ),
        )

    @app.post("/v1/memories/{frame_id}:correct", response_model=CorrectResponse)
    def post_correct_memory(
        frame_id: int,
        req: CorrectRequest,
        scope: Annotated[MemoryScope, Depends(_require_scope)],
    ) -> CorrectResponse:
        try:
            res = correct_memory(
                store,
                frame_id,
                req.new_content,
                scope=scope,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return CorrectResponse(
            superseded_frame=MemoryFrameModel.from_frame(
                res.superseded_frame,
                get_frame_provenance(
                    db.backend, res.superseded_frame.id, scope=scope
                ),
            ),
            replacement_frame=MemoryFrameModel.from_frame(
                res.replacement_frame,
                get_frame_provenance(
                    db.backend, res.replacement_frame.id, scope=scope
                ),
            ),
            scope_generation=res.scope_generation,
        )

    @app.delete("/v1/memories/{frame_id}", response_model=ForgetResponse)
    def delete_memory(
        frame_id: int,
        scope: Annotated[MemoryScope, Depends(_require_scope)],
    ) -> ForgetResponse:
        try:
            res = forget_memory(store, frame_id, scope=scope)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ForgetResponse(
            frame_id=res.frame_id,
            content_hash=res.content_hash,
            reason=res.reason,
            retired_entity_ids=res.retired_entity_ids,
            retired_relation_ids=res.retired_relation_ids,
            scope_generation=res.scope_generation,
        )

    @app.post("/v1/sources:retract", response_model=RetractSourceResponse)
    def post_retract_source(
        req: RetractSourceRequest,
        scope: Annotated[MemoryScope, Depends(_require_scope)],
    ) -> RetractSourceResponse:
        try:
            res = retract_source(store, req.source_ref, scope=scope)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RetractSourceResponse(
            source_ref=res.source_ref,
            forgotten_frame_ids=res.forgotten_frame_ids,
            retained_frame_ids=res.retained_frame_ids,
            scope_generation=res.scope_generation,
        )

    @app.post("/v1/memories:export")
    def post_export_memories(
        scope: Annotated[MemoryScope, Depends(_require_scope)],
    ) -> JSONResponse:
        payload = export_memories(store, scope=scope)
        return JSONResponse(content=payload)

    @app.get("/v1/jobs/{job_id}", response_model=JobResponse)
    def get_job_status(
        job_id: int,
        scope: Annotated[MemoryScope, Depends(_require_scope)],
    ) -> JobResponse:
        try:
            job = eff_queue.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if job.scope != scope:
            raise HTTPException(
                status_code=404, detail=f"no memory_job with id={job_id} in scope"
            )
        return _job_to_response(job)

    return app
