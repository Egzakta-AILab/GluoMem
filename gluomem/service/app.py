"""Authenticated, request-isolated GluoMem HTTP service."""

import logging
import threading
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, nullcontext
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

if TYPE_CHECKING:
    from gluomem.storage.postgres_backend import PostgresConnectionPool

from gluomem.client.models import (
    CorrectResponse,
    ForgetResponse,
    IngestEventResponse,
    JobResponse,
    ListMemoriesResponse,
    MemoryFrameModel,
    ProvenanceModel,
    RecallHitModel,
    RecallResponse,
    RetractSourceResponse,
    ReviewResponse,
    ScopeModel,
    SweepExpiredResponse,
)
from gluomem.config.app_config import AppConfig
from gluomem.embedding.compat import stored_fingerprint
from gluomem.extras import require
from gluomem.frames.eligibility import is_recall_eligible
from gluomem.frames.expiry import sweep_expired_episodes
from gluomem.frames.lifecycle import (
    correct_memory,
    forget_memory,
    get_scope_generation,
    retract_source,
    version_lineage,
)
from gluomem.frames.store import Frame, FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.llm.outputs import Extractors
from gluomem.mcp.server import McpServer
from gluomem.search.hybrid import HybridSearch
from gluomem.service.auth import (
    HEADER_NAMESPACE,
    HEADER_PRINCIPAL,
    HEADER_TENANT,
    resolve_service_key,
    scope_from_headers,
    verify_bearer_header,
)
from gluomem.service.jobs import BackpressureError, EventConflictError, JobQueue
from gluomem.service.schemas import (
    CorrectRequest,
    IngestEventRequest,
    RecallRequest,
    RetractSourceRequest,
    ReviewRequest,
    SweepExpiredRequest,
)
from gluomem.storage.db import MindDB
from gluomem.storage.scope import DEFAULT_SCOPE, MemoryKind, MemoryScope, eligible_sql, scope_sql
from gluomem.timeutil import now_utc_iso
from gluomem.write.events import ingest_source_event, review_memory

_LOG = logging.getLogger(__name__)


@dataclass
class _Runtime:
    db: MindDB
    store: FrameStore
    kg: KnowledgeGraph
    search: HybridSearch
    queue: JobQueue

    def _frames(self, frames: Sequence[Frame], scope: MemoryScope) -> list[MemoryFrameModel]:
        if not frames:
            return []
        ids = [frame.id for frame in frames]
        rows = self.db.fetchall(
            f"SELECT * FROM memory_provenance WHERE {scope_sql()} "
            f"AND frame_id IN ({','.join('?' for _ in ids)}) ORDER BY id",
            (*scope.as_tuple(), *ids),
        )
        provenance: dict[int, list[ProvenanceModel]] = {}
        for row in rows:
            provenance.setdefault(row["frame_id"], []).append(
                ProvenanceModel.model_validate(dict(row))
            )
        models = [MemoryFrameModel.model_validate(frame) for frame in frames]
        for model in models:
            model.provenance = provenance.get(model.id, [])
        return models


def _groups(
    frames: Sequence[MemoryFrameModel],
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    kinds: dict[str, list[int]] = {}
    refs: dict[str, list[int]] = {}
    for frame in frames:
        kinds.setdefault(frame.kind, []).append(frame.id)
        for ref in sorted({p.source_ref for p in frame.provenance}) or [f"gop:{frame.gop_id}"]:
            refs.setdefault(ref, []).append(frame.id)
    return kinds, refs


def create_app(
    db: MindDB,
    store: FrameStore,
    *,
    cfg: AppConfig | None = None,
    service_key: str | None = None,
    search: HybridSearch | None = None,
    kg: KnowledgeGraph | None = None,
    queue: JobQueue | None = None,
    extractors: Extractors | None = None,
    max_pending_per_scope: int = 50,
    pg_pool: PostgresConnectionPool | None = None,
) -> FastAPI:
    """Use one shared secret; deploy behind a trusted gateway asserting tenant/principal headers."""
    require("service", "GluoMem HTTP service")
    del kg
    cfg = cfg if cfg is not None else AppConfig()
    key = resolve_service_key(service_key)
    if db.dialect == "sqlite" and str(db.path) == ":memory:":
        raise ValueError("HTTP service requires file-backed SQLite or PostgreSQL")
    backend: Any = db.backend
    live: dict[int, MindDB] = {}
    lock = threading.Lock()
    pool_ref: list[PostgresConnectionPool | None] = [pg_pool]
    owns_pool = pg_pool is None

    def _ensure_pg_pool() -> PostgresConnectionPool:
        with lock:
            if pool_ref[0] is None:
                from gluomem.storage.postgres_backend import PostgresConnectionPool as _Pool

                pool_ref[0] = _Pool(
                    backend.dsn,
                    dim=db.dim,
                    postgres_schema=backend.postgres_schema,
                    min_size=cfg.storage.pool_min_size,
                    max_size=cfg.storage.pool_max_size,
                )
            return pool_ref[0]

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if db.dialect == "postgres":
            _app.state.pg_pool = _ensure_pg_pool()
        try:
            yield
        finally:
            with lock:
                for handle in live.values():
                    handle.close()
                live.clear()
                active_pool = pool_ref[0]
                if owns_pool and active_pool is not None:
                    active_pool.close()
                    pool_ref[0] = None

    def runtime() -> Iterator[_Runtime]:
        conn_ctx = _ensure_pg_pool().connection() if db.dialect == "postgres" else nullcontext(None)
        with conn_ctx as pooled_conn:
            if db.dialect == "postgres":
                handle = MindDB(
                    dim=db.dim,
                    backend_type="postgres",
                    dsn=backend.dsn,
                    postgres_schema=backend.postgres_schema,
                    conn=pooled_conn,
                    verify_schema=False,
                    owns_connection=False,
                )
            else:
                handle = MindDB(db.path, dim=db.dim)
            with lock:
                live[id(handle)] = handle
            try:
                local_store = FrameStore(
                    handle,
                    embedder=store.embedder,
                    chunking=store.chunking,
                    scope=store.scope,
                    lifecycle=cfg.lifecycle,
                )
                local_kg = KnowledgeGraph(handle, scope=store.scope)
                local_search = HybridSearch(
                    handle,
                    local_store,
                    cfg,
                    kg=local_kg,
                    reranker=search._reranker if search is not None else None,
                    clock=search._clock if search is not None else None,
                )
                local_queue = JobQueue(
                    handle.backend,
                    max_pending_per_scope=(
                        queue._max_pending_per_scope if queue is not None else max_pending_per_scope
                    ),
                    default_max_attempts=queue._default_max_attempts if queue is not None else 3,
                )
                yield _Runtime(handle, local_store, local_kg, local_search, local_queue)
            finally:
                if handle.in_transaction:
                    handle.rollback()
                with lock:
                    if live.pop(id(handle), None) is not None:
                        handle.close()

    app = FastAPI(title="GluoMem Service", version=version("gluomem"), lifespan=lifespan)

    @app.middleware("http")
    async def log_requests(request: Request, call_next: Any) -> Response:
        started = time.perf_counter()
        tenant = (request.headers.get(HEADER_TENANT) or "").strip()
        principal = (request.headers.get(HEADER_PRINCIPAL) or "").strip()
        namespace = (request.headers.get(HEADER_NAMESPACE) or "").strip() or DEFAULT_SCOPE.namespace
        req_scope: MemoryScope | None = None
        if tenant and principal:
            try:
                req_scope = MemoryScope(
                    tenant_id=tenant,
                    principal_id=principal,
                    namespace=namespace,
                )
            except ValueError:
                req_scope = None
        response: Response = await call_next(request)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        extra: dict[str, Any] = {
            "event": "http_request",
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "scope": req_scope.as_dict() if req_scope is not None else None,
        }
        if req_scope is not None:
            extra["tenant_id"] = req_scope.tenant_id
            extra["principal_id"] = req_scope.principal_id
            extra["namespace"] = req_scope.namespace
        _LOG.info(
            "%s %s -> %s", request.method, request.url.path, response.status_code, extra=extra
        )
        return response

    @app.exception_handler(KeyError)
    async def missing(_request: Request, _error: KeyError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "Memory resource not found"})

    @app.exception_handler(ValueError)
    @app.exception_handler(ValidationError)
    async def invalid(_request: Request, error: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(EventConflictError)
    async def conflict(_request: Request, error: EventConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(BackpressureError)
    async def backpressure(_request: Request, error: BackpressureError) -> JSONResponse:
        return JSONResponse(status_code=429, content={"detail": str(error)})

    def require_scope(
        authorization: Annotated[str | None, Header(max_length=4096)] = None,
        tenant: Annotated[str | None, Header(alias=HEADER_TENANT, max_length=128)] = None,
        principal: Annotated[str | None, Header(alias=HEADER_PRINCIPAL, max_length=128)] = None,
        namespace: Annotated[str | None, Header(alias=HEADER_NAMESPACE, max_length=128)] = None,
    ) -> MemoryScope:
        verify_bearer_header(authorization, key)
        return scope_from_headers(tenant, principal, namespace)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(rt: Annotated[_Runtime, Depends(runtime)]) -> dict[str, str]:
        if rt.db.scalar("SELECT 1") != 1:
            raise HTTPException(status_code=503, detail="Database probe failed")
        return {"status": "ok"}

    @app.get("/v1/status")
    def status(
        scope: Annotated[MemoryScope, Depends(require_scope)],
        rt: Annotated[_Runtime, Depends(runtime)],
    ) -> dict[str, Any]:
        fp = stored_fingerprint(rt.db)
        return {
            "dialect": rt.db.dialect,
            "embedder_fingerprint": fp.to_json() if fp else None,
            "queue": rt.queue.queue_stats(scope),
        }

    @app.post(
        "/v1/events",
        response_model=IngestEventResponse | JobResponse,
        responses={202: {"model": JobResponse}},
    )
    def post_event(
        req: IngestEventRequest,
        response: Response,
        scope: Annotated[MemoryScope, Depends(require_scope)],
        rt: Annotated[_Runtime, Depends(runtime)],
    ) -> IngestEventResponse | JobResponse:
        effective_async = (
            req.async_mode if req.async_mode is not None else (req.source_type != "explicit_memory")
        )
        if effective_async:
            job = rt.queue.enqueue(
                scope,
                "ingest_event",
                f"{req.event_id}:{req.revision}",
                req.model_dump(exclude={"async_mode"}),
            )
            response.status_code = 202
            return JobResponse.model_validate(job)
        result = ingest_source_event(
            req.to_source_event(scope),
            store=rt.store,
            kg=rt.kg,
            cfg=cfg,
            extractors=extractors,
        )
        return IngestEventResponse(
            source_event_row_id=result.event_row_id,
            event_id=result.event_id,
            revision=result.revision,
            deduped_event=result.deduplicated,
            frames=rt._frames(result.frames, scope),
            suppressed_hashes=result.suppressed_hashes,
        )

    @app.post("/v1/memories:recall", response_model=RecallResponse)
    def recall(
        req: RecallRequest,
        scope: Annotated[MemoryScope, Depends(require_scope)],
        rt: Annotated[_Runtime, Depends(runtime)],
    ) -> RecallResponse:
        if req.profile not in cfg.scoring.profiles:
            raise ValueError("Unknown scoring profile")
        started = time.perf_counter()
        result = rt.search.search(
            req.query,
            limit=req.k,
            profile=req.profile,
            since=req.since,
            until=req.until,
            scope=scope,
            now_iso=req.now_iso,
        )
        frames = rt._frames([hit.frame for hit in result.hits], scope)
        hits = [
            RecallHitModel(
                frame=frame,
                rrf=hit.rrf,
                relevance=hit.relevance,
                fusion=hit.fusion,
                rerank=hit.rerank,
                final=hit.final,
            )
            for hit, frame in zip(result.hits, frames, strict=True)
        ]
        return RecallResponse(
            hits=hits,
            abstained=result.abstained,
            abstention=asdict(result.abstention) if result.abstention is not None else {},
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    @app.get("/v1/memories", response_model=ListMemoriesResponse)
    def memories(
        scope: Annotated[MemoryScope, Depends(require_scope)],
        rt: Annotated[_Runtime, Depends(runtime)],
        include_review_queue: bool = True,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
        cursor: Annotated[int, Query(ge=0)] = 0,
        kind: MemoryKind | None = None,
        source_ref: Annotated[str | None, Query(max_length=1024)] = None,
    ) -> ListMemoriesResponse:
        where = f"{scope_sql('m')} AND m.id > ? AND ({eligible_sql('m')}"
        where += " OR m.status = 'candidate')" if include_review_queue else ")"
        params: list[Any] = [*scope.as_tuple(), cursor, now_utc_iso()]
        if kind is not None:
            where += " AND m.kind = ?"
            params.append(kind)
        if source_ref is not None:
            where += (
                f" AND EXISTS (SELECT 1 FROM memory_provenance p WHERE {scope_sql('p')} "
                "AND p.frame_id = m.id AND p.source_ref = ?)"
            )
            params.extend((*scope.as_tuple(), source_ref))
        rows = rt.db.fetchall(
            f"SELECT m.* FROM memory_frames m WHERE {where} ORDER BY m.id LIMIT ?",
            (*params, limit + 1),
        )
        frames = rt._frames([rt.store._row_to_frame(row) for row in rows[:limit]], scope)
        active = [frame for frame in frames if frame.status == "active"]
        kinds, refs = _groups(active)
        return ListMemoriesResponse(
            scope=ScopeModel.model_validate(scope),
            active_frames=active,
            by_kind=kinds,
            by_source_ref=refs,
            review_queue=[frame for frame in frames if frame.status == "candidate"],
            next_cursor=frames[-1].id if len(rows) > limit else None,
        )

    @app.get("/v1/memories/{frame_id}/lineage", response_model=list[MemoryFrameModel])
    def lineage(
        frame_id: int,
        scope: Annotated[MemoryScope, Depends(require_scope)],
        rt: Annotated[_Runtime, Depends(runtime)],
    ) -> list[MemoryFrameModel]:
        return rt._frames(version_lineage(rt.store, frame_id, scope=scope), scope)

    @app.post("/v1/memories/{frame_id}:review", response_model=ReviewResponse)
    def review(
        frame_id: int,
        req: ReviewRequest,
        scope: Annotated[MemoryScope, Depends(require_scope)],
        rt: Annotated[_Runtime, Depends(runtime)],
    ) -> ReviewResponse:
        frame = review_memory(
            rt.store, frame_id, action=req.action, edited_content=req.edited_content, scope=scope
        )
        return ReviewResponse(action=req.action, frame=rt._frames([frame], scope)[0])

    @app.post("/v1/memories/{frame_id}:correct", response_model=CorrectResponse)
    def correct(
        frame_id: int,
        req: CorrectRequest,
        scope: Annotated[MemoryScope, Depends(require_scope)],
        rt: Annotated[_Runtime, Depends(runtime)],
    ) -> CorrectResponse:
        result = correct_memory(rt.store, frame_id, req.new_content, scope=scope)
        old, new = rt._frames([result.superseded_frame, result.replacement_frame], scope)
        return CorrectResponse(
            superseded_frame=old, replacement_frame=new, scope_generation=result.scope_generation
        )

    @app.delete("/v1/memories/{frame_id}", response_model=ForgetResponse)
    def forget(
        frame_id: int,
        scope: Annotated[MemoryScope, Depends(require_scope)],
        rt: Annotated[_Runtime, Depends(runtime)],
    ) -> ForgetResponse:
        return ForgetResponse.model_validate(forget_memory(rt.store, frame_id, scope=scope))

    @app.post("/v1/sources:retract", response_model=RetractSourceResponse)
    def retract(
        req: RetractSourceRequest,
        scope: Annotated[MemoryScope, Depends(require_scope)],
        rt: Annotated[_Runtime, Depends(runtime)],
    ) -> RetractSourceResponse:
        return RetractSourceResponse.model_validate(
            retract_source(rt.store, req.source_ref, scope=scope)
        )

    @app.post(
        "/v1/memories:sweep-expired",
        response_model=SweepExpiredResponse | JobResponse,
        responses={202: {"model": JobResponse}},
    )
    def sweep_expired(
        scope: Annotated[MemoryScope, Depends(require_scope)],
        rt: Annotated[_Runtime, Depends(runtime)],
        response: Response,
        req: SweepExpiredRequest | None = None,
    ) -> SweepExpiredResponse | JobResponse:
        req = req or SweepExpiredRequest()
        if req.async_mode:
            eff_now = req.now_iso or now_utc_iso()
            dedup_key = req.dedup_key or f"sweep_expired:{eff_now}"
            payload: dict[str, Any] = {"now_iso": req.now_iso} if req.now_iso is not None else {}
            job = rt.queue.enqueue(
                scope,
                "sweep_expired",
                dedup_key,
                payload,
            )
            response.status_code = 202
            return JobResponse.model_validate(job)
        return SweepExpiredResponse.model_validate(
            sweep_expired_episodes(rt.store, scope=scope, now_iso=req.now_iso)
        )

    @app.post("/v1/memories:export")
    def export(
        scope: Annotated[MemoryScope, Depends(require_scope)],
        rt: Annotated[_Runtime, Depends(runtime)],
    ) -> dict[str, Any]:
        frames = rt._frames(
            [
                frame
                for frame in rt.store.all_frames(scope=scope)
                if is_recall_eligible(frame) or frame.status == "candidate"
            ],
            scope,
        )
        active = [frame for frame in frames if frame.status == "active"]
        kinds, refs = _groups(active)
        serialized = {
            frame.id: {
                **frame.model_dump(),
                "lineage": [
                    {
                        "source_frame_id": entry.source_frame_id,
                        "source_frame_type": entry.source_frame_type,
                        "source_content_hash": entry.source_content_hash,
                        "operation": entry.operation,
                    }
                    for entry in rt.store.lineage(frame.id)
                ],
            }
            for frame in frames
        }
        return {
            "scope": scope.as_dict(),
            "scope_generation": get_scope_generation(rt.db.backend, scope),
            "active_memories": [serialized[frame.id] for frame in active],
            "by_kind": kinds,
            "by_source_ref": refs,
            "review_queue": [
                serialized[frame.id] for frame in frames if frame.status == "candidate"
            ],
        }

    @app.get("/v1/jobs/{job_id}", response_model=JobResponse)
    def job_status(
        job_id: int,
        scope: Annotated[MemoryScope, Depends(require_scope)],
        rt: Annotated[_Runtime, Depends(runtime)],
    ) -> JobResponse:
        job = rt.queue.get_job(job_id)
        if job.scope != scope:
            raise KeyError(job_id)
        return JobResponse.model_validate(job)

    @app.post("/mcp")
    def mcp_endpoint(
        payload: dict[str, Any],
        response: Response,
        scope: Annotated[MemoryScope, Depends(require_scope)],
        rt: Annotated[_Runtime, Depends(runtime)],
    ) -> dict[str, Any]:
        from gluomem.client.sdk import EmbeddedMemoryClient

        embedded = EmbeddedMemoryClient(
            rt.db,
            rt.store,
            cfg=cfg,
            default_scope=scope,
            search=rt.search,
            kg=rt.kg,
            queue=rt.queue,
            extractors=extractors,
        )
        server = McpServer(embedded, default_scope=scope)
        reply = server.handle_message(payload, scope_override=scope)
        if reply is None:
            response.status_code = 202
            return {"status": "accepted"}
        return reply

    return app
