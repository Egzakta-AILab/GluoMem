"""Typed HTTP clients sharing one request-spec operations layer."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from pydantic import TypeAdapter, ValidationError

from gluomem.client.models import (
    HEADER_NAMESPACE,
    HEADER_PRINCIPAL,
    HEADER_TENANT,
    CorrectResponse,
    ForgetResponse,
    IngestEventResponse,
    JobResponse,
    ListMemoriesResponse,
    MemoryFrameModel,
    MemoryKind,
    ProvenanceModel,
    RecallHitModel,
    RecallResponse,
    RetractSourceResponse,
    ReviewResponse,
    ScopeModel,
    SourceType,
    SweepExpiredResponse,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from gluomem.config.app_config import AppConfig
    from gluomem.embedding.base import Embedder
    from gluomem.embedding.reranker import Reranker
    from gluomem.frames.store import Frame, FrameStore
    from gluomem.kg.graph import KnowledgeGraph
    from gluomem.llm.outputs import Extractors
    from gluomem.search.hybrid import HybridSearch
    from gluomem.service.jobs import JobQueue
    from gluomem.storage.db import MindDB
    from gluomem.storage.scope import MemoryScope

T = TypeVar("T")


class GluoMemApiError(RuntimeError):
    """The HTTP service returned a non-success response."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"GluoMem API error ({status_code}): {detail}")
        self.status_code, self.detail = status_code, detail


def _raise_for_error(response: Any) -> None:
    if 200 <= response.status_code < 300:
        return
    try:
        body = response.json()
        detail = str(body.get("detail", body)) if isinstance(body, dict) else str(body)
    except ValueError:
        detail = response.text
    raise GluoMemApiError(response.status_code, detail)


@dataclass(frozen=True)
class _Spec(Generic[T]):
    method: str
    path: str
    response: TypeAdapter[T]
    scope: ScopeModel | None = None
    json: dict[str, Any] | None = None
    params: dict[str, Any] | None = None


class _Operations:
    def __init__(self, default_scope: Any = None) -> None:
        self.scope = (
            ScopeModel.model_validate(default_scope) if default_scope is not None else ScopeModel()
        )

    def _spec(
        self,
        method: str,
        path: str,
        model: Any,
        *,
        scope: Any = None,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        public: bool = False,
    ) -> _Spec[Any]:
        effective = ScopeModel.model_validate(scope) if scope is not None else self.scope
        return _Spec(method, path, TypeAdapter(model), None if public else effective, json, params)

    def healthz(self) -> _Spec[dict[str, Any]]:
        return self._spec("GET", "/healthz", dict[str, Any], public=True)

    def readyz(self) -> _Spec[dict[str, Any]]:
        return self._spec("GET", "/readyz", dict[str, Any], public=True)

    def status(self, *, scope: Any = None) -> _Spec[dict[str, Any]]:
        return self._spec("GET", "/v1/status", dict[str, Any], scope=scope)

    def _event(
        self,
        *,
        event_id: str,
        source_ref: str,
        content: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        revision: int = 1,
        source_type: SourceType = "chat_turn",
        kind: MemoryKind | None = None,
        confidence: float = 1.0,
        require_review: bool = False,
        pinned: bool = False,
        event_time: str | None = None,
        created_at: str | None = None,
        force_reassert: bool = False,
        scope: Any = None,
        async_mode: bool = False,
    ) -> _Spec[Any]:
        body = {
            "event_id": event_id,
            "source_ref": source_ref,
            "content": content,
            "messages": messages or [],
            "revision": revision,
            "source_type": source_type,
            "kind": kind,
            "confidence": confidence,
            "require_review": require_review,
            "pinned": pinned,
            "event_time": event_time,
            "created_at": created_at,
            "force_reassert": force_reassert,
            "async_mode": async_mode,
        }
        return self._spec(
            "POST",
            "/v1/events",
            JobResponse if async_mode else IngestEventResponse,
            scope=scope,
            json=body,
        )

    def ingest_event(self, **kwargs: Any) -> _Spec[IngestEventResponse]:
        kwargs.pop("async_mode", None)
        return self._event(async_mode=False, **kwargs)

    def enqueue_event(self, **kwargs: Any) -> _Spec[JobResponse]:
        kwargs.pop("async_mode", None)
        return self._event(async_mode=True, **kwargs)

    def recall(
        self,
        query: str,
        *,
        k: int = 10,
        profile: str = "balanced",
        since: str | None = None,
        until: str | None = None,
        now_iso: str | None = None,
        scope: Any = None,
    ) -> _Spec[RecallResponse]:
        return self._spec(
            "POST",
            "/v1/memories:recall",
            RecallResponse,
            scope=scope,
            json={
                "query": query,
                "k": k,
                "profile": profile,
                "since": since,
                "until": until,
                "now_iso": now_iso,
            },
        )

    def list_memories(
        self,
        *,
        include_review_queue: bool = True,
        limit: int = 50,
        cursor: int = 0,
        kind: MemoryKind | None = None,
        source_ref: str | None = None,
        scope: Any = None,
    ) -> _Spec[ListMemoriesResponse]:
        params: dict[str, Any] = {
            "include_review_queue": include_review_queue,
            "limit": limit,
            "cursor": cursor,
        }
        if kind is not None:
            params["kind"] = kind
        if source_ref is not None:
            params["source_ref"] = source_ref
        return self._spec("GET", "/v1/memories", ListMemoriesResponse, scope=scope, params=params)

    def get_lineage(self, frame_id: int, *, scope: Any = None) -> _Spec[list[MemoryFrameModel]]:
        return self._spec(
            "GET", f"/v1/memories/{frame_id}/lineage", list[MemoryFrameModel], scope=scope
        )

    def review_memory(
        self,
        frame_id: int,
        action: Literal["accept", "edit", "dismiss"],
        *,
        edited_content: str | None = None,
        scope: Any = None,
    ) -> _Spec[ReviewResponse]:
        return self._spec(
            "POST",
            f"/v1/memories/{frame_id}:review",
            ReviewResponse,
            scope=scope,
            json={"action": action, "edited_content": edited_content},
        )

    def correct_memory(
        self, frame_id: int, new_content: str, *, scope: Any = None
    ) -> _Spec[CorrectResponse]:
        return self._spec(
            "POST",
            f"/v1/memories/{frame_id}:correct",
            CorrectResponse,
            scope=scope,
            json={"new_content": new_content},
        )

    def forget_memory(self, frame_id: int, *, scope: Any = None) -> _Spec[ForgetResponse]:
        return self._spec("DELETE", f"/v1/memories/{frame_id}", ForgetResponse, scope=scope)

    def retract_source(self, source_ref: str, *, scope: Any = None) -> _Spec[RetractSourceResponse]:
        return self._spec(
            "POST",
            "/v1/sources:retract",
            RetractSourceResponse,
            scope=scope,
            json={"source_ref": source_ref},
        )

    def _sweep_expired(
        self,
        *,
        now_iso: str | None = None,
        dedup_key: str | None = None,
        scope: Any = None,
        async_mode: bool = False,
    ) -> _Spec[Any]:
        body: dict[str, Any] = {"now_iso": now_iso, "async_mode": async_mode}
        if dedup_key is not None:
            body["dedup_key"] = dedup_key
        return self._spec(
            "POST",
            "/v1/memories:sweep-expired",
            JobResponse if async_mode else SweepExpiredResponse,
            scope=scope,
            json=body,
        )

    def sweep_expired(self, **kwargs: Any) -> _Spec[SweepExpiredResponse]:
        return self._sweep_expired(**kwargs)

    def enqueue_sweep_expired(self, **kwargs: Any) -> _Spec[JobResponse]:
        return self._sweep_expired(async_mode=True, **kwargs)

    def export_memories(self, *, scope: Any = None) -> _Spec[dict[str, Any]]:
        return self._spec("POST", "/v1/memories:export", dict[str, Any], scope=scope)

    def get_job(self, job_id: int, *, scope: Any = None) -> _Spec[JobResponse]:
        return self._spec("GET", f"/v1/jobs/{job_id}", JobResponse, scope=scope)


class _Transport:
    def _setup(self, service_key: str, default_scope: Any) -> None:
        self._key = service_key
        self._ops = _Operations(default_scope)

    def _request(self, spec: _Spec[Any]) -> dict[str, Any]:
        headers = (
            {}
            if spec.scope is None
            else {
                "Authorization": f"Bearer {self._key}",
                HEADER_TENANT: spec.scope.tenant_id,
                HEADER_PRINCIPAL: spec.scope.principal_id,
                HEADER_NAMESPACE: spec.scope.namespace,
            }
        )
        return {
            "method": spec.method,
            "url": spec.path,
            "headers": headers,
            "json": spec.json,
            "params": spec.params,
        }


class GluoMemClient(_Transport):
    """Synchronous typed client for the GluoMem v1 API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        service_key: str,
        default_scope: Any = None,
        timeout: float = 10.0,
        transport: Any = None,
    ) -> None:
        import httpx

        self._setup(service_key, default_scope)
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GluoMemClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _send(self, spec: _Spec[T]) -> T:
        response = self._client.request(**self._request(spec))
        _raise_for_error(response)
        return spec.response.validate_python(response.json())

    def healthz(self, **kwargs: Any) -> dict[str, Any]:
        return self._send(self._ops.healthz(**kwargs))

    def readyz(self, **kwargs: Any) -> dict[str, Any]:
        return self._send(self._ops.readyz(**kwargs))

    def status(self, **kwargs: Any) -> dict[str, Any]:
        return self._send(self._ops.status(**kwargs))

    def ingest_event(self, **kwargs: Any) -> IngestEventResponse:
        return self._send(self._ops.ingest_event(**kwargs))

    def enqueue_event(self, **kwargs: Any) -> JobResponse:
        return self._send(self._ops.enqueue_event(**kwargs))

    def recall(self, query: str, **kwargs: Any) -> RecallResponse:
        return self._send(self._ops.recall(query, **kwargs))

    def list_memories(self, **kwargs: Any) -> ListMemoriesResponse:
        return self._send(self._ops.list_memories(**kwargs))

    def get_lineage(self, frame_id: int, **kwargs: Any) -> list[MemoryFrameModel]:
        return self._send(self._ops.get_lineage(frame_id, **kwargs))

    def review_memory(
        self, frame_id: int, action: Literal["accept", "edit", "dismiss"], **kwargs: Any
    ) -> ReviewResponse:
        return self._send(self._ops.review_memory(frame_id, action, **kwargs))

    def correct_memory(self, frame_id: int, new_content: str, **kwargs: Any) -> CorrectResponse:
        return self._send(self._ops.correct_memory(frame_id, new_content, **kwargs))

    def forget_memory(self, frame_id: int, **kwargs: Any) -> ForgetResponse:
        return self._send(self._ops.forget_memory(frame_id, **kwargs))

    def retract_source(self, source_ref: str, **kwargs: Any) -> RetractSourceResponse:
        return self._send(self._ops.retract_source(source_ref, **kwargs))

    def sweep_expired(self, **kwargs: Any) -> SweepExpiredResponse:
        return self._send(self._ops.sweep_expired(**kwargs))

    def enqueue_sweep_expired(self, **kwargs: Any) -> JobResponse:
        return self._send(self._ops.enqueue_sweep_expired(**kwargs))

    def export_memories(self, **kwargs: Any) -> dict[str, Any]:
        return self._send(self._ops.export_memories(**kwargs))

    def get_job(self, job_id: int, **kwargs: Any) -> JobResponse:
        return self._send(self._ops.get_job(job_id, **kwargs))


class AsyncGluoMemClient(_Transport):
    """Asynchronous typed client for the GluoMem v1 API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        service_key: str,
        default_scope: Any = None,
        timeout: float = 10.0,
        transport: Any = None,
    ) -> None:
        import httpx

        self._setup(service_key, default_scope)
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncGluoMemClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def _asend(self, spec: _Spec[T]) -> T:
        response = await self._client.request(**self._request(spec))
        _raise_for_error(response)
        return spec.response.validate_python(response.json())

    async def healthz(self, **kwargs: Any) -> dict[str, Any]:
        return await self._asend(self._ops.healthz(**kwargs))

    async def readyz(self, **kwargs: Any) -> dict[str, Any]:
        return await self._asend(self._ops.readyz(**kwargs))

    async def status(self, **kwargs: Any) -> dict[str, Any]:
        return await self._asend(self._ops.status(**kwargs))

    async def ingest_event(self, **kwargs: Any) -> IngestEventResponse:
        return await self._asend(self._ops.ingest_event(**kwargs))

    async def enqueue_event(self, **kwargs: Any) -> JobResponse:
        return await self._asend(self._ops.enqueue_event(**kwargs))

    async def recall(self, query: str, **kwargs: Any) -> RecallResponse:
        return await self._asend(self._ops.recall(query, **kwargs))

    async def list_memories(self, **kwargs: Any) -> ListMemoriesResponse:
        return await self._asend(self._ops.list_memories(**kwargs))

    async def get_lineage(self, frame_id: int, **kwargs: Any) -> list[MemoryFrameModel]:
        return await self._asend(self._ops.get_lineage(frame_id, **kwargs))

    async def review_memory(
        self, frame_id: int, action: Literal["accept", "edit", "dismiss"], **kwargs: Any
    ) -> ReviewResponse:
        return await self._asend(self._ops.review_memory(frame_id, action, **kwargs))

    async def correct_memory(
        self, frame_id: int, new_content: str, **kwargs: Any
    ) -> CorrectResponse:
        return await self._asend(self._ops.correct_memory(frame_id, new_content, **kwargs))

    async def forget_memory(self, frame_id: int, **kwargs: Any) -> ForgetResponse:
        return await self._asend(self._ops.forget_memory(frame_id, **kwargs))

    async def retract_source(self, source_ref: str, **kwargs: Any) -> RetractSourceResponse:
        return await self._asend(self._ops.retract_source(source_ref, **kwargs))

    async def sweep_expired(self, **kwargs: Any) -> SweepExpiredResponse:
        return await self._asend(self._ops.sweep_expired(**kwargs))

    async def enqueue_sweep_expired(self, **kwargs: Any) -> JobResponse:
        return await self._asend(self._ops.enqueue_sweep_expired(**kwargs))

    async def export_memories(self, **kwargs: Any) -> dict[str, Any]:
        return await self._asend(self._ops.export_memories(**kwargs))

    async def get_job(self, job_id: int, **kwargs: Any) -> JobResponse:
        return await self._asend(self._ops.get_job(job_id, **kwargs))


def _group_frames(
    frames: Sequence[MemoryFrameModel],
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    kinds: dict[str, list[int]] = {}
    refs: dict[str, list[int]] = {}
    for frame in frames:
        kinds.setdefault(frame.kind, []).append(frame.id)
        for ref in sorted({p.source_ref for p in frame.provenance}) or [f"gop:{frame.gop_id}"]:
            refs.setdefault(ref, []).append(frame.id)
    return kinds, refs


_UNSET: Any = object()


class EmbeddedMemoryClient:
    """In-process memory client with the same surface and contracts as ``GluoMemClient``."""

    def __init__(
        self,
        db: MindDB,
        store: FrameStore | None = None,
        *,
        cfg: AppConfig | None = None,
        default_scope: Any = None,
        search: HybridSearch | None = None,
        kg: KnowledgeGraph | None = None,
        queue: JobQueue | None = None,
        extractors: Extractors | None = None,
        reranker: Reranker | None = None,
        owns_db: bool = False,
    ) -> None:
        from gluomem.config.app_config import AppConfig as _AppConfig
        from gluomem.embedding.providers import make_embedder
        from gluomem.frames.store import FrameStore as _FrameStore
        from gluomem.kg.graph import KnowledgeGraph as _KnowledgeGraph
        from gluomem.search.hybrid import HybridSearch as _HybridSearch
        from gluomem.service.jobs import JobQueue as _JobQueue
        from gluomem.storage.scope import (
            DEFAULT_SCOPE,
            resolve_scope,
        )
        from gluomem.storage.scope import (
            MemoryScope as _MemoryScope,
        )

        self.db = db
        self.cfg = cfg if cfg is not None else _AppConfig()
        base_scope = store.scope if store is not None else DEFAULT_SCOPE
        if default_scope is None:
            self._default_scope = resolve_scope(None, default=base_scope)
        elif isinstance(default_scope, _MemoryScope):
            self._default_scope = resolve_scope(default_scope, default=base_scope)
        else:
            sm = ScopeModel.model_validate(default_scope)
            self._default_scope = resolve_scope(
                _MemoryScope(sm.tenant_id, sm.principal_id, sm.namespace),
                default=base_scope,
            )
        if store is None:
            store = _FrameStore(
                db,
                embedder=make_embedder(self.cfg),
                chunking=self.cfg.chunking,
                scope=self._default_scope,
                lifecycle=self.cfg.lifecycle,
            )
        self.store = store
        self.kg = kg if kg is not None else _KnowledgeGraph(db, scope=self._default_scope)
        self.search = (
            search
            if search is not None
            else _HybridSearch(db, self.store, self.cfg, kg=self.kg, reranker=reranker)
        )
        self.queue = queue if queue is not None else _JobQueue(db.backend)
        self.extractors = extractors
        self._owns_db = owns_db

    @classmethod
    def open(
        cls,
        path: str | Path | None = None,
        *,
        cfg: AppConfig | None = None,
        default_scope: Any = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        extractors: Extractors | None = _UNSET,
    ) -> EmbeddedMemoryClient:
        from gluomem.config.app_config import AppConfig as _AppConfig
        from gluomem.embedding.providers import make_embedder
        from gluomem.frames.store import FrameStore as _FrameStore
        from gluomem.llm import make_extractors
        from gluomem.storage.db import MindDB as _MindDB
        from gluomem.storage.scope import (
            MemoryScope as _MemoryScope,
        )
        from gluomem.storage.scope import (
            resolve_scope,
        )

        eff_cfg = cfg if cfg is not None else _AppConfig()
        eff_embedder = embedder if embedder is not None else make_embedder(eff_cfg)
        eff_extractors: Extractors | None = (
            make_extractors(eff_cfg) if extractors is _UNSET else extractors
        )
        if default_scope is None:
            scope = resolve_scope(None)
        elif isinstance(default_scope, _MemoryScope):
            scope = resolve_scope(default_scope)
        else:
            sm = ScopeModel.model_validate(default_scope)
            scope = resolve_scope(_MemoryScope(sm.tenant_id, sm.principal_id, sm.namespace))
        backend_type: Literal["sqlite", "postgres"] = (
            "postgres" if eff_cfg.storage.backend == "postgres" else "sqlite"
        )
        db = _MindDB(
            path if path is not None else eff_cfg.storage.path,
            dim=eff_embedder.dim,
            backend_type=backend_type,
            dsn=eff_cfg.storage.dsn,
            postgres_schema=eff_cfg.storage.postgres_schema,
        )
        store = _FrameStore(
            db,
            embedder=eff_embedder,
            chunking=eff_cfg.chunking,
            scope=scope,
            lifecycle=eff_cfg.lifecycle,
        )
        return cls(
            db,
            store,
            cfg=eff_cfg,
            default_scope=scope,
            reranker=reranker,
            extractors=eff_extractors,
            owns_db=True,
        )

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def __enter__(self) -> EmbeddedMemoryClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _scope(self, scope: Any = None) -> MemoryScope:
        from gluomem.storage.scope import MemoryScope as _MemoryScope

        if scope is None:
            return self._default_scope
        if isinstance(scope, _MemoryScope):
            return scope
        validated = ScopeModel.model_validate(scope)
        return _MemoryScope(validated.tenant_id, validated.principal_id, validated.namespace)

    def _frames(self, frames: Sequence[Frame], scope: MemoryScope) -> list[MemoryFrameModel]:
        from gluomem.storage.scope import scope_sql

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

    def _call(self, fn: Callable[[], T]) -> T:
        from gluomem.service.jobs import BackpressureError, EventConflictError

        try:
            return fn()
        except KeyError as exc:
            raise GluoMemApiError(404, "Memory resource not found") from exc
        except EventConflictError as exc:
            raise GluoMemApiError(409, str(exc)) from exc
        except BackpressureError as exc:
            raise GluoMemApiError(429, str(exc)) from exc
        except (ValueError, ValidationError) as exc:
            raise GluoMemApiError(422, str(exc)) from exc

    def healthz(self) -> dict[str, Any]:
        return {"status": "ok"}

    def readyz(self) -> dict[str, Any]:
        if self.db.scalar("SELECT 1") != 1:
            raise GluoMemApiError(503, "Database probe failed")
        return {"status": "ok"}

    def status(self, *, scope: Any = None) -> dict[str, Any]:
        from gluomem.embedding.compat import stored_fingerprint

        def _op() -> dict[str, Any]:
            eff_scope = self._scope(scope)
            fp = stored_fingerprint(self.db)
            return {
                "dialect": self.db.dialect,
                "embedder_fingerprint": fp.to_json() if fp else None,
                "queue": self.queue.queue_stats(eff_scope),
            }

        return self._call(_op)

    def ingest_event(self, *, scope: Any = None, **kwargs: Any) -> IngestEventResponse:
        from gluomem.service.schemas import IngestEventRequest
        from gluomem.write.events import ingest_source_event

        def _op() -> IngestEventResponse:
            eff_scope = self._scope(scope)
            req = IngestEventRequest.model_validate(kwargs)
            result = ingest_source_event(
                req.to_source_event(eff_scope),
                store=self.store,
                kg=self.kg,
                cfg=self.cfg,
                extractors=self.extractors,
            )
            return IngestEventResponse(
                source_event_row_id=result.event_row_id,
                event_id=result.event_id,
                revision=result.revision,
                deduped_event=result.deduplicated,
                frames=self._frames(result.frames, eff_scope),
                suppressed_hashes=result.suppressed_hashes,
            )

        return self._call(_op)

    def enqueue_event(self, *, scope: Any = None, **kwargs: Any) -> JobResponse:
        from gluomem.service.schemas import IngestEventRequest

        def _op() -> JobResponse:
            eff_scope = self._scope(scope)
            req = IngestEventRequest.model_validate({**kwargs, "async_mode": True})
            job = self.queue.enqueue(
                eff_scope,
                "ingest_event",
                f"{req.event_id}:{req.revision}",
                req.model_dump(exclude={"async_mode"}),
            )
            return JobResponse.model_validate(job)

        return self._call(_op)

    def recall(
        self,
        query: str,
        *,
        k: int = 10,
        profile: str = "balanced",
        since: str | None = None,
        until: str | None = None,
        now_iso: str | None = None,
        scope: Any = None,
    ) -> RecallResponse:
        from gluomem.service.schemas import RecallRequest

        def _op() -> RecallResponse:
            eff_scope = self._scope(scope)
            req = RecallRequest.model_validate(
                {
                    "query": query,
                    "k": k,
                    "profile": profile,
                    "since": since,
                    "until": until,
                    "now_iso": now_iso,
                }
            )
            if req.profile not in self.cfg.scoring.profiles:
                raise ValueError("Unknown scoring profile")
            started = time.perf_counter()
            result = self.search.search(
                req.query,
                limit=req.k,
                profile=req.profile,
                since=req.since,
                until=req.until,
                scope=eff_scope,
                now_iso=req.now_iso,
            )
            frames = self._frames([hit.frame for hit in result.hits], eff_scope)
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

        return self._call(_op)

    def list_memories(
        self,
        *,
        include_review_queue: bool = True,
        limit: int = 50,
        cursor: int = 0,
        kind: MemoryKind | None = None,
        source_ref: str | None = None,
        scope: Any = None,
    ) -> ListMemoriesResponse:
        from gluomem.storage.scope import eligible_sql, scope_sql
        from gluomem.timeutil import now_utc_iso

        def _op() -> ListMemoriesResponse:
            if not (1 <= limit <= 100):
                raise ValueError("limit must be between 1 and 100")
            if cursor < 0:
                raise ValueError("cursor must be >= 0")
            eff_scope = self._scope(scope)
            where = f"{scope_sql('m')} AND m.id > ? AND ({eligible_sql('m')}"
            where += " OR m.status = 'candidate')" if include_review_queue else ")"
            params: list[Any] = [*eff_scope.as_tuple(), cursor, now_utc_iso()]
            if kind is not None:
                where += " AND m.kind = ?"
                params.append(kind)
            if source_ref is not None:
                where += (
                    f" AND EXISTS (SELECT 1 FROM memory_provenance p WHERE {scope_sql('p')} "
                    "AND p.frame_id = m.id AND p.source_ref = ?)"
                )
                params.extend((*eff_scope.as_tuple(), source_ref))
            rows = self.db.fetchall(
                f"SELECT m.* FROM memory_frames m WHERE {where} ORDER BY m.id LIMIT ?",
                (*params, limit + 1),
            )
            frames = self._frames(
                [self.store._row_to_frame(row) for row in rows[:limit]], eff_scope
            )
            active = [frame for frame in frames if frame.status == "active"]
            kinds, refs = _group_frames(active)
            return ListMemoriesResponse(
                scope=ScopeModel.model_validate(eff_scope),
                active_frames=active,
                by_kind=kinds,
                by_source_ref=refs,
                review_queue=[frame for frame in frames if frame.status == "candidate"],
                next_cursor=frames[-1].id if len(rows) > limit else None,
            )

        return self._call(_op)

    def get_lineage(self, frame_id: int, *, scope: Any = None) -> list[MemoryFrameModel]:
        from gluomem.frames.lifecycle import version_lineage

        def _op() -> list[MemoryFrameModel]:
            eff_scope = self._scope(scope)
            return self._frames(version_lineage(self.store, frame_id, scope=eff_scope), eff_scope)

        return self._call(_op)

    def review_memory(
        self,
        frame_id: int,
        action: Literal["accept", "edit", "dismiss"],
        *,
        edited_content: str | None = None,
        scope: Any = None,
    ) -> ReviewResponse:
        from gluomem.service.schemas import ReviewRequest
        from gluomem.write.events import review_memory

        def _op() -> ReviewResponse:
            eff_scope = self._scope(scope)
            req = ReviewRequest.model_validate({"action": action, "edited_content": edited_content})
            frame = review_memory(
                self.store,
                frame_id,
                action=req.action,
                edited_content=req.edited_content,
                scope=eff_scope,
            )
            return ReviewResponse(action=req.action, frame=self._frames([frame], eff_scope)[0])

        return self._call(_op)

    def correct_memory(
        self, frame_id: int, new_content: str, *, scope: Any = None
    ) -> CorrectResponse:
        from gluomem.frames.lifecycle import correct_memory
        from gluomem.service.schemas import CorrectRequest

        def _op() -> CorrectResponse:
            eff_scope = self._scope(scope)
            req = CorrectRequest.model_validate({"new_content": new_content})
            result = correct_memory(self.store, frame_id, req.new_content, scope=eff_scope)
            old, new = self._frames([result.superseded_frame, result.replacement_frame], eff_scope)
            return CorrectResponse(
                superseded_frame=old,
                replacement_frame=new,
                scope_generation=result.scope_generation,
            )

        return self._call(_op)

    def forget_memory(self, frame_id: int, *, scope: Any = None) -> ForgetResponse:
        from gluomem.frames.lifecycle import forget_memory

        def _op() -> ForgetResponse:
            eff_scope = self._scope(scope)
            return ForgetResponse.model_validate(
                forget_memory(self.store, frame_id, scope=eff_scope)
            )

        return self._call(_op)

    def retract_source(self, source_ref: str, *, scope: Any = None) -> RetractSourceResponse:
        from gluomem.frames.lifecycle import retract_source
        from gluomem.service.schemas import RetractSourceRequest

        def _op() -> RetractSourceResponse:
            eff_scope = self._scope(scope)
            req = RetractSourceRequest.model_validate({"source_ref": source_ref})
            return RetractSourceResponse.model_validate(
                retract_source(self.store, req.source_ref, scope=eff_scope)
            )

        return self._call(_op)

    def sweep_expired(
        self,
        *,
        now_iso: str | None = None,
        dedup_key: str | None = None,
        scope: Any = None,
    ) -> SweepExpiredResponse:
        from gluomem.frames.expiry import sweep_expired_episodes
        from gluomem.service.schemas import SweepExpiredRequest

        def _op() -> SweepExpiredResponse:
            eff_scope = self._scope(scope)
            req = SweepExpiredRequest.model_validate({"now_iso": now_iso, "dedup_key": dedup_key})
            return SweepExpiredResponse.model_validate(
                sweep_expired_episodes(self.store, scope=eff_scope, now_iso=req.now_iso)
            )

        return self._call(_op)

    def enqueue_sweep_expired(
        self,
        *,
        now_iso: str | None = None,
        dedup_key: str | None = None,
        scope: Any = None,
    ) -> JobResponse:
        from gluomem.service.schemas import SweepExpiredRequest
        from gluomem.timeutil import now_utc_iso

        def _op() -> JobResponse:
            eff_scope = self._scope(scope)
            req = SweepExpiredRequest.model_validate(
                {"now_iso": now_iso, "dedup_key": dedup_key, "async_mode": True}
            )
            eff_now = req.now_iso or now_utc_iso()
            key = req.dedup_key or f"sweep_expired:{eff_now}"
            payload: dict[str, Any] = {"now_iso": req.now_iso} if req.now_iso is not None else {}
            job = self.queue.enqueue(eff_scope, "sweep_expired", key, payload)
            return JobResponse.model_validate(job)

        return self._call(_op)

    def export_memories(self, *, scope: Any = None) -> dict[str, Any]:
        from gluomem.frames.eligibility import is_recall_eligible
        from gluomem.frames.lifecycle import get_scope_generation

        def _op() -> dict[str, Any]:
            eff_scope = self._scope(scope)
            frames = self._frames(
                [
                    frame
                    for frame in self.store.all_frames(scope=eff_scope)
                    if is_recall_eligible(frame) or frame.status == "candidate"
                ],
                eff_scope,
            )
            active = [frame for frame in frames if frame.status == "active"]
            kinds, refs = _group_frames(active)
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
                        for entry in self.store.lineage(frame.id)
                    ],
                }
                for frame in frames
            }
            return {
                "scope": eff_scope.as_dict(),
                "scope_generation": get_scope_generation(self.db.backend, eff_scope),
                "active_memories": [serialized[frame.id] for frame in active],
                "by_kind": kinds,
                "by_source_ref": refs,
                "review_queue": [
                    serialized[frame.id] for frame in frames if frame.status == "candidate"
                ],
            }

        return self._call(_op)

    def get_job(self, job_id: int, *, scope: Any = None) -> JobResponse:
        def _op() -> JobResponse:
            eff_scope = self._scope(scope)
            job = self.queue.get_job(job_id)
            if job.scope != eff_scope:
                raise KeyError(job_id)
            return JobResponse.model_validate(job)

        return self._call(_op)
