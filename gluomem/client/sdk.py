"""Typed HTTP clients sharing one request-spec operations layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from pydantic import TypeAdapter

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
    RecallResponse,
    RetractSourceResponse,
    ReviewResponse,
    ScopeModel,
    SourceType,
    SweepExpiredResponse,
)

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
        return self._event(**kwargs)

    def enqueue_event(self, **kwargs: Any) -> _Spec[JobResponse]:
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
