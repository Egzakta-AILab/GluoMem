"""Typed sync (`GluoMemClient`) and async (`AsyncGluoMemClient`) SDK over HTTP."""

from __future__ import annotations

from typing import Any, Literal

from gluomem.extras import require
from gluomem.service.auth import (
    HEADER_NAMESPACE,
    HEADER_PRINCIPAL,
    HEADER_TENANT,
)
from gluomem.service.schemas import (
    CorrectResponse,
    ForgetResponse,
    IngestEventResponse,
    JobResponse,
    ListMemoriesResponse,
    MemoryFrameModel,
    RecallResponse,
    RetractSourceResponse,
    ReviewResponse,
)
from gluomem.storage.scope import DEFAULT_SCOPE, MemoryKind, MemoryScope
from gluomem.write.events import SourceType


class GluoMemApiError(RuntimeError):
    """Raised when the GluoMem HTTP service returns a non-2xx response."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"GluoMem API error ({status_code}): {detail}")
        self.status_code = status_code
        self.detail = detail


def _build_headers(service_key: str, scope: MemoryScope) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {service_key}",
        HEADER_TENANT: scope.tenant_id,
        HEADER_PRINCIPAL: scope.principal_id,
        HEADER_NAMESPACE: scope.namespace,
    }


def _raise_for_error(resp: Any) -> None:
    if 200 <= int(resp.status_code) < 300:
        return
    try:
        body = resp.json()
        detail = str(body.get("detail", body))
    except Exception:
        detail = resp.text
    raise GluoMemApiError(int(resp.status_code), detail)


class GluoMemClient:
    """Synchronous typed client for the GluoMem v1 HTTP API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        service_key: str,
        default_scope: MemoryScope = DEFAULT_SCOPE,
        timeout: float = 10.0,
        transport: Any = None,
    ) -> None:
        require("service", "GluoMemClient")
        import httpx

        self._service_key = service_key
        self._default_scope = default_scope
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GluoMemClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _eff_scope(self, scope: MemoryScope | None) -> MemoryScope:
        return scope if scope is not None else self._default_scope

    def healthz(self) -> dict[str, Any]:
        resp = self._client.get("/healthz")
        _raise_for_error(resp)
        return dict(resp.json())

    def readyz(self) -> dict[str, Any]:
        resp = self._client.get("/readyz")
        _raise_for_error(resp)
        return dict(resp.json())

    def ingest_event(
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
        force_reassert: bool = False,
        scope: MemoryScope | None = None,
    ) -> IngestEventResponse:
        eff_scope = self._eff_scope(scope)
        body: dict[str, Any] = {
            "event_id": event_id,
            "revision": revision,
            "source_type": source_type,
            "source_ref": source_ref,
            "messages": messages or [],
            "content": content,
            "kind": kind,
            "confidence": confidence,
            "require_review": require_review,
            "pinned": pinned,
            "event_time": event_time,
            "force_reassert": force_reassert,
            "async_mode": False,
        }
        resp = self._client.post(
            "/v1/events",
            json=body,
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return IngestEventResponse.model_validate(resp.json())

    def enqueue_event(
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
        force_reassert: bool = False,
        scope: MemoryScope | None = None,
    ) -> JobResponse:
        eff_scope = self._eff_scope(scope)
        body: dict[str, Any] = {
            "event_id": event_id,
            "revision": revision,
            "source_type": source_type,
            "source_ref": source_ref,
            "messages": messages or [],
            "content": content,
            "kind": kind,
            "confidence": confidence,
            "require_review": require_review,
            "pinned": pinned,
            "event_time": event_time,
            "force_reassert": force_reassert,
            "async_mode": True,
        }
        resp = self._client.post(
            "/v1/events",
            json=body,
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return JobResponse.model_validate(resp.json())

    def recall(
        self,
        query: str,
        *,
        k: int = 10,
        profile: str = "balanced",
        since: str | None = None,
        until: str | None = None,
        now_iso: str | None = None,
        scope: MemoryScope | None = None,
    ) -> RecallResponse:
        eff_scope = self._eff_scope(scope)
        resp = self._client.post(
            "/v1/memories:recall",
            json={
                "query": query,
                "k": k,
                "profile": profile,
                "since": since,
                "until": until,
                "now_iso": now_iso,
            },
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return RecallResponse.model_validate(resp.json())

    def list_memories(
        self,
        *,
        include_review_queue: bool = True,
        scope: MemoryScope | None = None,
    ) -> ListMemoriesResponse:
        eff_scope = self._eff_scope(scope)
        resp = self._client.get(
            "/v1/memories",
            params={"include_review_queue": include_review_queue},
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return ListMemoriesResponse.model_validate(resp.json())

    def get_lineage(
        self,
        frame_id: int,
        *,
        scope: MemoryScope | None = None,
    ) -> list[MemoryFrameModel]:
        eff_scope = self._eff_scope(scope)
        resp = self._client.get(
            f"/v1/memories/{frame_id}/lineage",
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return [MemoryFrameModel.model_validate(item) for item in resp.json()]

    def review_memory(
        self,
        frame_id: int,
        action: Literal["accept", "edit", "dismiss"],
        *,
        edited_content: str | None = None,
        kind: MemoryKind | None = None,
        scope: MemoryScope | None = None,
    ) -> ReviewResponse:
        eff_scope = self._eff_scope(scope)
        resp = self._client.post(
            f"/v1/memories/{frame_id}:review",
            json={
                "action": action,
                "edited_content": edited_content,
                "kind": kind,
            },
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return ReviewResponse.model_validate(resp.json())

    def correct_memory(
        self,
        frame_id: int,
        new_content: str,
        *,
        kind: MemoryKind | None = None,
        scope: MemoryScope | None = None,
    ) -> CorrectResponse:
        eff_scope = self._eff_scope(scope)
        resp = self._client.post(
            f"/v1/memories/{frame_id}:correct",
            json={"new_content": new_content, "kind": kind},
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return CorrectResponse.model_validate(resp.json())

    def forget_memory(
        self,
        frame_id: int,
        *,
        scope: MemoryScope | None = None,
    ) -> ForgetResponse:
        eff_scope = self._eff_scope(scope)
        resp = self._client.delete(
            f"/v1/memories/{frame_id}",
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return ForgetResponse.model_validate(resp.json())

    def retract_source(
        self,
        source_ref: str,
        *,
        scope: MemoryScope | None = None,
    ) -> RetractSourceResponse:
        eff_scope = self._eff_scope(scope)
        resp = self._client.post(
            "/v1/sources:retract",
            json={"source_ref": source_ref},
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return RetractSourceResponse.model_validate(resp.json())

    def export_memories(
        self,
        *,
        scope: MemoryScope | None = None,
    ) -> dict[str, Any]:
        eff_scope = self._eff_scope(scope)
        resp = self._client.post(
            "/v1/memories:export",
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return dict(resp.json())

    def get_job(
        self,
        job_id: int,
        *,
        scope: MemoryScope | None = None,
    ) -> JobResponse:
        eff_scope = self._eff_scope(scope)
        resp = self._client.get(
            f"/v1/jobs/{job_id}",
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return JobResponse.model_validate(resp.json())


class AsyncGluoMemClient:
    """Asynchronous typed client for the GluoMem v1 HTTP API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        service_key: str,
        default_scope: MemoryScope = DEFAULT_SCOPE,
        timeout: float = 10.0,
        transport: Any = None,
    ) -> None:
        require("service", "AsyncGluoMemClient")
        import httpx

        self._service_key = service_key
        self._default_scope = default_scope
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncGluoMemClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    def _eff_scope(self, scope: MemoryScope | None) -> MemoryScope:
        return scope if scope is not None else self._default_scope

    async def healthz(self) -> dict[str, Any]:
        resp = await self._client.get("/healthz")
        _raise_for_error(resp)
        return dict(resp.json())

    async def readyz(self) -> dict[str, Any]:
        resp = await self._client.get("/readyz")
        _raise_for_error(resp)
        return dict(resp.json())

    async def ingest_event(
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
        force_reassert: bool = False,
        scope: MemoryScope | None = None,
    ) -> IngestEventResponse:
        eff_scope = self._eff_scope(scope)
        body: dict[str, Any] = {
            "event_id": event_id,
            "revision": revision,
            "source_type": source_type,
            "source_ref": source_ref,
            "messages": messages or [],
            "content": content,
            "kind": kind,
            "confidence": confidence,
            "require_review": require_review,
            "pinned": pinned,
            "event_time": event_time,
            "force_reassert": force_reassert,
            "async_mode": False,
        }
        resp = await self._client.post(
            "/v1/events",
            json=body,
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return IngestEventResponse.model_validate(resp.json())

    async def enqueue_event(
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
        force_reassert: bool = False,
        scope: MemoryScope | None = None,
    ) -> JobResponse:
        eff_scope = self._eff_scope(scope)
        body: dict[str, Any] = {
            "event_id": event_id,
            "revision": revision,
            "source_type": source_type,
            "source_ref": source_ref,
            "messages": messages or [],
            "content": content,
            "kind": kind,
            "confidence": confidence,
            "require_review": require_review,
            "pinned": pinned,
            "event_time": event_time,
            "force_reassert": force_reassert,
            "async_mode": True,
        }
        resp = await self._client.post(
            "/v1/events",
            json=body,
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return JobResponse.model_validate(resp.json())

    async def recall(
        self,
        query: str,
        *,
        k: int = 10,
        profile: str = "balanced",
        since: str | None = None,
        until: str | None = None,
        now_iso: str | None = None,
        scope: MemoryScope | None = None,
    ) -> RecallResponse:
        eff_scope = self._eff_scope(scope)
        resp = await self._client.post(
            "/v1/memories:recall",
            json={
                "query": query,
                "k": k,
                "profile": profile,
                "since": since,
                "until": until,
                "now_iso": now_iso,
            },
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return RecallResponse.model_validate(resp.json())

    async def list_memories(
        self,
        *,
        include_review_queue: bool = True,
        scope: MemoryScope | None = None,
    ) -> ListMemoriesResponse:
        eff_scope = self._eff_scope(scope)
        resp = await self._client.get(
            "/v1/memories",
            params={"include_review_queue": include_review_queue},
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return ListMemoriesResponse.model_validate(resp.json())

    async def review_memory(
        self,
        frame_id: int,
        action: Literal["accept", "edit", "dismiss"],
        *,
        edited_content: str | None = None,
        kind: MemoryKind | None = None,
        scope: MemoryScope | None = None,
    ) -> ReviewResponse:
        eff_scope = self._eff_scope(scope)
        resp = await self._client.post(
            f"/v1/memories/{frame_id}:review",
            json={
                "action": action,
                "edited_content": edited_content,
                "kind": kind,
            },
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return ReviewResponse.model_validate(resp.json())

    async def correct_memory(
        self,
        frame_id: int,
        new_content: str,
        *,
        kind: MemoryKind | None = None,
        scope: MemoryScope | None = None,
    ) -> CorrectResponse:
        eff_scope = self._eff_scope(scope)
        resp = await self._client.post(
            f"/v1/memories/{frame_id}:correct",
            json={"new_content": new_content, "kind": kind},
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return CorrectResponse.model_validate(resp.json())

    async def forget_memory(
        self,
        frame_id: int,
        *,
        scope: MemoryScope | None = None,
    ) -> ForgetResponse:
        eff_scope = self._eff_scope(scope)
        resp = await self._client.delete(
            f"/v1/memories/{frame_id}",
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return ForgetResponse.model_validate(resp.json())

    async def retract_source(
        self,
        source_ref: str,
        *,
        scope: MemoryScope | None = None,
    ) -> RetractSourceResponse:
        eff_scope = self._eff_scope(scope)
        resp = await self._client.post(
            "/v1/sources:retract",
            json={"source_ref": source_ref},
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return RetractSourceResponse.model_validate(resp.json())

    async def export_memories(
        self,
        *,
        scope: MemoryScope | None = None,
    ) -> dict[str, Any]:
        eff_scope = self._eff_scope(scope)
        resp = await self._client.post(
            "/v1/memories:export",
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return dict(resp.json())

    async def get_job(
        self,
        job_id: int,
        *,
        scope: MemoryScope | None = None,
    ) -> JobResponse:
        eff_scope = self._eff_scope(scope)
        resp = await self._client.get(
            f"/v1/jobs/{job_id}",
            headers=_build_headers(self._service_key, eff_scope),
        )
        _raise_for_error(resp)
        return JobResponse.model_validate(resp.json())
