"""Kvark host integration bundle (`KvarkMemoryAdapter`)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from gluomem.client.models import IngestEventResponse, JobResponse, MemoryKind
from gluomem.client.sdk import AsyncGluoMemClient
from gluomem.harnesses.hooks import format_memory_context, strip_thinking_blocks
from gluomem.storage.scope import MemoryScope

_LOG = logging.getLogger(__name__)


def kvark_scope(
    organization_id: str,
    user_id: str,
    workspace_id: str = "personal",
) -> MemoryScope:
    """Map Kvark ``(organization_id, user_id, workspace_id)`` identity to ``MemoryScope``."""
    return MemoryScope(
        tenant_id=organization_id,
        principal_id=user_id,
        namespace=workspace_id,
    )


class KvarkMemoryAdapter:
    """Async adapter integrating Kvark chat pipelines with the GluoMem HTTP service.

    - Maps ``(organization_id, user_id, workspace_id)`` ->
      ``MemoryScope(tenant_id, principal_id, namespace)``.
    - Provides bounded, fail-open ``recall()`` so memory retrieval never blocks
      or breaks a chat turn.
    - Provides non-blocking ``capture_turn()`` backed by ``AsyncGluoMemClient.enqueue_event()``.
    """

    def __init__(
        self,
        client: AsyncGluoMemClient,
        *,
        recall_timeout_seconds: float = 1.5,
        default_k: int = 5,
        default_profile: str = "balanced",
    ) -> None:
        self._client = client
        self._recall_timeout_seconds = float(recall_timeout_seconds)
        self._default_k = int(default_k)
        self._default_profile = default_profile

    @property
    def client(self) -> AsyncGluoMemClient:
        return self._client

    @staticmethod
    def scope_for(
        organization_id: str,
        user_id: str,
        workspace_id: str = "personal",
    ) -> MemoryScope:
        return kvark_scope(organization_id, user_id, workspace_id)

    async def recall(
        self,
        query: str,
        *,
        organization_id: str,
        user_id: str,
        workspace_id: str = "personal",
        k: int | None = None,
        profile: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Bounded fail-open memory recall returning a ``<memory_context>`` block or ``""``."""
        cleaned = query.strip()
        if not cleaned:
            return ""
        scope = kvark_scope(organization_id, user_id, workspace_id)
        eff_timeout = self._recall_timeout_seconds if timeout is None else float(timeout)
        try:
            coro = self._client.recall(
                cleaned,
                k=k if k is not None else self._default_k,
                profile=profile or self._default_profile,
                scope=scope,
            )
            response = (
                await asyncio.wait_for(coro, timeout=eff_timeout) if eff_timeout > 0 else await coro
            )
        except Exception as exc:
            _LOG.warning(
                "Kvark memory recall failed open",
                extra={
                    "event": "kvark_recall_fail_open",
                    "scope": scope.as_dict(),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return ""
        if response.abstained or not response.hits:
            return ""
        dumped = response.model_dump(mode="json")
        return format_memory_context(list(dumped.get("hits", [])))

    async def capture_turn(
        self,
        *,
        organization_id: str,
        user_id: str,
        workspace_id: str = "personal",
        session_id: str,
        turn_id: str | None = None,
        user_message: str = "",
        assistant_message: str = "",
        messages: list[dict[str, Any]] | None = None,
        revision: int = 1,
        fail_open: bool = True,
    ) -> JobResponse | None:
        """Enqueue a conversational turn for asynchronous extraction and consolidation."""
        scope = kvark_scope(organization_id, user_id, workspace_id)
        normalized_messages: list[dict[str, Any]] = []
        if messages:
            for idx, msg in enumerate(messages):
                role = str(msg.get("role", "user")).strip() or "user"
                content = strip_thinking_blocks(str(msg.get("content", "")))
                if not content:
                    continue
                mid = str(msg.get("message_id") or f"m{idx + 1}")
                entry: dict[str, Any] = {
                    "message_id": mid,
                    "role": role,
                    "content": content,
                }
                if msg.get("timestamp"):
                    entry["timestamp"] = str(msg["timestamp"])
                normalized_messages.append(entry)
        else:
            user_clean = strip_thinking_blocks(user_message)
            if user_clean:
                normalized_messages.append(
                    {"message_id": "user", "role": "user", "content": user_clean}
                )
            assistant_clean = strip_thinking_blocks(assistant_message)
            if assistant_clean:
                normalized_messages.append(
                    {"message_id": "assistant", "role": "assistant", "content": assistant_clean}
                )

        if not normalized_messages:
            return None

        source_ref = f"kvark:session:{session_id}"
        if turn_id is not None and turn_id.strip():
            event_id = f"kvark:{session_id}:{turn_id.strip()}"
        else:
            digest = hashlib.sha256(
                "\n".join(f"{m['role']}:{m['content']}" for m in normalized_messages).encode(
                    "utf-8"
                )
            ).hexdigest()[:16]
            event_id = f"kvark:{session_id}:{digest}"

        try:
            return await self._client.enqueue_event(
                event_id=event_id,
                revision=revision,
                source_type="chat_turn",
                source_ref=source_ref,
                messages=normalized_messages,
                scope=scope,
            )
        except Exception as exc:
            if not fail_open:
                raise
            _LOG.warning(
                "Kvark turn capture failed open",
                extra={
                    "event": "kvark_capture_fail_open",
                    "scope": scope.as_dict(),
                    "event_id": event_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return None

    async def remember_explicit(
        self,
        content: str,
        *,
        organization_id: str,
        user_id: str,
        workspace_id: str = "personal",
        kind: MemoryKind = "fact",
        source_ref: str = "kvark:explicit",
        event_id: str | None = None,
        pinned: bool = False,
    ) -> IngestEventResponse:
        """Synchronously persist an explicit user-verified memory."""
        scope = kvark_scope(organization_id, user_id, workspace_id)
        resolved_event_id = (
            event_id or "kvark:explicit:" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        )
        return await self._client.ingest_event(
            event_id=resolved_event_id,
            source_type="explicit_memory",
            source_ref=source_ref,
            content=content,
            kind=kind,
            pinned=pinned,
            scope=scope,
        )


__all__ = ["KvarkMemoryAdapter", "kvark_scope"]
