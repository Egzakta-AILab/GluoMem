"""Lifecycle hook adapter for Cursor, Claude Code, Hermes, OpenClaw, and custom harnesses."""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

from gluomem.storage.scope import DEFAULT_SCOPE, MemoryScope

if TYPE_CHECKING:
    from gluomem.client.sdk import EmbeddedMemoryClient, GluoMemClient

_THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)


def strip_thinking_blocks(text: str) -> str:
    """Strip ``<thinking>...</thinking>`` blocks before memory ingestion."""
    return _THINKING_RE.sub("", text).strip()


def resolve_hook_scope(
    payload: dict[str, Any],
    *,
    default_scope: MemoryScope = DEFAULT_SCOPE,
) -> MemoryScope:
    """Resolve ``MemoryScope`` from a hook payload or fall back to ``default_scope``."""
    raw_scope = payload.get("scope")
    if isinstance(raw_scope, dict):
        return MemoryScope(
            tenant_id=str(raw_scope.get("tenant_id") or default_scope.tenant_id),
            principal_id=str(raw_scope.get("principal_id") or default_scope.principal_id),
            namespace=str(raw_scope.get("namespace") or default_scope.namespace),
        )
    return MemoryScope(
        tenant_id=str(payload.get("tenant_id") or default_scope.tenant_id),
        principal_id=str(
            payload.get("principal_id") or payload.get("user_id") or default_scope.principal_id
        ),
        namespace=str(payload.get("namespace") or default_scope.namespace),
    )


def format_memory_context(hits: list[dict[str, Any]]) -> str:
    """Format recalled memory frames into a compact ``<memory_context>`` block."""
    if not hits:
        return ""
    lines = ["<memory_context>"]
    for hit in hits:
        frame = hit.get("frame", hit)
        fid = frame.get("id", "?")
        kind = frame.get("kind", "fact")
        content = str(frame.get("content", "")).strip()
        if content:
            lines.append(f"- [{fid} | {kind}] {content}")
    if len(lines) == 1:
        return ""
    lines.append("</memory_context>")
    return "\n".join(lines)


def handle_recall_hook(
    client: EmbeddedMemoryClient | GluoMemClient,
    payload: dict[str, Any],
    *,
    default_scope: MemoryScope = DEFAULT_SCOPE,
    k: int = 5,
    profile: str = "balanced",
) -> dict[str, Any]:
    """Process a ``beforeSubmitPrompt`` / ``UserPromptSubmit`` recall hook payload."""
    query = str(
        payload.get("prompt")
        or payload.get("query")
        or payload.get("user_message")
        or payload.get("message")
        or ""
    ).strip()
    if not query:
        return {
            "continue": True,
            "abstained": True,
            "additionalContext": "",
            "hits": [],
        }

    scope = resolve_hook_scope(payload, default_scope=default_scope)
    eff_k = int(payload.get("k", k))
    eff_profile = str(payload.get("profile", profile))
    resp = client.recall(query, k=eff_k, profile=eff_profile, scope=scope)
    dumped = resp.model_dump(mode="json")
    hits = [] if resp.abstained else list(dumped.get("hits", []))
    context_block = format_memory_context(hits)
    return {
        "continue": True,
        "abstained": resp.abstained,
        "additionalContext": context_block,
        "hits": hits,
    }


def handle_capture_hook(
    client: EmbeddedMemoryClient | GluoMemClient,
    payload: dict[str, Any],
    *,
    default_scope: MemoryScope = DEFAULT_SCOPE,
    async_mode: bool = False,
) -> dict[str, Any]:
    """Process a ``stop`` / ``TurnEnd`` capture hook payload."""
    scope = resolve_hook_scope(payload, default_scope=default_scope)
    session_id = str(
        payload.get("session_id")
        or payload.get("conversation_id")
        or payload.get("source_ref")
        or "hook:session"
    )
    source_ref = session_id if ":" in session_id else f"session:{session_id}"

    messages: list[dict[str, Any]] = []
    raw_messages = payload.get("messages")
    if isinstance(raw_messages, list):
        for idx, msg in enumerate(raw_messages):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "user"))
            cleaned = strip_thinking_blocks(str(msg.get("content", "")))
            if not cleaned:
                continue
            messages.append(
                {
                    "message_id": str(msg.get("message_id") or f"m{idx + 1}"),
                    "role": role,
                    "content": cleaned,
                    "timestamp": msg.get("timestamp"),
                }
            )
    else:
        user_msg = strip_thinking_blocks(
            str(payload.get("prompt") or payload.get("user_message") or "")
        )
        if user_msg:
            messages.append({"message_id": "m1", "role": "user", "content": user_msg})
        assistant_msg = strip_thinking_blocks(
            str(payload.get("response") or payload.get("assistant_message") or "")
        )
        if assistant_msg:
            messages.append({"message_id": "m2", "role": "assistant", "content": assistant_msg})

    raw_content = strip_thinking_blocks(str(payload.get("content") or ""))
    if not messages and not raw_content:
        return {"continue": True, "ingested": False, "reason": "empty_turn"}

    canonical_seed = json.dumps(
        {"source_ref": source_ref, "messages": messages, "content": raw_content},
        sort_keys=True,
        ensure_ascii=False,
    )
    event_id = str(
        payload.get("event_id")
        or payload.get("turn_id")
        or ("hook-" + hashlib.sha256(canonical_seed.encode("utf-8")).hexdigest()[:20])
    )
    revision = int(payload.get("revision", 1))
    source_type = str(payload.get("source_type", "chat_turn"))

    kwargs: dict[str, Any] = {
        "event_id": event_id,
        "revision": revision,
        "source_type": source_type,
        "source_ref": source_ref,
        "messages": messages,
        "content": raw_content or None,
        "scope": scope,
    }
    if async_mode or bool(payload.get("async_mode", False)):
        job = client.enqueue_event(**kwargs)
        return {
            "continue": True,
            "ingested": True,
            "async": True,
            "job": job.model_dump(mode="json"),
        }
    res = client.ingest_event(**kwargs)
    return {
        "continue": True,
        "ingested": True,
        "async": False,
        "event_id": res.event_id,
        "deduped_event": res.deduped_event,
        "frame_ids": [f.id for f in res.frames],
    }


def run_hook_cli(
    mode: str,
    raw_stdin: str,
    *,
    client: EmbeddedMemoryClient | GluoMemClient,
    default_scope: MemoryScope = DEFAULT_SCOPE,
    k: int = 5,
    profile: str = "balanced",
    async_mode: bool = False,
) -> str:
    """Parse ``raw_stdin`` JSON, execute ``recall`` or ``capture``, and return JSON string."""
    payload: dict[str, Any] = json.loads(raw_stdin) if raw_stdin.strip() else {}
    if mode == "recall":
        out = handle_recall_hook(
            client,
            payload,
            default_scope=default_scope,
            k=k,
            profile=profile,
        )
    elif mode == "capture":
        out = handle_capture_hook(
            client,
            payload,
            default_scope=default_scope,
            async_mode=async_mode,
        )
    else:
        raise ValueError(f"Unknown hook mode: {mode!r}")
    return json.dumps(out, ensure_ascii=False)
