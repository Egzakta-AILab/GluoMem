"""Provider-neutral MemoryToolPack for OpenAI, Anthropic, DSPy, LangGraph, and MCP."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from gluomem.storage.scope import MEMORY_KINDS, MemoryKind, MemoryScope

if TYPE_CHECKING:
    from collections.abc import Callable

    from gluomem.client.models import (
        CorrectResponse,
        ForgetResponse,
        IngestEventResponse,
        MemoryFrameModel,
        RecallResponse,
    )


@runtime_checkable
class MemoryClientProtocol(Protocol):
    """Common client contract satisfied by ``EmbeddedMemoryClient`` and ``GluoMemClient``."""

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
    ) -> RecallResponse: ...

    def ingest_event(self, *, scope: Any = None, **kwargs: Any) -> IngestEventResponse: ...

    def correct_memory(
        self, frame_id: int, new_content: str, *, scope: Any = None
    ) -> CorrectResponse: ...

    def forget_memory(self, frame_id: int, *, scope: Any = None) -> ForgetResponse: ...

    def get_lineage(self, frame_id: int, *, scope: Any = None) -> list[MemoryFrameModel]: ...

    def export_memories(self, *, scope: Any = None) -> dict[str, Any]: ...


_SCOPE_PROPERTIES: dict[str, Any] = {
    "type": "object",
    "description": "Optional isolation scope override (tenant_id, principal_id, namespace).",
    "properties": {
        "tenant_id": {"type": "string"},
        "principal_id": {"type": "string"},
        "namespace": {"type": "string"},
    },
    "additionalProperties": False,
}

TOOL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "memory_recall",
        "description": (
            "Search scoped long-term memory for facts, preferences, decisions, procedures, "
            "or episodes relevant to a query."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "k": {
                    "type": "integer",
                    "description": "Maximum number of memory hits to return (1-50).",
                    "default": 10,
                },
                "profile": {
                    "type": "string",
                    "description": "Scoring profile (e.g. balanced, precision).",
                    "default": "balanced",
                },
                "since": {
                    "type": "string",
                    "description": "Optional ISO-8601 lower bound on event_time.",
                },
                "until": {
                    "type": "string",
                    "description": "Optional ISO-8601 upper bound on event_time.",
                },
                "scope": _SCOPE_PROPERTIES,
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "memory_remember",
        "description": (
            "Store a durable memory statement or event in the active scope with full provenance."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Self-contained memory statement to remember.",
                },
                "kind": {
                    "type": "string",
                    "enum": list(MEMORY_KINDS),
                    "description": "Memory category (defaults to fact for explicit memories).",
                },
                "source_ref": {
                    "type": "string",
                    "description": "Source reference URI or session identifier.",
                    "default": "agent:tool",
                },
                "event_id": {
                    "type": "string",
                    "description": "Optional idempotency key for this write.",
                },
                "pinned": {
                    "type": "boolean",
                    "description": "Pin the memory so it does not expire.",
                    "default": False,
                },
                "require_review": {
                    "type": "boolean",
                    "description": "Hold the memory as a candidate in the review queue.",
                    "default": False,
                },
                "scope": _SCOPE_PROPERTIES,
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "memory_correct",
        "description": (
            "Supersede an existing memory frame with corrected content and scrub the old plaintext."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "frame_id": {
                    "type": "integer",
                    "description": "ID of the active or candidate memory frame to correct.",
                },
                "new_content": {
                    "type": "string",
                    "description": "Replacement statement.",
                },
                "scope": _SCOPE_PROPERTIES,
            },
            "required": ["frame_id", "new_content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "memory_forget",
        "description": (
            "Hard-erase a memory frame, its version lineage, orphaned KG edges, "
            "and record a tombstone."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "frame_id": {
                    "type": "integer",
                    "description": "ID of the memory frame to forget.",
                },
                "scope": _SCOPE_PROPERTIES,
            },
            "required": ["frame_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "memory_inspect_provenance",
        "description": (
            "Inspect the full version lineage and source provenance receipts for a memory frame."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "frame_id": {
                    "type": "integer",
                    "description": "ID of the memory frame to inspect.",
                },
                "scope": _SCOPE_PROPERTIES,
            },
            "required": ["frame_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "memory_export",
        "description": (
            "Export all active memories, provenance, lineage, and candidate review "
            "queue for a scope."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": _SCOPE_PROPERTIES,
            },
            "additionalProperties": False,
        },
    },
)


class MemoryToolPack:
    """Universal agentic tool adapter over ``EmbeddedMemoryClient`` or ``GluoMemClient``."""

    def __init__(
        self,
        client: MemoryClientProtocol,
        *,
        default_scope: MemoryScope | dict[str, str] | None = None,
    ) -> None:
        self._client = client
        self._default_scope = default_scope

    def _eff_scope(self, scope: Any = None) -> Any:
        return scope if scope is not None else self._default_scope

    def memory_recall(
        self,
        query: str,
        *,
        k: int = 10,
        profile: str = "balanced",
        since: str | None = None,
        until: str | None = None,
        scope: Any = None,
    ) -> dict[str, Any]:
        """Search scoped memory for relevant frames."""
        resp = self._client.recall(
            query,
            k=k,
            profile=profile,
            since=since,
            until=until,
            scope=self._eff_scope(scope),
        )
        return resp.model_dump(mode="json")

    def memory_remember(
        self,
        content: str,
        *,
        kind: MemoryKind | None = None,
        source_ref: str = "agent:tool",
        event_id: str | None = None,
        pinned: bool = False,
        require_review: bool = False,
        scope: Any = None,
    ) -> dict[str, Any]:
        """Store an explicit or extracted memory statement with provenance."""
        eff_event_id = event_id or (
            "mem-"
            + hashlib.sha256(f"{source_ref}:{content}:{uuid.uuid4().hex}".encode()).hexdigest()[:16]
        )
        resp = self._client.ingest_event(
            event_id=eff_event_id,
            source_type="explicit_memory",
            source_ref=source_ref,
            content=content,
            kind=kind or "fact",
            pinned=pinned,
            require_review=require_review,
            scope=self._eff_scope(scope),
        )
        return resp.model_dump(mode="json")

    def memory_correct(
        self,
        frame_id: int,
        new_content: str,
        *,
        scope: Any = None,
    ) -> dict[str, Any]:
        """Supersede an existing memory frame with corrected content."""
        resp = self._client.correct_memory(
            frame_id,
            new_content,
            scope=self._eff_scope(scope),
        )
        return resp.model_dump(mode="json")

    def memory_forget(
        self,
        frame_id: int,
        *,
        scope: Any = None,
    ) -> dict[str, Any]:
        """Hard-delete a memory frame and record a replay-suppression tombstone."""
        resp = self._client.forget_memory(
            frame_id,
            scope=self._eff_scope(scope),
        )
        return resp.model_dump(mode="json")

    def memory_inspect_provenance(
        self,
        frame_id: int,
        *,
        scope: Any = None,
    ) -> dict[str, Any]:
        """Return the version lineage and provenance records for a memory frame."""
        chain = self._client.get_lineage(frame_id, scope=self._eff_scope(scope))
        target = next((f for f in chain if f.id == frame_id), chain[-1] if chain else None)
        return {
            "frame_id": frame_id,
            "frame": target.model_dump(mode="json") if target is not None else None,
            "provenance": [p.model_dump(mode="json") for p in target.provenance]
            if target is not None
            else [],
            "lineage": [f.model_dump(mode="json") for f in chain],
        }

    def memory_export(self, *, scope: Any = None) -> dict[str, Any]:
        """Export all active memories, provenance, and review queue for a scope."""
        return self._client.export_memories(scope=self._eff_scope(scope))

    def mcp_tools(self) -> list[dict[str, Any]]:
        """Return MCP-formatted tool definitions."""
        return [dict(spec) for spec in TOOL_SPECS]

    def openai_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI function-calling tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["inputSchema"],
                },
            }
            for spec in TOOL_SPECS
        ]

    def anthropic_tools(self) -> list[dict[str, Any]]:
        """Return Anthropic Messages API tool definitions."""
        return [
            {
                "name": spec["name"],
                "description": spec["description"],
                "input_schema": spec["inputSchema"],
            }
            for spec in TOOL_SPECS
        ]

    def as_callables(self) -> list[Callable[..., dict[str, Any]]]:
        """Return bound Python callables for DSPy ReAct or LangGraph ToolNode."""
        return [
            self.memory_recall,
            self.memory_remember,
            self.memory_correct,
            self.memory_forget,
            self.memory_inspect_provenance,
            self.memory_export,
        ]

    def dispatch(self, name: str, arguments: dict[str, Any] | str | None = None) -> dict[str, Any]:
        """Dispatch a tool call by name with JSON string or dict arguments."""
        if isinstance(arguments, str):
            parsed: dict[str, Any] = json.loads(arguments) if arguments.strip() else {}
        else:
            parsed = dict(arguments or {})

        match name:
            case "memory_recall":
                return self.memory_recall(
                    query=str(parsed["query"]),
                    k=int(parsed.get("k", 10)),
                    profile=str(parsed.get("profile", "balanced")),
                    since=parsed.get("since"),
                    until=parsed.get("until"),
                    scope=parsed.get("scope"),
                )
            case "memory_remember":
                return self.memory_remember(
                    content=str(parsed["content"]),
                    kind=parsed.get("kind"),
                    source_ref=str(parsed.get("source_ref", "agent:tool")),
                    event_id=parsed.get("event_id"),
                    pinned=bool(parsed.get("pinned", False)),
                    require_review=bool(parsed.get("require_review", False)),
                    scope=parsed.get("scope"),
                )
            case "memory_correct":
                return self.memory_correct(
                    frame_id=int(parsed["frame_id"]),
                    new_content=str(parsed["new_content"]),
                    scope=parsed.get("scope"),
                )
            case "memory_forget":
                return self.memory_forget(
                    frame_id=int(parsed["frame_id"]),
                    scope=parsed.get("scope"),
                )
            case "memory_inspect_provenance":
                return self.memory_inspect_provenance(
                    frame_id=int(parsed["frame_id"]),
                    scope=parsed.get("scope"),
                )
            case "memory_export":
                return self.memory_export(scope=parsed.get("scope"))
            case _:
                raise KeyError(f"Unknown memory tool: {name!r}")
