"""Hermes Agent integration bundle (`HermesMemoryAdapter` + MCP config)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gluomem.harnesses.tools import MemoryClientProtocol, MemoryToolPack
from gluomem.storage.scope import MemoryScope

MCP_CONFIG_PATH = Path(__file__).parent / "mcp_config.json"


class HermesMemoryAdapter:
    """Bind ``MemoryToolPack`` to a Hermes Agent session with scoped tool dispatch."""

    def __init__(
        self,
        client: MemoryClientProtocol,
        *,
        tenant_id: str = "local",
        principal_id: str = "default",
        namespace: str = "hermes",
    ) -> None:
        self._scope = MemoryScope(
            tenant_id=tenant_id,
            principal_id=principal_id,
            namespace=namespace,
        )
        self._pack = MemoryToolPack(client, default_scope=self._scope)

    @property
    def scope(self) -> MemoryScope:
        return self._scope

    @property
    def tool_pack(self) -> MemoryToolPack:
        return self._pack

    def openai_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI function-calling tool schemas for Hermes Agent."""
        return self._pack.openai_tools()

    def mcp_tools(self) -> list[dict[str, Any]]:
        """Return MCP tool descriptors for Hermes Agent."""
        return self._pack.mcp_tools()

    def dispatch(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        scope_override: MemoryScope | None = None,
    ) -> dict[str, Any]:
        """Execute a GluoMem tool call from Hermes Agent."""
        # Compatibility argument retained, but never authorizes scope escalation.
        # Pin at the boundary even with older MemoryToolPack implementations.
        args = dict(arguments or {})
        args["scope"] = self._scope.as_dict()
        return self._pack.dispatch(tool_name, args)


def load_hermes_mcp_config() -> dict[str, Any]:
    """Load the bundled Hermes MCP server configuration."""
    return json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8"))


__all__ = ["HermesMemoryAdapter", "MCP_CONFIG_PATH", "load_hermes_mcp_config"]
