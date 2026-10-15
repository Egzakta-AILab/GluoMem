"""Zero-dependency Model Context Protocol (MCP) JSON-RPC 2.0 server (stdio + HTTP)."""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from gluomem.harnesses.tools import MemoryClientProtocol, MemoryToolPack

if TYPE_CHECKING:
    from typing import TextIO

    from gluomem.storage.scope import MemoryScope

MCP_PROTOCOL_VERSION = "2024-11-05"


class McpServer:
    """Stateless JSON-RPC 2.0 MCP dispatcher over ``MemoryToolPack``."""

    def __init__(
        self,
        client: MemoryClientProtocol,
        *,
        default_scope: MemoryScope | dict[str, str] | None = None,
    ) -> None:
        self.toolpack = MemoryToolPack(client, default_scope=default_scope)

    def handle_message(
        self,
        message: dict[str, Any],
        *,
        scope_override: MemoryScope | dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Handle a single JSON-RPC 2.0 message and return a response dict (or None)."""
        req_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}

        if not isinstance(method, str):
            return _error_response(req_id, -32600, "Invalid Request: missing method")

        # Notifications have no "id" and expect no response.
        if "id" not in message or method.startswith("notifications/"):
            return None

        match method:
            case "initialize":
                requested_version = (
                    params.get("protocolVersion")
                    if isinstance(params, dict) and isinstance(params.get("protocolVersion"), str)
                    else MCP_PROTOCOL_VERSION
                )
                return _result_response(
                    req_id,
                    {
                        "protocolVersion": requested_version,
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {
                            "name": "gluomem",
                            "version": version("gluomem"),
                        },
                    },
                )
            case "ping":
                return _result_response(req_id, {})
            case "tools/list":
                return _result_response(req_id, {"tools": self.toolpack.mcp_tools()})
            case "tools/call":
                if not isinstance(params, dict) or "name" not in params:
                    return _error_response(req_id, -32602, "Invalid params: 'name' is required")
                tool_name = str(params["name"])
                raw_args = params.get("arguments") or {}
                if not isinstance(raw_args, dict):
                    return _error_response(
                        req_id, -32602, "Invalid params: 'arguments' must be an object"
                    )
                call_args = dict(raw_args)
                if scope_override is not None and "scope" not in call_args:
                    call_args["scope"] = scope_override
                try:
                    output = self.toolpack.dispatch(tool_name, call_args)
                except KeyError as exc:
                    return _error_response(req_id, -32602, f"Unknown tool: {exc}")
                except Exception as exc:  # noqa: BLE001
                    return _result_response(
                        req_id,
                        {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps({"error": str(exc)}, ensure_ascii=False),
                                }
                            ],
                            "isError": True,
                        },
                    )
                return _result_response(
                    req_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(output, ensure_ascii=False),
                            }
                        ],
                        "structuredContent": output,
                        "isError": False,
                    },
                )
            case _:
                return _error_response(req_id, -32601, f"Method not found: {method}")


def _result_response(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error_response(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def run_stdio_server(
    server: McpServer,
    *,
    instream: TextIO | None = None,
    outstream: TextIO | None = None,
) -> int:
    """Run the MCP server loop over newline-delimited JSON-RPC 2.0 stdio."""
    inp = instream if instream is not None else sys.stdin
    out = outstream if outstream is not None else sys.stdout
    for raw_line in inp:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            resp = _error_response(None, -32700, f"Parse error: {exc}")
            out.write(json.dumps(resp, ensure_ascii=False) + "\n")
            out.flush()
            continue
        if not isinstance(msg, dict):
            resp = _error_response(None, -32600, "Invalid Request: expected JSON object")
            out.write(json.dumps(resp, ensure_ascii=False) + "\n")
            out.flush()
            continue
        reply = server.handle_message(msg)
        if reply is not None:
            out.write(json.dumps(reply, ensure_ascii=False) + "\n")
            out.flush()
    return 0
