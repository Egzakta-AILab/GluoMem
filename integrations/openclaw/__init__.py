"""OpenClaw host integration bundle (`hooks.json` and `mcp.json`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HOOKS_CONFIG_PATH = Path(__file__).parent / "hooks.json"
MCP_CONFIG_PATH = Path(__file__).parent / "mcp.json"


def load_openclaw_hooks_config() -> dict[str, Any]:
    """Load the bundled OpenClaw lifecycle hooks configuration."""
    return json.loads(HOOKS_CONFIG_PATH.read_text(encoding="utf-8"))


def load_openclaw_mcp_config() -> dict[str, Any]:
    """Load the bundled OpenClaw MCP configuration."""
    return json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8"))


__all__ = [
    "HOOKS_CONFIG_PATH",
    "MCP_CONFIG_PATH",
    "load_openclaw_hooks_config",
    "load_openclaw_mcp_config",
]
