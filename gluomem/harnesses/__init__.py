"""Universal Agentic Harness Layer (MemoryToolPack and Hook adapters)."""

from gluomem.harnesses.hooks import handle_capture_hook, handle_recall_hook, run_hook_cli
from gluomem.harnesses.tools import MemoryClientProtocol, MemoryToolPack

__all__ = [
    "MemoryClientProtocol",
    "MemoryToolPack",
    "handle_capture_hook",
    "handle_recall_hook",
    "run_hook_cli",
]
