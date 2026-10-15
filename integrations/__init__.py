"""Host integration bundles for Kvark, Hermes Agent, and OpenClaw."""

from __future__ import annotations

from integrations.hermes import HermesMemoryAdapter
from integrations.kvark import KvarkMemoryAdapter, kvark_scope

__all__ = ["HermesMemoryAdapter", "KvarkMemoryAdapter", "kvark_scope"]
