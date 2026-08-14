"""gluomem.llm.usage — thread-safe token-usage accumulation (dspy-free).

``dspy`` reports per-prediction usage as ``{model_slug: {prompt_tokens,
completion_tokens, ...}}``; extraction and judging fan out across threads,
so accumulation must be locked. Snapshots are plain dicts consumable by the
bench manifest and the ingest cost summary.
"""

from __future__ import annotations

import threading
from typing import Any


class UsageAccumulator:
    """Accumulates dspy ``get_lm_usage()`` dicts across threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_model: dict[str, dict[str, int]] = {}

    def add(self, usage: dict[str, Any] | None) -> None:
        if not usage:
            return
        with self._lock:
            for slug, stats in usage.items():
                if not isinstance(stats, dict):
                    continue
                acc = self._by_model.setdefault(
                    slug, {"prompt_tokens": 0, "completion_tokens": 0}
                )
                acc["prompt_tokens"] += int(stats.get("prompt_tokens", 0) or 0)
                acc["completion_tokens"] += int(
                    stats.get("completion_tokens", 0) or 0
                )

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Deep-copied ``{slug: {prompt_tokens, completion_tokens}}``."""
        with self._lock:
            return {slug: dict(stats) for slug, stats in self._by_model.items()}


def usage_delta(
    after: dict[str, dict[str, int]], before: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    """Per-model difference between two snapshots (new activity only)."""
    out: dict[str, dict[str, int]] = {}
    for slug, stats in after.items():
        prev = before.get(slug, {})
        delta = {
            key: stats.get(key, 0) - prev.get(key, 0)
            for key in ("prompt_tokens", "completion_tokens")
        }
        if any(delta.values()):
            out[slug] = delta
    return out


def total_tokens(usage: dict[str, dict[str, int]]) -> tuple[int, int]:
    """(prompt_tokens, completion_tokens) summed across models."""
    return (
        sum(s.get("prompt_tokens", 0) for s in usage.values()),
        sum(s.get("completion_tokens", 0) for s in usage.values()),
    )


__all__ = ["UsageAccumulator", "total_tokens", "usage_delta"]
