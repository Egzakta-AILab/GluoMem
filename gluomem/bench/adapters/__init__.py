"""gluomem.bench.adapters — dataset adapter registry.

Each adapter owns its dataset's parsing, SHA pinning and eligibility rules
behind the shared ``DatasetAdapter`` protocol (``gluomem/bench/types.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from gluomem.bench.adapters.locomo import DatasetError, LoCoMoAdapter
from gluomem.bench.adapters.longmemeval import LongMemEvalAdapter

if TYPE_CHECKING:
    from gluomem.bench.types import DatasetAdapter
    from gluomem.config.app_config import AppConfig

ADAPTERS: dict[str, Callable[[AppConfig], DatasetAdapter]] = {
    "locomo": LoCoMoAdapter,
    "longmemeval_oracle": lambda cfg: LongMemEvalAdapter(cfg, "longmemeval_oracle"),
    "longmemeval_s": lambda cfg: LongMemEvalAdapter(cfg, "longmemeval_s"),
}


def make_adapter(name: str, cfg: AppConfig) -> DatasetAdapter:
    factory = ADAPTERS.get(name)
    if factory is None:
        known = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"unknown dataset '{name}'; known: {known}")
    return factory(cfg)


__all__ = [
    "ADAPTERS",
    "DatasetError",
    "LoCoMoAdapter",
    "LongMemEvalAdapter",
    "make_adapter",
]
