"""gluomem.extras — optional-dependency contract.

The core install covers config, storage, frames, search, the KG, fake
extractors and the mock embedder. Model-backed components live behind extras
and fail with :class:`MissingExtraError` naming the extra to install; they never
fall back to a fake or mock implementation because a package is missing.

Checks use ``importlib.util.find_spec`` so they import nothing heavy and
download nothing.
"""

from __future__ import annotations

import importlib.util

#: extra name -> (top-level modules it must provide, what it enables)
EXTRAS: dict[str, tuple[tuple[str, ...], str]] = {
    "llm": (("dspy",), "DSPy extraction, answerer, judges and prompt optimization"),
    "embeddings": (
        ("sentence_transformers",),
        "in-process sentence-transformers embedder and cross-encoder reranker",
    ),
    "mem0": (("mem0",), "mem0 comparison baseline"),
}


class MissingExtraError(ImportError):
    """A configured component needs an optional extra that is not installed."""

    def __init__(self, extra: str, feature: str, missing: list[str]) -> None:
        self.extra = extra
        self.feature = feature
        self.missing = missing
        super().__init__(
            f"{feature} requires the '{extra}' extra (missing: {', '.join(missing)}). "
            f"Install it with `pip install 'gluomem[{extra}]'` or "
            f"`uv sync --extra {extra}`."
        )


def _missing(extra: str) -> list[str]:
    modules, _ = EXTRAS[extra]
    return [m for m in modules if importlib.util.find_spec(m) is None]


def is_available(extra: str) -> bool:
    """True when every module the extra provides is importable."""
    return not _missing(extra)


def require(extra: str, feature: str) -> None:
    """Raise :class:`MissingExtraError` unless ``extra`` is installed."""
    missing = _missing(extra)
    if missing:
        raise MissingExtraError(extra, feature, missing)


def status() -> dict[str, bool]:
    """Installation status of every extra (no imports, no downloads)."""
    return {name: is_available(name) for name in EXTRAS}
