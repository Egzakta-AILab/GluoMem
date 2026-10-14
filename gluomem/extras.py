"""Optional-dependency availability and install hints.

Core (SQLite, I/P/B frames, hybrid search, KG, scoring, fake extractors,
mock embedder) needs no model packages. Model-backed and external storage
features live behind optional extras declared in ``pyproject.toml``:

- ``llm`` — DSPy extraction, QA answering, LLM judge, offline optimization
- ``embeddings`` — in-process MiniLM embeddings and cross-encoder reranking
- ``postgres`` — PostgreSQL + pgvector storage backend
"""

from __future__ import annotations

import importlib.util

EXTRAS: dict[str, tuple[tuple[str, ...], str]] = {
    "llm": (
        ("dspy",),
        "DSPy extraction, QA answering, LLM judge, and prompt optimization",
    ),
    "embeddings": (
        ("sentence_transformers",),
        "in-process MiniLM embeddings and cross-encoder reranking",
    ),
    "postgres": (
        ("psycopg", "pgvector"),
        "PostgreSQL + pgvector storage backend",
    ),
}


class MissingExtraError(ImportError):
    """Raised when an optional extra is needed for the requested operation."""

    def __init__(self, extra: str, feature: str, missing: list[str]) -> None:
        self.extra = extra
        self.feature = feature
        self.missing = missing
        mods = ", ".join(missing)
        super().__init__(
            f"{feature} requires the '{extra}' extra ({mods} not installed). "
            f"Install with: pip install 'gluomem[{extra}]' "
            f"(in a checkout: uv sync --extra {extra})"
        )


def _missing(extra: str) -> list[str]:
    if extra not in EXTRAS:
        raise KeyError(f"unknown extra '{extra}'; known: {sorted(EXTRAS)}")
    modules, _ = EXTRAS[extra]
    return [m for m in modules if importlib.util.find_spec(m) is None]


def is_available(extra: str) -> bool:
    """Return True when every distribution for ``extra`` is importable."""
    return not _missing(extra)


def require(extra: str, feature: str | None = None) -> None:
    """Raise :class:`MissingExtraError` unless ``extra`` is installed."""
    missing = _missing(extra)
    if missing:
        _, default_feature = EXTRAS[extra]
        raise MissingExtraError(extra, feature or default_feature, missing)


def status() -> dict[str, bool]:
    """Installation status of every optional extra (no imports, no downloads)."""
    return {name: is_available(name) for name in EXTRAS}
