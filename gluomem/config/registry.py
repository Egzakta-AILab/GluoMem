"""gluomem.config.registry — ModelRegistry wrapping ``cfg.models``.

Exposes ``resolve(name) -> ModelEntry`` (raising a clear error on an unknown
name) and a dim lookup. The registry is a thin, config-backed lookup: it holds
no mutable state of its own.
"""

from __future__ import annotations

from gluomem.config.app_config import AppConfig, ModelEntry


class UnknownModelError(KeyError):
    """Raised when ``ModelRegistry.resolve`` is asked for an unknown name."""


class ModelRegistry:
    """Resolve model entries by registry name from an ``AppConfig``."""

    def __init__(self, cfg: AppConfig) -> None:
        self._models: dict[str, ModelEntry] = dict(cfg.models)

    def resolve(self, name: str) -> ModelEntry:
        """Return the ``ModelEntry`` registered under ``name``.

        Raises ``UnknownModelError`` (a ``KeyError`` subclass) with a clear
        message if the name is absent from the registry.
        """
        try:
            return self._models[name]
        except KeyError:
            known = ", ".join(sorted(self._models)) or "(none)"
            raise UnknownModelError(
                f"Unknown model '{name}'. Registered models: {known}"
            ) from None

    def dim_of(self, name: str) -> int:
        """Return the native embedding dim of model ``name``.

        Raises ``UnknownModelError`` if the name is unknown, and ``ValueError``
        if the model has no ``dim`` (e.g. an LLM or reranker).
        """
        entry = self.resolve(name)
        if entry.dim is None:
            raise ValueError(
                f"Model '{name}' has no native dim (provider={entry.provider})"
            )
        return entry.dim

    def names(self) -> list[str]:
        """Return the sorted list of registered model names."""
        return sorted(self._models)
