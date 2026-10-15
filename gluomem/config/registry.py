"""Model registry: resolve named model entries from ``AppConfig.models``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig, ModelEntry


class UnknownModelError(KeyError):
    """Raised when a model name is not registered in ``cfg.models``."""


class ModelRegistry:
    """Lookup table for configured model entries."""

    def __init__(self, cfg: AppConfig) -> None:
        self._models: dict[str, ModelEntry] = dict(cfg.models)

    def resolve(self, name: str) -> ModelEntry:
        """Resolve a model name to its :class:`ModelEntry`."""
        try:
            return self._models[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._models)) or "(none)"
            raise UnknownModelError(f"Unknown model '{name}'; registered models: {known}") from exc

    def dim_of(self, name: str) -> int:
        """Return the native embedding dim of model ``name``."""
        entry = self.resolve(name)
        if entry.dim is None:
            raise ValueError(f"Model '{name}' has no native dim (provider={entry.provider})")
        return entry.dim

    def names(self) -> list[str]:
        """Sorted list of registered model names."""
        return sorted(self._models)

    def __contains__(self, name: object) -> bool:
        return name in self._models
