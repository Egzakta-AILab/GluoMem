"""ModelRegistry tests.

``ModelRegistry`` wraps ``cfg.models`` and resolves a model by registry name,
returning its native dim, and raises a clear error on an unknown name.
"""

from __future__ import annotations

import pytest

from gluomem.config.app_config import load_config
from gluomem.config.registry import ModelRegistry, UnknownModelError


def _registry() -> ModelRegistry:
    return ModelRegistry(load_config())


def test_resolve_minilm_dim_384() -> None:
    """resolve('minilm').dim == 384."""
    reg = _registry()
    entry = reg.resolve("minilm")
    assert entry.dim == 384
    assert entry.provider == "inprocess"


def test_resolve_ollama_dim_768() -> None:
    """resolve('ollama').dim == 768."""
    reg = _registry()
    entry = reg.resolve("ollama")
    assert entry.dim == 768


def test_resolve_unknown_raises_clear_error() -> None:
    """An unknown name raises a clear error."""
    reg = _registry()
    with pytest.raises(UnknownModelError) as exc:
        reg.resolve("nope")
    assert "nope" in str(exc.value)


def test_dim_lookup_helper() -> None:
    """The dim lookup returns the native dim for a known model."""
    reg = _registry()
    assert reg.dim_of("minilm") == 384
    assert reg.dim_of("ollama") == 768


def test_names_lists_registered_models() -> None:
    reg = _registry()
    names = reg.names()
    assert "minilm" in names
    assert "ollama" in names
    assert "haiku" in names
