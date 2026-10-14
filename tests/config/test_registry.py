"""ModelRegistry tests."""

from __future__ import annotations

import pytest

from gluomem.config.app_config import load_config
from gluomem.config.registry import ModelRegistry, UnknownModelError


def test_registry_resolve_dim_and_names() -> None:
    reg = ModelRegistry(load_config())
    entry = reg.resolve("minilm")
    assert entry.dim == 384
    assert entry.provider == "inprocess"
    assert reg.dim_of("minilm") == 384
    assert "minilm" in reg
    assert "haiku" in reg.names()

    with pytest.raises(ValueError, match="no native dim"):
        reg.dim_of("haiku")
    with pytest.raises(UnknownModelError, match="nope"):
        reg.resolve("nope")
