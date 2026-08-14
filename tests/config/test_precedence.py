"""Precedence tests for the overlay loader + with_overrides.

Precedence (lowest → highest): field defaults < conf/config.yaml < variant
overlay < GLUOMEM_ env (``__`` nested delimiter) < with_overrides() dotted keys.

Each test drives the public ``load_config`` / ``with_overrides`` surface and
asserts the documented ordering holds end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gluomem.config.app_config import AppConfig, load_config


def test_env_overrides_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """GLUOMEM_SEARCH__RRF_K=80 beats the file's 60."""
    monkeypatch.setenv("GLUOMEM_SEARCH__RRF_K", "80")
    cfg = load_config()
    assert cfg.search.rrf_k == 80


def test_variant_overrides_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Variant beats file; untouched keys keep file defaults."""
    # Write a tmp variant overlay that only changes rrf_k.
    variants_dir = tmp_path / "variants"
    variants_dir.mkdir()
    (variants_dir / "fast.yaml").write_text("search:\n  rrf_k: 70\n", encoding="utf-8")

    # Ensure no env interferes.
    monkeypatch.delenv("GLUOMEM_SEARCH__RRF_K", raising=False)

    cfg = load_config(variant="fast", variants_dir=variants_dir)
    assert cfg.search.rrf_k == 70  # variant beat the file's 60
    assert cfg.scoring.half_life_days == 30  # untouched key keeps file default


def test_override_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """with_overrides() beats env."""
    monkeypatch.setenv("GLUOMEM_SEARCH__RRF_K", "80")
    cfg = load_config()
    assert cfg.search.rrf_k == 80  # env beat file

    overridden = cfg.with_overrides({"search.rrf_k": 99})
    assert isinstance(overridden, AppConfig)
    assert overridden.search.rrf_k == 99  # override beat env


def test_override_dotted_key_reaches_nested_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dotted override key resolves into the nested field (embedding.model)."""
    monkeypatch.setenv("GLUOMEM_SEARCH__RRF_K", "80")
    cfg = load_config()
    # Switching the embedder to ollama (dim 768) while keeping vector_dim at 384
    # would trigger the dim-mismatch validator. We override BOTH so it stays valid.
    overridden = cfg.with_overrides(
        {"embedding.model": "ollama", "storage.vector_dim": 768}
    )
    assert overridden.embedding.model == "ollama"
    assert overridden.storage.vector_dim == 768
