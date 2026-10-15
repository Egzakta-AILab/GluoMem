"""Tests for ``AppConfig``, ``ModelRegistry``, and ``load_config``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from gluomem.config import (
    AppConfig,
    ModelEntry,
    ModelRegistry,
    UnknownModelError,
    load_config,
)


def test_default_config_loads_all_sections_and_constants() -> None:
    cfg = load_config()
    assert cfg.components.allow_fake is False
    assert set(cfg.models) >= {
        "deepseek-flash",
        "gpt5-mini",
        "qwen3",
        "haiku",
        "sonnet",
        "minilm",
        "mock",
        "reranker",
    }
    assert cfg.storage.backend == "sqlite"
    assert cfg.storage.vector_dim == 384 == cfg.models[cfg.embedding.model].dim
    assert (
        cfg.chunking.max_chars,
        cfg.chunking.overlap_chars,
        cfg.chunking.min_chunk_chars,
    ) == (2000, 200, 1500)
    assert cfg.search.rrf_k == 60
    assert (
        cfg.search.abstain.enabled,
        cfg.search.abstain.score,
        cfg.search.abstain.threshold,
    ) == (True, "fusion", 0.0)
    assert cfg.search.default_profile == "balanced"
    assert set(cfg.scoring.profiles) >= {
        "balanced",
        "recent",
        "important",
        "connected",
    }
    w = cfg.scoring.profiles["balanced"].weights
    assert (w.temporal, w.popularity, w.contextual, w.importance) == (
        0.4,
        0.2,
        0.2,
        0.2,
    )
    assert (cfg.scoring.half_life_days, cfg.scoring.flat_boost_days) == (30, 7)
    assert cfg.scoring.log_coef == 0.1
    assert cfg.scoring.bfs_steps == [1.0, 0.7, 0.4, 0.2, 0.0]
    assert cfg.scoring.importance_mult == {
        "critical": 2.0,
        "important": 1.5,
        "normal": 1.0,
        "temporary": 0.7,
        "deprecated": 0.3,
    }
    assert (
        cfg.write.compaction.p_frame_trigger,
        cfg.write.compaction.keep_recent,
    ) == (10, 5)
    assert cfg.write.lanes.fact.importance == "normal"
    assert cfg.write.lanes.profile.importance == "important"
    assert (cfg.kg.seed_min_len, cfg.kg.bfs_symmetric) == (2, True)
    assert "k8s" in cfg.kg.tech_allowlist
    assert len(cfg.safety.scanner.patterns) >= 5
    assert cfg.dspy.default_model == "deepseek-flash"
    assert set(cfg.dspy.tasks) >= {
        "fact_extraction",
        "event_extraction",
        "cognify",
        "contradiction",
        "answer",
    }
    assert cfg.benchmark.judge.family == "trio_strict"
    assert cfg.benchmark.judge.backbones == ["haiku", "gpt5-mini", "qwen3"]
    assert cfg.benchmark.mem0.model == "deepseek-flash"
    assert cfg.benchmark.mem0.embedder_model == "minilm"
    assert cfg.benchmark.mem0.top_k is None
    assert cfg.benchmark.mem0.infer is True


def test_app_config_direct_construction_matches_load_config() -> None:
    direct = AppConfig()
    loaded = load_config()
    assert direct.model_dump() == loaded.model_dump()


def test_scoring_weights_simplex_validation() -> None:
    base = load_config()

    with pytest.raises(ValidationError, match="simplex"):
        base.with_overrides(
            {
                "scoring.profiles.balanced.weights": {
                    "temporal": 0.5,
                    "popularity": 0.5,
                    "contextual": 0.5,
                    "importance": 0.5,
                }
            }
        )

    with pytest.raises(ValidationError, match="negative weights"):
        base.with_overrides(
            {
                "scoring.profiles.balanced.weights": {
                    "temporal": -0.1,
                    "popularity": 0.5,
                    "contextual": 0.3,
                    "importance": 0.3,
                }
            }
        )


def test_cross_validation_errors() -> None:
    base = load_config()

    with pytest.raises(ValidationError, match="vector_dim"):
        base.with_overrides({"storage.vector_dim": 768})

    with pytest.raises(ValidationError, match="Unknown model reference"):
        base.with_overrides({"embedding.model": "nonexistent-embedder"})

    with pytest.raises(ValidationError, match="has no dim in the registry"):
        base.with_overrides({"benchmark.mem0.embedder_model": "reranker"})

    with pytest.raises(ValidationError, match="Unknown model reference"):
        base.with_overrides({"search.rerank.model": "no-such-reranker"})

    with pytest.raises(ValidationError, match="Unknown model reference"):
        base.with_overrides({"dspy.default_model": "unknown-llm"})

    with pytest.raises(ValidationError, match="Unknown model reference"):
        base.with_overrides({"dspy.tasks.cognify.model": "unknown-llm"})


def test_precedence_and_with_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLUOMEM_SEARCH__RRF_K", "77")
    monkeypatch.setenv("GLUOMEM_KG__SEED_MIN_LEN", "4")
    cfg = load_config()
    assert cfg.search.rrf_k == 77
    assert cfg.kg.seed_min_len == 4

    cfg2 = load_config(overrides={"search.rrf_k": 88, "kg": {"bfs_symmetric": False}})
    assert cfg2.search.rrf_k == 88
    assert cfg2.kg.seed_min_len == 4
    assert cfg2.kg.bfs_symmetric is False

    cfg3 = cfg2.with_overrides(
        {
            "scoring.profiles.balanced.weights": {
                "temporal": 0.0,
                "popularity": 1.0,
                "contextual": 0.0,
                "importance": 0.0,
            }
        }
    )
    assert cfg2.scoring.profiles["balanced"].weights.temporal == 0.4
    assert cfg3.scoring.profiles["balanced"].weights.popularity == 1.0


def test_dotenv_populates_os_environ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=sk-from-dotenv\n", encoding="utf-8")

    load_config(env_file=env_file)
    assert os.environ.get("OPENROUTER_API_KEY") == "sk-from-dotenv"

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-shell")
    env_file.write_text("OPENROUTER_API_KEY=sk-from-dotenv-2\n", encoding="utf-8")
    load_config(env_file=env_file)
    assert os.environ.get("OPENROUTER_API_KEY") == "sk-from-shell"


def test_model_registry_operations() -> None:
    cfg = load_config()
    reg = ModelRegistry(cfg)
    assert "deepseek-flash" in reg
    assert set(reg.names()) == set(cfg.models.keys())

    entry = reg.resolve("minilm")
    assert isinstance(entry, ModelEntry)
    assert (entry.provider, entry.dim) == ("inprocess", 384)
    assert reg.dim_of("minilm") == 384

    with pytest.raises(UnknownModelError, match="Unknown model 'missing'"):
        reg.resolve("missing")

    with pytest.raises(ValueError, match="has no native dim"):
        reg.dim_of("deepseek-flash")
