"""Configuration defaults, packaged YAML parity, and overlay precedence tests."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from gluomem.config.app_config import DEFAULT_CONFIG_PATH, AppConfig, load_config


def test_core_and_search_defaults() -> None:
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.storage.backend == "sqlite"
    assert cfg.storage.vector_dim == 384
    assert cfg.storage.postgres_schema == "gluomem"
    assert cfg.search.rrf_k == 60
    assert cfg.scoring.half_life_days == 30
    assert cfg.scoring.flat_boost_days == 7
    w = cfg.scoring.profiles["balanced"].weights
    assert (w.temporal, w.popularity, w.contextual, w.importance) == (0.4, 0.2, 0.2, 0.2)
    assert math.isclose(
        w.temporal + w.popularity + w.contextual + w.importance, 1.0, abs_tol=1e-9
    )
    assert cfg.scoring.importance_mult == {
        "critical": 2.0,
        "important": 1.5,
        "normal": 1.0,
        "temporary": 0.7,
        "deprecated": 0.3,
    }


def test_dspy_and_models_defaults() -> None:
    cfg = load_config()
    assert cfg.dspy.default_model == "deepseek-flash"
    assert cfg.dspy.adapter == "chat"
    assert cfg.dspy.track_usage is True
    assert cfg.dspy.reasoning_enabled is False
    assert all(t.module == "predict" for t in cfg.dspy.tasks.values())
    assert cfg.dspy.artifacts.dir == "artifacts/dspy"
    assert cfg.dspy.artifacts.require is False
    assert cfg.dspy.artifacts.pins == {}

    for task in ("fact_extraction", "event_extraction", "cognify", "contradiction", "answer"):
        assert cfg.dspy.tasks[task].model == "deepseek-flash"
    assert cfg.write.contradiction.model == "deepseek-flash"

    llm_names = (
        "haiku",
        "sonnet",
        "gpt4o",
        "sonnet5",
        "gpt5-mini",
        "qwen3",
        "deepseek-flash",
        "glm5",
    )
    for name in llm_names:
        entry = cfg.models[name]
        assert entry.provider == "openrouter", name
        assert entry.provider_model.startswith("openrouter/"), name
        assert entry.api_key_env == "OPENROUTER_API_KEY", name
        assert entry.extra == {}, name
        assert entry.api_base is None, name


def test_write_safety_and_benchmark_defaults() -> None:
    cfg = load_config()
    assert cfg.write.frame_model == "ipb"
    assert cfg.write.extraction_backend == "dspy"
    assert cfg.write.raw_turns.enabled is True
    assert cfg.write.raw_turns.importance == "temporary"
    assert cfg.write.lanes.fact.importance == "normal"
    assert cfg.write.lanes.event.importance == "normal"
    assert cfg.write.lanes.profile.importance == "important"
    assert cfg.write.subject.min_entity_overlap == 2
    assert cfg.write.cognify.enabled is True
    assert cfg.write.cognify.retire_conflicting_edges is True
    assert cfg.write.extraction_cache.mode == "off"
    assert cfg.write.extraction_cache.path == ""
    assert len(cfg.safety.scanner.patterns) >= 4
    assert all(0.0 <= p.weight <= 1.0 for p in cfg.safety.scanner.patterns)

    assert cfg.benchmark.judge.family == "trio_strict"
    assert cfg.benchmark.judge.backbones == ["haiku", "gpt5-mini", "qwen3"]
    assert cfg.benchmark.judge.also_record_single is True
    assert cfg.benchmark.judge.backend == "dspy"
    assert cfg.benchmark.answer.task == "answer"
    assert cfg.benchmark.answer.context_k == 50
    assert cfg.benchmark.answer.abstain_text == "I don't know."
    assert cfg.benchmark.retrieval.ks == [5, 10]
    assert cfg.benchmark.sample.per_category == 80
    assert cfg.benchmark.sample.categories == [1, 2, 3, 4]
    assert cfg.benchmark.sample.seed == 42
    locomo = cfg.benchmark.datasets["locomo"]
    assert locomo.sha256 == (
        "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
    )
    assert locomo.file == "locomo10.json"
    assert cfg.benchmark.full_context.keep == "tail"
    assert cfg.benchmark.parallelism == 8
    assert cfg.benchmark.ci_gate.metric == "evidence_hit@5"
    assert cfg.benchmark.mem0.model == "deepseek-flash"
    assert cfg.benchmark.mem0.embedder_model == "minilm"
    assert cfg.benchmark.mem0.top_k is None
    assert cfg.benchmark.mem0.infer is True


def test_packaged_defaults_equal_repo_config(tmp_path: Path) -> None:
    assert DEFAULT_CONFIG_PATH.is_file()
    from_yaml = load_config().model_dump()
    without_yaml = load_config(path=tmp_path / "absent.yaml").model_dump()
    assert from_yaml == without_yaml


def test_overlay_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    variants_dir = tmp_path / "variants"
    variants_dir.mkdir()
    (variants_dir / "fast.yaml").write_text("search:\n  rrf_k: 70\n", encoding="utf-8")

    monkeypatch.delenv("GLUOMEM_SEARCH__RRF_K", raising=False)
    cfg_var = load_config(variant="fast", variants_dir=variants_dir)
    assert cfg_var.search.rrf_k == 70
    assert cfg_var.scoring.half_life_days == 30

    monkeypatch.setenv("GLUOMEM_SEARCH__RRF_K", "80")
    cfg_env = load_config(variant="fast", variants_dir=variants_dir)
    assert cfg_env.search.rrf_k == 80

    overridden = cfg_env.with_overrides(
        {"search.rrf_k": 99, "dspy.tasks.answer.model": "sonnet5"}
    )
    assert overridden.search.rrf_k == 99
    assert overridden.dspy.tasks["answer"].model == "sonnet5"


def test_env_file_loads_secrets_without_overriding_real_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENROUTER_API_KEY=sk-or-from-dotenv\nOTHER_SECRET=abc\n", encoding="utf-8"
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OTHER_SECRET", raising=False)
    load_config(env_file=env_file)
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-from-dotenv"
    assert os.environ["OTHER_SECRET"] == "abc"

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real-env")
    load_config(env_file=env_file)
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-real-env"
