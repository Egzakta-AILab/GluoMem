"""Defaults tests for the AppConfig tree.

A freshly loaded ``AppConfig`` (no env/overrides) must reproduce the documented
defaults exactly. These tests drive the public ``load_config`` surface and read
attributes off the returned config object.
"""

from __future__ import annotations

import math

from gluomem.config.app_config import AppConfig, load_config


def _cfg() -> AppConfig:
    # No args ⇒ defaults < conf/config.yaml; no env, no overrides.
    return load_config()


def test_rrf_k_default_is_60() -> None:
    """search.rrf_k defaults to 60."""
    cfg = _cfg()
    assert cfg.search.rrf_k == 60


def test_scoring_half_life_and_flat_boost() -> None:
    """half_life_days defaults to 30, flat_boost_days to 7."""
    cfg = _cfg()
    assert cfg.scoring.half_life_days == 30
    assert cfg.scoring.flat_boost_days == 7


def test_storage_vector_dim_is_384() -> None:
    """storage.vector_dim defaults to 384 (MiniLM native dim)."""
    cfg = _cfg()
    assert cfg.storage.vector_dim == 384


def test_balanced_profile_weights_simplex() -> None:
    """Balanced-profile weights are exact and sum to 1.0."""
    cfg = _cfg()
    w = cfg.scoring.profiles["balanced"].weights
    assert w.temporal == 0.4
    assert w.popularity == 0.2
    assert w.contextual == 0.2
    assert w.importance == 0.2
    total = w.temporal + w.popularity + w.contextual + w.importance
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_importance_mult_map_exact() -> None:
    """Importance multiplier map matches the documented values exactly."""
    cfg = _cfg()
    im = cfg.scoring.importance_mult
    assert im == {
        "critical": 2.0,
        "important": 1.5,
        "normal": 1.0,
        "temporary": 0.7,
        "deprecated": 0.3,
    }


def test_dspy_section_defaults() -> None:
    """The dspy section is typed; extraction defaults to deepseek-flash."""
    cfg = _cfg()
    # No dspy import is required to access the section.
    assert cfg.dspy.default_model == "deepseek-flash"
    assert cfg.dspy.adapter == "chat"
    assert cfg.dspy.track_usage is True
    for task in ("fact_extraction", "event_extraction", "cognify", "contradiction"):
        assert cfg.dspy.tasks[task].model == "deepseek-flash"
    assert cfg.write.contradiction.model == "deepseek-flash"


def test_dspy_artifacts_defaults() -> None:
    """Artifact store config: dir set, nothing required, no pins."""
    cfg = _cfg()
    assert cfg.dspy.artifacts.dir == "artifacts/dspy"
    assert cfg.dspy.artifacts.require is False
    assert cfg.dspy.artifacts.pins == {}


def test_llm_registry_routes_via_openrouter() -> None:
    """Every LLM entry rides OpenRouter with the single OPENROUTER key.

    A model swap stays a registry-name change; the provider/gateway is uniform.
    """
    cfg = _cfg()
    llm_names = [
        "haiku", "sonnet", "gpt4o",
        "sonnet5", "gpt5-mini", "qwen3", "deepseek-flash", "glm5",
    ]
    for name in llm_names:
        entry = cfg.models[name]
        assert entry.provider == "openrouter", name
        assert entry.provider_model.startswith("openrouter/"), name
        assert entry.api_key_env == "OPENROUTER_API_KEY", name


def test_model_entry_extra_and_api_base_default_empty() -> None:
    """ModelEntry.extra (LiteLLM kwargs pass-through) defaults empty."""
    cfg = _cfg()
    for name in ("glm5", "deepseek-flash", "sonnet5"):
        assert cfg.models[name].extra == {}, name
        assert cfg.models[name].api_base is None, name


def test_reasoning_policy_default_off_and_predict_everywhere() -> None:
    """Experiment baseline: hidden reasoning off, every task module=predict."""
    cfg = _cfg()
    assert cfg.dspy.reasoning_enabled is False
    assert all(t.module == "predict" for t in cfg.dspy.tasks.values())


def test_no_arg_load_returns_appconfig_instance() -> None:
    cfg = _cfg()
    assert isinstance(cfg, AppConfig)


def test_write_m4_surface_defaults() -> None:
    """Write-section defaults: frame_model/backend/lanes/subject/raw_turns/cognify."""
    cfg = _cfg()
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


def test_safety_scanner_defaults() -> None:
    """safety.scanner ships a non-empty compile-checked marker list."""
    cfg = _cfg()
    assert len(cfg.safety.scanner.patterns) >= 4


def test_benchmark_m5_surface_defaults() -> None:
    """Benchmark-section defaults: trio judge, answer task, sample, SHA-pinned data."""
    cfg = _cfg()
    assert cfg.benchmark.judge.family == "trio_strict"
    # All three backbones resolve in the model registry.
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


def test_benchmark_mem0_defaults() -> None:
    """mem0 competitor config: same backbone, MiniLM embedder."""
    cfg = _cfg()
    assert cfg.benchmark.mem0.model == "deepseek-flash"
    assert cfg.benchmark.mem0.embedder_model == "minilm"
    assert cfg.benchmark.mem0.top_k is None  # falls back to answer.context_k
    assert cfg.benchmark.mem0.infer is True


def test_answer_task_default_deepseek_flash() -> None:
    """The answerer (experiment subject) defaults to deepseek-flash;
    a subject swap is one dotted override."""
    cfg = _cfg()
    assert cfg.dspy.tasks["answer"].model == "deepseek-flash"
    assert cfg.dspy.tasks["answer"].module == "predict"
    assert cfg.dspy.tasks["answer"].temperature == 0.0
    swapped = cfg.with_overrides({"dspy.tasks.answer.model": "sonnet5"})
    assert swapped.dspy.tasks["answer"].model == "sonnet5"
    assert all(0.0 <= p.weight <= 1.0 for p in cfg.safety.scanner.patterns)


def test_env_file_loads_secrets_without_overriding_real_env(
    tmp_path, monkeypatch
) -> None:
    """Keys from .env land in os.environ; a real env var always wins."""
    import os

    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENROUTER_API_KEY=sk-or-from-dotenv\nOTHER_SECRET=abc\n", encoding="utf-8"
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OTHER_SECRET", raising=False)
    load_config(env_file=env_file)
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-from-dotenv"
    assert os.environ["OTHER_SECRET"] == "abc"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OTHER_SECRET", raising=False)

    # override=False: an already-set real env var beats the file.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real-env")
    load_config(env_file=env_file)
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-real-env"
