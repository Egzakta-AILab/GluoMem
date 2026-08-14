"""Fail-loud cross-validator tests.

Every misconfiguration must raise a *clear* error (a pydantic ValidationError
or an explicit ValueError), never a silently coerced value. The aligned
default config must load cleanly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gluomem.config.app_config import load_config


def test_default_config_loads_clean() -> None:
    """Aligned default config (384==384) loads with no error."""
    cfg = load_config()
    assert cfg.storage.vector_dim == 384
    assert cfg.embedding.model == "minilm"


def test_negative_rrf_k_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """rrf_k = -5 raises a validation error."""
    monkeypatch.setenv("GLUOMEM_SEARCH__RRF_K", "-5")
    with pytest.raises(ValidationError) as exc:
        load_config()
    msg = str(exc.value)
    assert "rrf_k" in msg.lower()


def test_nonsimplex_profile_fails() -> None:
    """Weights {0.5,0.5,0.5,0.5} (sum 2.0) raise."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides(
            {
                "scoring.profiles.balanced.weights": {
                    "temporal": 0.5,
                    "popularity": 0.5,
                    "contextual": 0.5,
                    "importance": 0.5,
                }
            }
        )
    msg = str(exc.value)
    assert "balanced" in msg.lower() or "sum" in msg.lower() or "simplex" in msg.lower()


def test_dim_mismatch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """vector_dim=512 vs minilm dim=384 raises, naming both dims."""
    monkeypatch.setenv("GLUOMEM_STORAGE__VECTOR_DIM", "512")
    with pytest.raises(ValidationError) as exc:
        load_config()
    msg = str(exc.value)
    # The error must name both dims so the operator can fix the mismatch.
    assert "512" in msg
    assert "384" in msg


def test_conv_parallelism_below_one_fails() -> None:
    """write.conv_parallelism must be >= 1 (0 = no workers is nonsense)."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"write.conv_parallelism": 0})
    assert "conv_parallelism" in str(exc.value)


def test_max_concurrent_llm_below_one_fails() -> None:
    """write.max_concurrent_llm must be >= 1 (the gate needs >=1 slot)."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"write.max_concurrent_llm": 0})
    assert "max_concurrent_llm" in str(exc.value)


def test_dedup_mode_must_be_exact_or_semantic() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"write.dedup.mode": "fuzzy"})
    assert "exact" in str(exc.value) and "semantic" in str(exc.value)


def test_dedup_threshold_out_of_range_fails() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.with_overrides({"write.dedup.similarity_threshold": 1.5})


def test_unknown_model_ref_fails() -> None:
    """embedding.model='does_not_exist' raises."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"embedding.model": "does_not_exist"})
    msg = str(exc.value)
    assert "does_not_exist" in msg


def test_bad_frame_model_fails() -> None:
    """write.frame_model outside {ipb,i_only,overwrite} raises."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"write.frame_model": "yolo"})
    assert "frame_model" in str(exc.value)


def test_extraction_cache_bad_mode_fails() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"write.extraction_cache.mode": "bogus"})
    assert "extraction_cache.mode" in str(exc.value)


def test_extraction_cache_non_off_requires_path() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"write.extraction_cache.mode": "record"})
    assert "path" in str(exc.value)


def test_extraction_cache_off_is_default_and_needs_no_path() -> None:
    cfg = load_config()
    assert cfg.write.extraction_cache.mode == "off"
    assert cfg.write.extraction_cache.path == ""


def test_kg_entity_type_mode_bad_value_fails() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"kg.entity_type_mode": "purge"})
    assert "entity_type_mode" in str(exc.value)


def test_kg_entity_type_mode_non_off_requires_allowlist() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides(
            {"kg.entity_type_mode": "drop", "kg.entity_type_allowlist": []}
        )
    assert "entity_type_allowlist" in str(exc.value)


def test_bad_fail_mode_fails() -> None:
    """safety.fail_mode outside {quarantine,reject} raises."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"safety.fail_mode": "ignore"})
    assert "fail_mode" in str(exc.value)


def test_bad_extraction_backend_fails() -> None:
    """write.extraction_backend outside {dspy,fake,none} raises."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"write.extraction_backend": "gpt"})
    assert "extraction_backend" in str(exc.value)


def test_bad_lane_importance_fails() -> None:
    """Lane importance not in scoring.importance_mult raises."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"write.lanes.profile.importance": "mega"})
    msg = str(exc.value)
    assert "mega" in msg and "importance" in msg


def test_zero_entity_overlap_fails() -> None:
    """subject.min_entity_overlap must be >= 1."""
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.with_overrides({"write.subject.min_entity_overlap": 0})


def test_noncompiling_scanner_regex_fails() -> None:
    """A scanner regex that doesn't compile fails at config load."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides(
            {"safety.scanner.patterns": [{"regex": "([unclosed", "weight": 1.0}]}
        )
    assert "compile" in str(exc.value)


def test_unknown_judge_backbone_fails() -> None:
    """A judge backbone missing from the registry raises, naming it.

    Validation happens at config load, not later at LM resolution time.
    """
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides(
            {"benchmark.judge.backbones": ["haiku", "gpt5-mini", "ollama-judge"]}
        )
    assert "ollama-judge" in str(exc.value)


def test_trio_strict_requires_three_backbones() -> None:
    """trio_strict with 2 backbones raises."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"benchmark.judge.backbones": ["haiku", "gpt5-mini"]})
    assert "trio_strict" in str(exc.value)


def test_duplicate_judge_backbones_fail() -> None:
    """Duplicate backbones raise (three independent families required)."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides(
            {"benchmark.judge.backbones": ["haiku", "haiku", "qwen3"]}
        )
    assert "unique" in str(exc.value)


def test_bad_judge_family_fails() -> None:
    """judge.family outside {trio_strict,single} raises."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"benchmark.judge.family": "quintet"})
    assert "family" in str(exc.value)


def test_bad_bench_backend_fails() -> None:
    """answer/judge backend outside {dspy,fake} raises."""
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.with_overrides({"benchmark.answer.backend": "real"})
    with pytest.raises(ValidationError):
        cfg.with_overrides({"benchmark.judge.backend": "llm"})


def test_unknown_answer_task_fails() -> None:
    """benchmark.answer.task must be a dspy.tasks key."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"benchmark.answer.task": "no_such_task"})
    assert "no_such_task" in str(exc.value)


def test_bad_sample_categories_fail() -> None:
    """Sample categories must be unique ints within 1..5."""
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.with_overrides({"benchmark.sample.categories": [1, 1, 2]})
    with pytest.raises(ValidationError):
        cfg.with_overrides({"benchmark.sample.categories": [0, 6]})
    with pytest.raises(ValidationError):
        cfg.with_overrides({"benchmark.sample.categories": []})


def test_bad_ci_gate_metric_fails() -> None:
    """Unparseable gate metric raises."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"benchmark.ci_gate.metric": "vibes"})
    assert "vibes" in str(exc.value)


def test_ci_gate_k_must_be_computed() -> None:
    """A gate @k outside benchmark.retrieval.ks is a config error."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"benchmark.ci_gate.metric": "evidence_hit@7"})
    assert "k=7" in str(exc.value)


def test_ci_gate_min_bounds() -> None:
    """Gate min outside [0,1] raises."""
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.with_overrides({"benchmark.ci_gate.min": 1.5})


def test_bad_full_context_keep_fails() -> None:
    """full_context.keep outside {head,tail} raises."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"benchmark.full_context.keep": "middle"})
    assert "keep" in str(exc.value)


def test_mem0_model_must_resolve() -> None:
    """mem0 backbone must be a registry model."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"benchmark.mem0.model": "gpt-99"})
    assert "gpt-99" in str(exc.value)


def test_mem0_embedder_requires_dim() -> None:
    """mem0 embedder needs a registry dim (faiss dims)."""
    cfg = load_config()
    # 'reranker' resolves in the registry but carries no dim.
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"benchmark.mem0.embedder_model": "reranker"})
    assert "dim" in str(exc.value)


def test_mem0_top_k_floor() -> None:
    """mem0.top_k must be >= 1 when set."""
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.with_overrides({"benchmark.mem0.top_k": 0})


def test_bad_subject_trigger_fails() -> None:
    """write.subject.trigger outside {entity_overlap,kg_edge_conflict,both}."""
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"write.subject.trigger": "vibes"})
    assert "trigger" in str(exc.value)


def test_contradiction_top_k_floor() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.with_overrides({"write.contradiction.top_k": 0})


def test_bad_dedup_on_hit_fails() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"write.dedup.on_hit": "merge"})
    assert "drop" in str(exc.value) and "keep_longest" in str(exc.value)


def test_bad_answer_style_fails() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"benchmark.answer.style": "chatty"})
    assert "preference_aware" in str(exc.value)


def test_predicate_families_reject_singleton_group() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides({"kg.predicate_families": [["LIVES_IN"]]})
    assert "predicate_families" in str(exc.value)


def test_predicate_families_must_be_disjoint() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides(
            {
                "kg.predicate_families": [
                    ["LIVES_IN", "MOVED_TO"],
                    ["moved_to", "RESIDES_IN"],  # case-insensitive collision
                ]
            }
        )
    assert "disjoint" in str(exc.value)


def test_entity_type_prompt_enum_requires_allowlist() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError) as exc:
        cfg.with_overrides(
            {
                "kg.entity_type_mode": "off",
                "kg.entity_type_allowlist": [],
                "kg.entity_type_prompt_enum": True,
            }
        )
    assert "entity_type_prompt_enum" in str(exc.value)


def test_cli_override_parses_json_lists() -> None:
    from gluomem.cli import _parse_override

    assert _parse_override("benchmark.sample.categories=[1,2,7]") == (
        "benchmark.sample.categories",
        [1, 2, 7],
    )
    import pytest as _pytest

    with _pytest.raises(Exception, match="JSON"):
        _parse_override("k=[1,2,")
