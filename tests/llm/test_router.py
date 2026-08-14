"""LMRouter tests — registry resolution, kwargs pass-through, context binding.

Constructing dspy.LM is network-free; nothing here needs a key or a socket.
"""

from __future__ import annotations

import dspy
import pytest

from gluomem.config.app_config import load_config
from gluomem.llm.router import LMRouter, _make_adapter


def _router(overrides: dict | None = None) -> LMRouter:
    cfg = load_config()
    if overrides:
        cfg = cfg.with_overrides(overrides)
    return LMRouter(cfg)


def test_lm_for_builds_openrouter_lm() -> None:
    router = _router()
    lm = router.lm_for("fact_extraction")
    assert lm.model == "openrouter/deepseek/deepseek-v4-flash"
    assert lm.kwargs["temperature"] == 0.0
    assert lm.kwargs["max_tokens"] == 4096


def test_api_key_read_from_entry_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-123")
    lm = _router().lm_for("cognify")
    assert lm.kwargs["api_key"] == "sk-or-test-123"


def test_model_swap_is_a_registry_name_change() -> None:
    """Swapping the extraction model is a config-only registry-name change."""
    router = _router({"dspy.tasks.fact_extraction.model": "sonnet5"})
    lm = router.lm_for("fact_extraction")
    assert lm.model == "openrouter/anthropic/claude-sonnet-5"


def test_entry_extra_passes_through_to_lm_kwargs() -> None:
    provider_pin = {
        "order": ["deepseek"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    router = _router(
        {"models.deepseek-flash.extra": {"extra_body": {"provider": provider_pin}}}
    )
    lm = router.lm_for("fact_extraction")
    # Entry extra merges with the router-injected reasoning policy.
    extra_body = lm.kwargs["extra_body"]
    assert isinstance(extra_body, dict)
    assert extra_body["provider"] == provider_pin
    assert extra_body["reasoning"] == {"enabled": False}


def test_reasoning_policy_injected_for_every_openrouter_model() -> None:
    """Hidden reasoning bills 5-12x visible output and eats max_tokens
    unpredictably — the global policy turns it off on every OpenRouter-routed
    LM (all 5 backbones + haiku/sonnet support the param; OpenRouter ignores
    it for non-reasoning models like gpt-4o)."""
    router = _router()
    for name in ("deepseek-flash", "sonnet5", "gpt5-mini", "qwen3", "glm5", "haiku"):
        lm = router.lm_for_model(name)
        extra_body = lm.kwargs["extra_body"]
        assert isinstance(extra_body, dict), name
        assert extra_body["reasoning"] == {"enabled": False}, name


def test_reasoning_policy_none_sends_nothing() -> None:
    router = _router({"dspy.reasoning_enabled": None})
    lm = router.lm_for_model("deepseek-flash")
    assert "extra_body" not in lm.kwargs


def test_entry_level_reasoning_override_beats_policy() -> None:
    router = _router(
        {
            "models.deepseek-flash.extra": {
                "extra_body": {"reasoning": {"enabled": True}}
            }
        }
    )
    lm = router.lm_for_model("deepseek-flash")
    extra_body = lm.kwargs["extra_body"]
    assert isinstance(extra_body, dict)
    assert extra_body["reasoning"] == {"enabled": True}


def test_unknown_task_fails_loud() -> None:
    with pytest.raises(KeyError, match="unknown dspy task 'nope'"):
        _router().lm_for("nope")


def test_lm_for_model_registry_level() -> None:
    lm = _router().lm_for_model("gpt5-mini", temperature=0.7)
    assert lm.model == "openrouter/openai/gpt-5.4-mini"
    assert lm.kwargs["temperature"] == 0.7


def test_bind_sets_and_restores_context(monkeypatch: pytest.MonkeyPatch) -> None:
    # bind() is a live entry and preflights the key; a dummy value satisfies
    # the preflight — nothing here actually calls the provider.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-bind")
    router = _router()
    before = dspy.settings.lm
    with router.bind("contradiction"):
        bound = dspy.settings.lm
        assert bound is not None
        assert bound.model == "openrouter/deepseek/deepseek-v4-flash"
    assert dspy.settings.lm is before


def test_bind_fails_loud_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """bind() preflights the task model's api_key_env: a missing env var fails
    with an error naming it, instead of a provider 401 mid-run."""
    # Build the router FIRST: load_config() re-loads the project .env into
    # os.environ, so the delenv must come after construction.
    router = _router()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        with router.bind("contradiction"):
            pass  # pragma: no cover — bind must raise before entering


def test_bind_treats_empty_api_key_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty key cannot authenticate — preflight treats it as missing."""
    router = _router()
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        with router.bind("contradiction"):
            pass  # pragma: no cover — bind must raise before entering


def test_assert_api_key_passes_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-123")
    _router().assert_api_key("gpt5-mini")  # no raise


def test_adapter_mapping_fails_loud_on_unknown() -> None:
    assert isinstance(_make_adapter("chat"), dspy.ChatAdapter)
    assert isinstance(_make_adapter("json"), dspy.JSONAdapter)
    with pytest.raises(ValueError, match="unknown dspy.adapter"):
        _make_adapter("xml")
