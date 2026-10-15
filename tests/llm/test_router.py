"""LMRouter, ConcurrencyGate, UsageAccumulator, and live OpenRouter smoke tests."""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path

import dotenv
import dspy
import pytest

from gluomem.config.app_config import load_config
from gluomem.llm.concurrency import ConcurrencyGate
from gluomem.llm.modules import DspyExtractors
from gluomem.llm.router import LMRouter, _make_adapter
from gluomem.llm.usage import UsageAccumulator, total_tokens, usage_delta

dotenv.load_dotenv(Path(__file__).parents[2] / ".env", override=False)
SLUG = "openrouter/deepseek/deepseek-v4-flash"


def _router(overrides: dict | None = None) -> LMRouter:
    cfg = load_config()
    if overrides:
        cfg = cfg.with_overrides(overrides)
    return LMRouter(cfg)


def test_lm_for_and_model_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-123")
    router = _router()
    lm = router.lm_for("fact_extraction")
    assert lm.model == SLUG
    assert lm.kwargs["temperature"] == 0.0
    assert lm.kwargs["max_tokens"] == 4096
    assert lm.kwargs["api_key"] == "sk-or-test-123"

    swapped = _router({"dspy.tasks.fact_extraction.model": "sonnet5"}).lm_for("fact_extraction")
    assert swapped.model == "openrouter/anthropic/claude-sonnet-5"

    reg_lm = router.lm_for_model("gpt5-mini", temperature=0.7)
    assert reg_lm.model == "openrouter/openai/gpt-5.4-mini"
    assert reg_lm.kwargs["temperature"] == 0.7

    with pytest.raises(KeyError, match="unknown dspy task 'nope'"):
        router.lm_for("nope")


def test_reasoning_policy_and_extra_passthrough() -> None:
    router = _router()
    for name in ("deepseek-flash", "sonnet5", "gpt5-mini", "qwen3", "glm5", "haiku"):
        lm = router.lm_for_model(name)
        assert lm.kwargs["extra_body"]["reasoning"] == {"enabled": False}

    none_lm = _router({"dspy.reasoning_enabled": None}).lm_for_model("deepseek-flash")
    assert "extra_body" not in none_lm.kwargs

    provider_pin = {"order": ["deepseek"], "allow_fallbacks": False}
    over_lm = _router(
        {
            "models.deepseek-flash.extra": {
                "extra_body": {
                    "provider": provider_pin,
                    "reasoning": {"enabled": True},
                }
            }
        }
    ).lm_for_model("deepseek-flash")
    assert over_lm.kwargs["extra_body"]["provider"] == provider_pin
    assert over_lm.kwargs["extra_body"]["reasoning"] == {"enabled": True}


def test_bind_and_api_key_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _router()
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-bind")
    router.assert_api_key("gpt5-mini")

    before = dspy.settings.lm
    with router.bind("contradiction"):
        assert dspy.settings.lm is not None
        assert dspy.settings.lm.model == SLUG
    assert dspy.settings.lm is before

    for bad_val in (None, ""):
        if bad_val is None:
            monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        else:
            monkeypatch.setenv("OPENROUTER_API_KEY", bad_val)
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            with router.bind("contradiction"):
                pass


def test_adapter_mapping() -> None:
    assert isinstance(_make_adapter("chat"), dspy.ChatAdapter)
    assert isinstance(_make_adapter("json"), dspy.JSONAdapter)
    with pytest.raises(ValueError, match="unknown dspy.adapter"):
        _make_adapter("xml")


def test_concurrency_gate_bounds_and_releases() -> None:
    with pytest.raises(ValueError, match="limit must be >= 1"):
        ConcurrencyGate(0)

    gate = ConcurrencyGate(3)
    lock = threading.Lock()
    active = 0
    peak = 0
    done = 0

    def worker() -> None:
        nonlocal active, peak, done
        with gate:
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with lock:
                active -= 1
        with lock:
            done += 1

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert 2 <= peak <= 3 and done == 12

    gate1 = ConcurrencyGate(1)
    with pytest.raises(RuntimeError), gate1:
        raise RuntimeError("boom")
    with gate1:
        pass


def test_usage_accumulator_deltas_and_judge_surface() -> None:
    acc = UsageAccumulator()
    acc.add({SLUG: {"prompt_tokens": 100, "completion_tokens": 20}})
    acc.add({SLUG: {"prompt_tokens": 50, "completion_tokens": None, "total_tokens": 99}})
    acc.add(None)
    acc.add({})
    snap = acc.snapshot()
    assert snap == {SLUG: {"prompt_tokens": 150, "completion_tokens": 20}}
    snap[SLUG]["prompt_tokens"] = 0
    assert acc.snapshot()[SLUG]["prompt_tokens"] == 150

    before = {SLUG: {"prompt_tokens": 100, "completion_tokens": 10}}
    after = {
        SLUG: {"prompt_tokens": 180, "completion_tokens": 25},
        "other/model": {"prompt_tokens": 5, "completion_tokens": 1},
    }
    delta = usage_delta(after, before)
    assert total_tokens(delta) == (85, 16)
    assert usage_delta(before, before) == {}

    fx = DspyExtractors(load_config())
    assert isinstance(fx.usage.snapshot(), dict)


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
def test_live_openrouter_smoke() -> None:
    fx = DspyExtractors(load_config())
    facts = fx.extract_facts("Alice moved to Boston in 2021.")
    assert len(facts) >= 1 and all(0.0 <= f.confidence <= 1.0 for f in facts)
    events = fx.extract_events("We signed the contract yesterday.", datetime(2023, 5, 8))
    assert any(
        e.event_date is not None and e.event_date.date().isoformat() == "2023-05-07" for e in events
    )
