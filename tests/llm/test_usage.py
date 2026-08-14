"""Usage accumulation — thread-safe merge, deltas, totals, judge harvest."""

from __future__ import annotations

from dspy.utils.dummies import DummyLM

from gluomem.config.app_config import load_config
from gluomem.llm.judge import QAJudge
from gluomem.llm.router import LMRouter
from gluomem.llm.usage import UsageAccumulator, total_tokens, usage_delta

SLUG = "openrouter/deepseek/deepseek-v4-flash"


def test_accumulator_merges_across_calls() -> None:
    acc = UsageAccumulator()
    acc.add({SLUG: {"prompt_tokens": 100, "completion_tokens": 20}})
    acc.add({SLUG: {"prompt_tokens": 50, "completion_tokens": 5}})
    acc.add(None)
    acc.add({})
    snap = acc.snapshot()
    assert snap == {SLUG: {"prompt_tokens": 150, "completion_tokens": 25}}
    # Snapshot is a copy — mutating it never touches the accumulator.
    snap[SLUG]["prompt_tokens"] = 0
    assert acc.snapshot()[SLUG]["prompt_tokens"] == 150


def test_accumulator_handles_missing_and_none_fields() -> None:
    acc = UsageAccumulator()
    acc.add({SLUG: {"prompt_tokens": None, "completion_tokens": 7}})
    acc.add({SLUG: {"total_tokens": 99}})  # unknown keys ignored
    assert acc.snapshot() == {SLUG: {"prompt_tokens": 0, "completion_tokens": 7}}


def test_usage_delta_and_totals() -> None:
    before = {SLUG: {"prompt_tokens": 100, "completion_tokens": 10}}
    after = {
        SLUG: {"prompt_tokens": 180, "completion_tokens": 25},
        "other/model": {"prompt_tokens": 5, "completion_tokens": 1},
    }
    delta = usage_delta(after, before)
    assert delta == {
        SLUG: {"prompt_tokens": 80, "completion_tokens": 15},
        "other/model": {"prompt_tokens": 5, "completion_tokens": 1},
    }
    assert total_tokens(delta) == (85, 16)
    assert usage_delta(before, before) == {}


def test_qa_judge_exposes_usage_accumulator() -> None:
    """DummyLM reports no usage, but the surface exists and stays empty
    rather than crashing — live judges fill it."""
    lm = DummyLM(
        [
            {
                "reasoning": "ok",
                "verdict": {"correct": True, "score": 1.0, "rationale": "r"},
            }
        ]
    )
    judge = QAJudge(LMRouter(load_config()), "haiku", lm=lm)
    judge.judge("q", "gold", "gold")
    assert isinstance(judge.usage.snapshot(), dict)
