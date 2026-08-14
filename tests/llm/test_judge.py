"""QAJudge tests — typed Verdict under DummyLM, registry-name backbone swap."""

from __future__ import annotations

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from gluomem.config.app_config import load_config
from gluomem.llm.judge import QAJudge, build_judge_module
from gluomem.llm.outputs import Verdict
from gluomem.llm.router import LMRouter


def test_judge_returns_typed_verdict() -> None:
    lm = DummyLM(
        [
            {
                "reasoning": "Exact match.",
                "verdict": {"correct": True, "score": 1.0, "rationale": "matches gold"},
            }
        ]
    )
    judge = QAJudge(LMRouter(load_config()), "sonnet5", lm=lm)
    verdict = judge.judge("Where does Alice live?", "Boston", "Boston")
    assert isinstance(verdict, Verdict)
    assert verdict.correct is True and verdict.score == 1.0


def test_judge_backbone_is_registry_name() -> None:
    router = LMRouter(load_config())
    judge = QAJudge(router, "gpt5-mini", temperature=0.0)
    assert judge.model == "gpt5-mini"
    assert judge._lm.model == "openrouter/openai/gpt-5.4-mini"
    assert judge._lm.kwargs["temperature"] == 0.0
    with pytest.raises(KeyError):
        QAJudge(router, "not-a-model")


def test_judge_module_is_cot_over_judgeqa() -> None:
    module = build_judge_module()
    assert isinstance(module, dspy.ChainOfThought)
