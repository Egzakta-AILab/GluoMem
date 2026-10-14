"""Answerer and QAJudge tests under DummyLM."""

from __future__ import annotations

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from gluomem.config.app_config import load_config
from gluomem.llm.answer import Answerer, build_answer_module
from gluomem.llm.judge import QAJudge, build_judge_module
from gluomem.llm.outputs import AnswerResult, Verdict
from gluomem.llm.router import LMRouter
from gluomem.llm.signatures import (
    TASK_SIGNATURES,
    AnswerQA,
    AnswerQAPreferenceAware,
    JudgeQA,
)


def test_answerer_routing_module_and_style_flips() -> None:
    cfg = load_config()
    lm = DummyLM([{"answer": "Seattle"}])
    answerer = Answerer(cfg, lm=lm)
    res = answerer("Where does Bob live?", "[2023-08-02] Bob: I moved to Seattle.")
    assert isinstance(res, AnswerResult) and res.text == "Seattle"

    swapped = Answerer(
        cfg.with_overrides({"dspy.tasks.answer.model": "sonnet5"}),
        lm=DummyLM([{"answer": "x"}]),
    )
    assert swapped.model == "sonnet5"

    real_ans = Answerer(cfg, router=LMRouter(cfg))
    assert real_ans._lm.model == "openrouter/deepseek/deepseek-v4-flash"
    assert real_ans._lm.kwargs["temperature"] == 0.0

    default_mod = build_answer_module(cfg)
    assert isinstance(default_mod, dspy.Predict)
    assert default_mod.signature is not None
    assert default_mod.signature.instructions == AnswerQAPreferenceAware.instructions

    cot_mod = build_answer_module(cfg.with_overrides({"dspy.tasks.answer.module": "cot"}))
    assert isinstance(cot_mod, dspy.ChainOfThought)

    base_mod = build_answer_module(cfg.with_overrides({"benchmark.answer.style": "base"}))
    assert base_mod.signature is not None
    assert base_mod.signature.instructions == AnswerQA.instructions
    assert "answer" not in TASK_SIGNATURES and AnswerQA not in TASK_SIGNATURES.values()


def test_qajudge_verdict_backbone_and_calibration_docstring() -> None:
    assert "Grade leniently" in (JudgeQA.__doc__ or "")
    assert isinstance(build_judge_module(), dspy.ChainOfThought)

    router = LMRouter(load_config())
    lm = DummyLM(
        [
            {
                "reasoning": "Exact match.",
                "verdict": {"correct": True, "score": 1.0, "rationale": "matches gold"},
            }
        ]
    )
    judge = QAJudge(router, "sonnet5", lm=lm)
    verdict = judge.judge("Where does Alice live?", "Boston", "Boston")
    assert isinstance(verdict, Verdict) and verdict.correct and verdict.score == 1.0

    gpt_judge = QAJudge(router, "gpt5-mini", temperature=0.0)
    assert gpt_judge.model == "gpt5-mini"
    assert gpt_judge._lm.model == "openrouter/openai/gpt-5.4-mini"

    with pytest.raises(KeyError):
        QAJudge(router, "not-a-model")
