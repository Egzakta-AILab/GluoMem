"""Answerer tests — typed AnswerResult under DummyLM, task routing, module map."""

from __future__ import annotations

import dspy
from dspy.utils.dummies import DummyLM

from gluomem.config.app_config import load_config
from gluomem.llm.answer import Answerer, build_answer_module
from gluomem.llm.outputs import AnswerResult
from gluomem.llm.signatures import TASK_SIGNATURES, AnswerQA


def test_answerer_returns_typed_result() -> None:
    lm = DummyLM([{"answer": "Seattle"}])
    answerer = Answerer(load_config(), lm=lm)
    result = answerer("Where does Bob live?", "[2023-08-02] Bob: I moved to Seattle.")
    assert isinstance(result, AnswerResult)
    assert result.text == "Seattle"


def test_answerer_routes_via_answer_task() -> None:
    """Model comes from dspy.tasks.answer — a subject swap is one override."""
    cfg = load_config().with_overrides({"dspy.tasks.answer.model": "sonnet5"})
    answerer = Answerer(cfg, lm=DummyLM([{"answer": "x"}]))
    assert answerer.model == "sonnet5"


def test_answerer_builds_real_lm_from_registry() -> None:
    from gluomem.llm.router import LMRouter

    cfg = load_config()
    answerer = Answerer(cfg, router=LMRouter(cfg))
    # Default subject = deepseek-flash, temp 0.0 from the task entry.
    assert answerer._lm.model == "openrouter/deepseek/deepseek-v4-flash"
    assert answerer._lm.kwargs["temperature"] == 0.0


def test_module_flip_predict_to_cot() -> None:
    cfg = load_config()
    assert isinstance(build_answer_module(cfg), dspy.Predict)
    cot_cfg = cfg.with_overrides({"dspy.tasks.answer.module": "cot"})
    assert isinstance(build_answer_module(cot_cfg), dspy.ChainOfThought)


def test_answerqa_not_in_task_signatures() -> None:
    """Guard: `optimize --task all` iterates TASK_SIGNATURES and would demand
    an answer trainset; answer-prompt optimization is deliberately out of scope."""
    assert AnswerQA not in TASK_SIGNATURES.values()
    assert "answer" not in TASK_SIGNATURES


def test_judgeqa_docstring_carries_leniency_calibration() -> None:
    """The generosity sentence is deliberate prompt signal (Mem0-precedent
    comparability); drift here changes the judge protocol."""
    from gluomem.llm.signatures import JudgeQA

    doc = JudgeQA.__doc__ or ""
    assert "Grade leniently" in doc


def test_answer_style_flip_swaps_signature() -> None:
    """benchmark.answer.style is a pure config flip between the two prompts.
    preference_aware is the default; style=base selects the unmodified base
    prompt byte-identically."""
    from gluomem.llm.signatures import AnswerQAPreferenceAware

    default = build_answer_module(load_config())
    assert default.signature.instructions == AnswerQAPreferenceAware.instructions
    assert "never re-present" in default.signature.instructions
    base_cfg = load_config().with_overrides({"benchmark.answer.style": "base"})
    base = build_answer_module(base_cfg)
    assert base.signature.instructions == AnswerQA.instructions
