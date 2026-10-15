"""gluomem.llm.judge — QA judge over the JudgeQA signature.

One ``QAJudge`` = one backbone (a registry model name) judging
(question, gold, prediction) → typed ``Verdict``. The benchmark harness
composes three of these into the trio-strict family
(``benchmark.judge.backbones``) and persists per-row ``JudgeRecord`` JSONL
for offline rescore.

Judging is deliberately ``ChainOfThought`` on a strong model; temperature
defaults to 0.0 for reproducible verdicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy

from gluomem.llm.outputs import Verdict
from gluomem.llm.signatures import JudgeQA, JudgeQATyped

if TYPE_CHECKING:
    from gluomem.llm.router import LMRouter


def build_judge_module() -> dspy.Module:
    return dspy.ChainOfThought(JudgeQA)


def build_typed_judge_module() -> dspy.Module:
    return dspy.ChainOfThought(JudgeQATyped)


class QAJudge:
    """One judge backbone; ``model`` is a registry name (swap = config change)."""

    def __init__(
        self,
        router: LMRouter,
        model: str,
        *,
        temperature: float = 0.0,
        lm: dspy.BaseLM | None = None,
        track_usage: bool = True,
    ) -> None:
        from gluomem.llm.usage import UsageAccumulator

        # ``lm`` overrides the routed model — offline tests inject a DummyLM.
        self._lm: dspy.BaseLM = (
            lm if lm is not None else router.lm_for_model(model, temperature=temperature)
        )
        self._module = build_judge_module()
        self._typed_module = build_typed_judge_module()
        self._track_usage = track_usage
        self.model = model
        # Token accounting across every verdict (thread-safe; the bench
        # runner fans judging out). Harvested into the run manifest.
        self.usage = UsageAccumulator()

    def judge(
        self, question: str, gold: str, prediction: str, rubric: str | None = None
    ) -> Verdict:
        """``rubric=None`` (default) uses the generic lenient JudgeQA prompt;
        a rubric string routes through JudgeQATyped (per-question-type
        grading, e.g. LongMemEval)."""
        with dspy.context(lm=self._lm, track_usage=self._track_usage):
            if rubric is None:
                pred = self._module(question=question, gold=gold, prediction=prediction)
            else:
                pred = self._typed_module(
                    question=question,
                    gold=gold,
                    prediction=prediction,
                    rubric=rubric,
                )
        if self._track_usage:
            self.usage.add(pred.get_lm_usage())
        return pred.verdict
