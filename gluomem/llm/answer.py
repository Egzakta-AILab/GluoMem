"""gluomem.llm.answer — benchmark answerer over the AnswerQA signature.

The answerer is the experiment SUBJECT: the model whose memory-grounded QA
accuracy the harness measures. It routes through ``cfg.dspy.tasks[<task>]``
(``benchmark.answer.task``, default ``"answer"``), so a subject swap is one
override: ``--override dspy.tasks.answer.model=sonnet5``.

Mirrors ``QAJudge``: registry-name routing, ``lm=`` injection for offline
DummyLM tests, ``dspy.context`` binding (never global configure).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy

from gluomem.llm.outputs import AnswerResult
from gluomem.llm.signatures import AnswerQA, AnswerQAPreferenceAware

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig
    from gluomem.llm.router import LMRouter


def build_answer_module(cfg: AppConfig) -> dspy.Module:
    task_cfg = cfg.dspy.tasks[cfg.benchmark.answer.task]
    signature = (
        AnswerQAPreferenceAware
        if cfg.benchmark.answer.style == "preference_aware"
        else AnswerQA
    )
    if task_cfg.module == "cot":
        return dspy.ChainOfThought(signature)
    return dspy.Predict(signature)


class Answerer:
    """One answer backbone; model comes from the configured answer task."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        router: LMRouter | None = None,
        lm: dspy.BaseLM | None = None,
    ) -> None:
        if lm is None:
            if router is None:
                from gluomem.llm.router import LMRouter as _LMRouter

                router = _LMRouter(cfg)
            lm = router.lm_for(cfg.benchmark.answer.task)
        self._lm: dspy.BaseLM = lm
        self._module = build_answer_module(cfg)
        self._track_usage = cfg.dspy.track_usage
        self.model = cfg.dspy.tasks[cfg.benchmark.answer.task].model

    def __call__(self, question: str, context: str) -> AnswerResult:
        with dspy.context(lm=self._lm, track_usage=self._track_usage):
            pred = self._module(question=question, context=context)
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        usage = pred.get_lm_usage() if self._track_usage else None
        if usage:
            # One entry per model slug; sum across (usually exactly one).
            prompt_tokens = sum(
                u.get("prompt_tokens", 0) or 0 for u in usage.values()
            )
            completion_tokens = sum(
                u.get("completion_tokens", 0) or 0 for u in usage.values()
            )
        return AnswerResult(
            text=str(pred.answer),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


__all__ = ["Answerer", "build_answer_module"]
