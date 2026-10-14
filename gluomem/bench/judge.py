"""Single-backbone QA judge wrapper and offline fake judge."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from gluomem.bench.types import JudgeRecord, Prediction, ProtocolTag, Sample, Score
from gluomem.llm.outputs import Verdict

if TYPE_CHECKING:
    from collections.abc import Callable

    from gluomem.bench.types import JudgeFn

_log = logging.getLogger(__name__)

_FAKE_BANNER = (
    "!!! GLUOMEM BENCH IS USING THE FAKE JUDGE (DETERMINISTIC, NO LLM) — "
    "VERDICTS ARE SUBSTRING-MATCH, NOT MODEL-BASED !!!"
)
_fake_banner_emitted = False


class SingleJudge:
    """One judge backbone; every verdict lands in the sink as a ``JudgeRecord``."""

    family = "single"

    def __init__(
        self,
        backbone: str,
        judge_fn: JudgeFn,
        *,
        model_slug: str,
        dataset_sha: str,
        sink: Callable[[JudgeRecord], None] | None = None,
        rubric_for: Callable[[Sample], str | None] | None = None,
    ) -> None:
        self.backbone = backbone
        self._judge_fn = judge_fn
        self._model_slug = model_slug
        self._dataset_sha = dataset_sha
        self._sink = sink
        self._rubric_for = rubric_for
        self._tag = ProtocolTag(
            metric_family="qa_accuracy",
            judge_family="single",
            judge_count=1,
            backbone=backbone,
            dataset_sha=dataset_sha,
        )

    def score(self, s: Sample, p: Prediction) -> Score:
        rubric = self._rubric_for(s) if self._rubric_for is not None else None
        if rubric is None:
            verdict = self._judge_fn(s.question, s.gold, p.answer)
        else:
            verdict = self._judge_fn(s.question, s.gold, p.answer, rubric)
        if self._sink is not None:
            self._sink(
                JudgeRecord(
                    sample_id=s.id,
                    backbone=self.backbone,
                    model_slug=self._model_slug,
                    correct=verdict.correct,
                    score=verdict.score,
                    rationale=verdict.rationale,
                    question=s.question,
                    gold=s.gold,
                    prediction=p.answer,
                )
            )
        return Score(value=1.0 if verdict.correct else 0.0, tag=self._tag, n=1)


def fake_judge_fn(
    question: str,  # noqa: ARG001
    gold: str,
    prediction: str,
    rubric: str | None = None,  # noqa: ARG001
) -> Verdict:
    """Deterministic offline ``JudgeFn``: correct iff normalized gold appears in prediction."""
    global _fake_banner_emitted
    if not _fake_banner_emitted:
        print(_FAKE_BANNER, file=sys.stderr, flush=True)
        _log.warning(_FAKE_BANNER)
        _fake_banner_emitted = True
    correct = gold.strip().lower() in prediction.strip().lower()
    return Verdict(
        correct=correct,
        score=1.0 if correct else 0.0,
        rationale="fake judge: normalized substring match",
    )


__all__ = ["SingleJudge", "fake_judge_fn"]
