"""gluomem.bench.judge — single + trio-strict judge composition (dspy-free).

A ``SingleJudge`` wraps one backbone's ``JudgeFn`` (the dspy-backed
``QAJudge.judge`` on the live path, ``fake_judge_fn`` offline) and emits a
self-contained ``JudgeRecord`` per verdict. ``TrioStrictJudge`` composes
exactly three: a sample is correct only on unanimous CORRECT.

The trio decision is derivable from the three per-backbone records alone —
``metrics.qa_scores_from_records`` re-derives it during ``bench rescore``
with zero model calls, and the singles double as the self-judge-inflation
measurement.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from gluomem.bench.types import JudgeRecord, Prediction, ProtocolTag, Sample, Score
from gluomem.llm.outputs import Verdict

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from gluomem.bench.types import JudgeFn

_log = logging.getLogger(__name__)

_FAKE_BANNER = (
    "!!! GLUOMEM BENCH IS USING THE FAKE JUDGE (DETERMINISTIC, NO LLM) — "
    "VERDICTS ARE SUBSTRING-MATCH, NOT MODEL-BASED !!!"
)
_fake_banner_emitted = False


class SingleJudge:
    """One judge backbone; every verdict lands in the sink as a JudgeRecord."""

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
        # Per-sample grading rubric selector (LongMemEval per-type grading).
        # None (default) = the plain 3-arg judge call.
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


class TrioStrictJudge:
    """Three independent backbones; correct iff unanimous."""

    family = "trio_strict"

    def __init__(self, members: Sequence[SingleJudge]) -> None:
        if len(members) != 3:
            raise ValueError(
                f"trio_strict requires exactly 3 member judges, got {len(members)}"
            )
        self._members = list(members)
        backbones = sorted(m.backbone for m in members)
        shas = {m._tag.dataset_sha for m in members}
        if len(shas) != 1:
            raise ValueError(f"member judges span datasets: {sorted(shas)}")
        self._tag = ProtocolTag(
            metric_family="qa_accuracy",
            judge_family="trio_strict",
            judge_count=3,
            backbone="+".join(backbones),
            dataset_sha=shas.pop(),
        )

    def score(self, s: Sample, p: Prediction) -> Score:
        # Members emit their own JudgeRecords — singles recorded for free.
        verdicts = [m.score(s, p) for m in self._members]
        unanimous = all(v.value == 1.0 for v in verdicts)
        return Score(value=1.0 if unanimous else 0.0, tag=self._tag, n=1)


def fake_judge_fn(
    question: str,  # noqa: ARG001
    gold: str,
    prediction: str,
    rubric: str | None = None,  # noqa: ARG001 - accepted for rubric-path parity
) -> Verdict:
    """Deterministic offline JudgeFn: correct iff normalized gold appears in
    the prediction. LOUD about not being a model."""
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


__all__ = ["SingleJudge", "TrioStrictJudge", "fake_judge_fn"]
