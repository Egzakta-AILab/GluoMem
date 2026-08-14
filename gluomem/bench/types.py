"""gluomem.bench.types — the harness contract, dspy-free.

Frozen dataclasses + Protocols shared by every bench module. ``ProtocolTag``
is the metric-discipline mechanism: every ``Score`` carries one, and
``MetricRegistry.aggregate`` (metrics.py) refuses to combine scores whose
tags differ — F1, Recall@k, MCQ-accuracy, LLM-judge-accuracy and nDCG never
share a column.

``JudgeRecord`` rows are self-contained (question/gold/prediction embedded)
so ``bench rescore`` can re-derive every headline from committed JSONL with
zero model calls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gluomem.llm.outputs import AnswerResult, Verdict
    from gluomem.write.ingest import Conversation

# (question, context) -> AnswerResult; (question, gold, prediction) -> Verdict.
# Both sides of the seam are dspy-free Pydantic (gluomem/llm/outputs.py); the
# dspy-backed implementations are built lazily in runner.py only.
AnswerFn = Callable[[str, str], "AnswerResult"]
# (question, gold, prediction) -> Verdict; rubric-aware judges additionally
# accept a 4th positional rubric string (SingleJudge only passes it when a
# rubric is set, so plain 3-arg callables keep working on the default path).
JudgeFn = Callable[..., "Verdict"]


@dataclass(frozen=True)
class Sample:
    """One benchmark QA row. ``id`` = ``{conv_id}:qa:{idx}`` (file order)."""

    id: str
    conv_id: str
    question: str
    gold: str
    category: int
    category_name: str
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievedDoc:
    """One retrieved context line; ``dia_id`` is None for extracted frames
    (facts/events/profile carry no turn identity — documented limitation)."""

    rank: int
    text: str
    score: float
    dia_id: str | None = None
    frame_id: int | None = None


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    answer: str
    retrieved: tuple[RetrievedDoc, ...] = ()
    abstained: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None


@dataclass(frozen=True)
class ProtocolTag:
    """Provenance stamp on every Score; incompatible tags never aggregate.

    ``backbone`` is ``"none"`` for objective metrics, a registry name for
    single judges, and the sorted ``+``-join for trio panels.
    """

    metric_family: str
    judge_family: str
    judge_count: int
    backbone: str
    dataset_sha: str


@dataclass(frozen=True)
class Score:
    value: float
    tag: ProtocolTag
    n: int


@dataclass(frozen=True)
class JudgeRecord:
    """One backbone's verdict on one sample — a row of judgments.jsonl."""

    sample_id: str
    backbone: str
    model_slug: str
    correct: bool
    score: float
    rationale: str
    question: str
    gold: str
    prediction: str


@dataclass(frozen=True)
class RunPaths:
    """Artifact layout of one bench run directory."""

    root: Path
    manifest: Path
    predictions: Path
    judgments: Path
    scores: Path
    report: Path


@runtime_checkable
class DatasetAdapter(Protocol):
    """A SHA-pinned benchmark dataset: verify, load QA samples, and expose
    the typed conversations + dia bridges the runner/retrievers consume
    (the harness requires more than the bare verify/load minimum)."""

    name: str

    @property
    def path(self) -> Path: ...

    @property
    def sha(self) -> str: ...

    def verify(self) -> None: ...

    def fetch(self, *, force: bool = False) -> Path: ...

    def load(self) -> list[Sample]: ...

    def eligible_counts(self) -> dict[int, int]: ...

    def conversations(self) -> list[Conversation]: ...

    def dia_content(self) -> dict[str, str]: ...

    def content_to_dias(self) -> dict[str, tuple[str, ...]]: ...


@runtime_checkable
class Retriever(Protocol):
    """Answers one Sample end-to-end (retrieve -> answer). gluomem + baselines
    all implement this so they run under the identical harness."""

    name: str

    def answer(self, s: Sample) -> Prediction: ...


@runtime_checkable
class Judge(Protocol):
    """Scores one (Sample, Prediction) pair into a tagged Score."""

    family: str

    def score(self, s: Sample, p: Prediction) -> Score: ...


__all__ = [
    "AnswerFn",
    "DatasetAdapter",
    "Judge",
    "JudgeFn",
    "JudgeRecord",
    "Prediction",
    "ProtocolTag",
    "RetrievedDoc",
    "Retriever",
    "RunPaths",
    "Sample",
    "Score",
]
