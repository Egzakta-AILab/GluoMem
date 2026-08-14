"""gluomem.bench — benchmark harness + baselines.

IMPORT DISCIPLINE: importing ``gluomem.bench`` NEVER imports dspy. The dspy
path (real Answerer/QAJudge) is reached only through function-local imports
inside ``runner.py`` when the configured backend is ``"dspy"``; ``rescore``,
``report`` and ``metrics`` stay dspy-free forever so ``bench rescore``
re-derives headlines offline.
"""

from __future__ import annotations

from gluomem.bench.types import (
    AnswerFn,
    DatasetAdapter,
    Judge,
    JudgeFn,
    JudgeRecord,
    Prediction,
    ProtocolTag,
    RetrievedDoc,
    Retriever,
    RunPaths,
    Sample,
    Score,
)

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
