"""Bench types — frozen-ness, tag equality/hashing, dspy-free import."""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest

from gluomem.bench.types import (
    JudgeRecord,
    Prediction,
    ProtocolTag,
    RetrievedDoc,
    Sample,
    Score,
)


def _tag(**kw: object) -> ProtocolTag:
    base: dict[str, object] = {
        "metric_family": "qa_accuracy",
        "judge_family": "trio_strict",
        "judge_count": 3,
        "backbone": "gpt5-mini+haiku+qwen3",
        "dataset_sha": "abc123",
    }
    base.update(kw)
    return ProtocolTag(**base)  # type: ignore[arg-type]


def test_dataclasses_are_frozen() -> None:
    s = Sample(
        id="c1:qa:0",
        conv_id="c1",
        question="q",
        gold="g",
        category=4,
        category_name="single-hop",
        evidence=("D1:1",),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.gold = "tampered"  # type: ignore[misc]
    p = Prediction(sample_id=s.id, answer="a")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.answer = "tampered"  # type: ignore[misc]


def test_tag_equality_and_hash() -> None:
    assert _tag() == _tag()
    assert hash(_tag()) == hash(_tag())
    assert _tag(backbone="haiku") != _tag()
    # Tags are usable as dict/set keys (the aggregate-refusal mechanism).
    assert len({_tag(), _tag(), _tag(metric_family="precision@5")}) == 2


def test_score_and_judge_record_shape() -> None:
    sc = Score(value=1.0, tag=_tag(), n=1)
    assert sc.value == 1.0 and sc.n == 1
    rec = JudgeRecord(
        sample_id="c1:qa:0",
        backbone="haiku",
        model_slug="openrouter/anthropic/claude-haiku-4.5",
        correct=True,
        score=1.0,
        rationale="matches",
        question="q",
        gold="g",
        prediction="g indeed",
    )
    assert rec.correct is True


def test_retrieved_doc_defaults() -> None:
    d = RetrievedDoc(rank=0, text="Alice: hi", score=0.5)
    assert d.dia_id is None and d.frame_id is None


def test_import_bench_never_imports_dspy() -> None:
    """Package-level import stays dspy-free (offline rescore guarantee)."""
    code = "import gluomem.bench, sys; sys.exit(1 if 'dspy' in sys.modules else 0)"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
