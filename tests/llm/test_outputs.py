"""Seam tests — gluomem/llm/outputs.py (dspy-free contract).

Asserts model validation bounds, protocol conformance of the offline
implementations, and — critically — that importing the seam never imports
dspy (the write path and offline gate depend on this).
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from gluomem.llm import (
    ContradictionVerdict,
    ExtractedEvent,
    ExtractedFact,
    Extractors,
    FakeExtractors,
    KGEntity,
    NullExtractors,
    Verdict,
)


def test_confidence_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        ExtractedFact(text="x", confidence=1.5)
    with pytest.raises(ValidationError):
        ExtractedEvent(text="x", confidence=-0.1)
    with pytest.raises(ValidationError):
        Verdict(correct=True, score=2.0, rationale="r")


def test_model_defaults() -> None:
    fact = ExtractedFact(text="Alice lives in Boston.")
    assert fact.confidence == 1.0
    event = ExtractedEvent(text="moved")
    assert event.event_date is None
    entity = KGEntity(entity_type="person", name="alice")
    assert entity.properties == {}
    verdict = ContradictionVerdict(is_contradiction=False)
    assert verdict.description == "" and verdict.confidence == 1.0


def test_offline_impls_conform_to_protocol() -> None:
    assert isinstance(FakeExtractors(banner=False), Extractors)
    assert isinstance(NullExtractors(), Extractors)


def test_importing_seam_never_imports_dspy() -> None:
    """`import gluomem.llm` must NOT pull dspy (fresh interpreter check)."""
    code = (
        "import sys; import gluomem.llm; "
        "sys.exit(1 if 'dspy' in sys.modules else 0)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
