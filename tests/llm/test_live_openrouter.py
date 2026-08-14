"""LIVE OpenRouter smokes — @slow, excluded from the offline gate.

Run explicitly with:  uv run pytest -m slow tests/llm/test_live_openrouter.py
Requires OPENROUTER_API_KEY; spends (tiny amounts of) real tokens.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import dotenv
import pytest

from gluomem.config.app_config import load_config
from gluomem.llm.modules import DspyExtractors

# skipif evaluates at import time — pull the repo .env in first so a key
# stored there (not in the shell) still enables the live smokes.
dotenv.load_dotenv(Path(__file__).parents[2] / ".env", override=False)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set",
    ),
]


@pytest.fixture(scope="module")
def extractors() -> DspyExtractors:
    return DspyExtractors(load_config())


def test_live_fact_extraction(extractors: DspyExtractors) -> None:
    facts = extractors.extract_facts(
        "Alice moved to Boston in 2021. She works at Acme as a data engineer."
    )
    assert len(facts) >= 1
    assert all(f.text for f in facts)
    assert all(0.0 <= f.confidence <= 1.0 for f in facts)


def test_live_event_extraction_resolves_relative_date(
    extractors: DspyExtractors,
) -> None:
    events = extractors.extract_events(
        "We signed the contract yesterday.", datetime(2023, 5, 8)
    )
    assert len(events) >= 1
    dates = [e.event_date for e in events if e.event_date is not None]
    assert dates, "expected at least one event with a resolved date"
    assert any(d.date().isoformat() == "2023-05-07" for d in dates)


def test_live_cognify(extractors: DspyExtractors) -> None:
    result = extractors.cognify("Alice works at Acme in Boston.")
    assert len(result.entities) >= 2
    names = {e.name.lower() for e in result.entities}
    assert any("alice" in n for n in names)


def test_live_contradiction(extractors: DspyExtractors) -> None:
    verdict = extractors.check_contradiction(
        "Alice lives in Seattle.", "Alice lives in Boston."
    )
    assert verdict.is_contradiction is True
    same = extractors.check_contradiction(
        "Alice lives in Boston.", "Alice resides in Boston."
    )
    assert same.is_contradiction is False


def test_live_provider_pin_honored() -> None:
    """extra_body.provider routing reaches OpenRouter (allow_fallbacks off)."""
    cfg = load_config().with_overrides(
        {
            "models.deepseek-flash.extra": {
                "extra_body": {
                    "provider": {"allow_fallbacks": False, "require_parameters": True}
                }
            }
        }
    )
    fx = DspyExtractors(cfg)
    facts = fx.extract_facts("Bob likes tea.")
    assert isinstance(facts, list)
