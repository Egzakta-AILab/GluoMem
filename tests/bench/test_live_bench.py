"""LIVE bench smokes — @slow, excluded from the offline gate.

Run explicitly with:  uv run pytest -m slow tests/bench/test_live_bench.py
Requires OPENROUTER_API_KEY; spends (tiny amounts of) real tokens —
one answerer call + trio judging over two rows, well under $0.05.
"""

from __future__ import annotations

import os
from pathlib import Path

import dotenv
import pytest

from gluomem.config.app_config import load_config

dotenv.load_dotenv(Path(__file__).parents[2] / ".env", override=False)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set",
    ),
]


def test_live_answerer_grounds_on_context() -> None:
    from gluomem.llm.answer import Answerer

    answerer = Answerer(load_config())
    result = answerer(
        "Where does Bob live now?",
        "[2023-05-08] Bob: I live in Boston now.\n"
        "[2023-08-02] Bob: Update from me: I moved to Seattle last week.",
    )
    assert "seattle" in result.text.lower()
    assert result.prompt_tokens is None or result.prompt_tokens > 0


def test_live_trio_unanimous_on_exact_match() -> None:
    """All three judge backbones accept an exact-match prediction and reject
    a contradicting one — the trio-strict live sanity check."""
    from gluomem.bench.judge import SingleJudge, TrioStrictJudge
    from gluomem.bench.types import Prediction, Sample
    from gluomem.llm.judge import QAJudge
    from gluomem.llm.router import LMRouter

    cfg = load_config()
    router = LMRouter(cfg)
    sha = "live-smoke"
    trio = TrioStrictJudge(
        [
            SingleJudge(
                b,
                QAJudge(router, b).judge,
                model_slug=cfg.models[b].provider_model,
                dataset_sha=sha,
            )
            for b in cfg.benchmark.judge.backbones
        ]
    )
    sample = Sample(
        id="live:qa:0",
        conv_id="live",
        question="What pet did Alice adopt?",
        gold="a cat",
        category=4,
        category_name="single-hop",
    )
    correct = trio.score(sample, Prediction(sample_id=sample.id, answer="a cat"))
    assert correct.value == 1.0
    wrong = trio.score(
        sample, Prediction(sample_id=sample.id, answer="a golden retriever")
    )
    assert wrong.value == 0.0
