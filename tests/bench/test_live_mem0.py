"""LIVE mem0 smoke — @slow, needs OPENROUTER_API_KEY + `uv sync --extra mem0`.

Run explicitly:  uv run pytest -m slow tests/bench/test_live_mem0.py
Validates the pinned mem0ai API (from_config/add/search) and the litellm
usage tap against real traffic for pennies BEFORE any full paid ingest.
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

FIXTURE = Path(__file__).parent.parent / "fixtures" / "locomo_tiny.json"


def test_live_mem0_ingest_search_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("mem0", reason="uv sync --extra mem0 not installed")
    import hashlib

    from gluomem.bench.mem0 import Mem0Retriever, ingest_mem0_per_conversation
    from gluomem.llm.usage import total_tokens

    cfg = load_config().with_overrides(
        {
            "benchmark.datasets_dir": str(FIXTURE.parent),
            "benchmark.datasets.locomo.file": FIXTURE.name,
            "benchmark.datasets.locomo.sha256": hashlib.sha256(
                FIXTURE.read_bytes()
            ).hexdigest(),
        }
    )
    db_dir = tmp_path / "mem0_stores"
    (report,) = ingest_mem0_per_conversation(
        cfg, dataset="locomo", db_dir=db_dir
    )
    assert report.memories > 0, "mem0 extracted nothing from the fixture"
    prompt, completion = total_tokens(report.usage)
    assert prompt > 0, "usage tap captured no tokens (litellm callback broke?)"
    assert report.cost_usd is not None and report.cost_usd < 0.05

    # Retrieval round-trip through the real store.
    from gluomem.bench.types import Sample
    from gluomem.llm.answer import Answerer

    retriever = Mem0Retriever(cfg, db_dir=db_dir, answer_fn=Answerer(cfg))
    prediction = retriever.answer(
        Sample(
            id="conv-tiny-1:qa:0",
            conv_id="conv-tiny-1",
            question="Where does Bob live now?",
            gold="Seattle",
            category=4,
            category_name="single-hop",
        )
    )
    assert not prediction.abstained
    assert prediction.retrieved
