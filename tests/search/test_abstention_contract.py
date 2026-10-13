"""Score contract: ranking and abstention are separate.

Deterministic fixtures with a keyword stub reranker and the mock embedder.
They cover supported evidence, weak evidence, no evidence and incompatible
calibration. No threshold here is a calibrated production value; every
decision records ``calibration == "uncalibrated"`` unless a test declares one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gluomem.config.app_config import AppConfig, load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.embedding.reranker import Reranker
from gluomem.frames.store import FrameStore
from gluomem.search.hybrid import HybridSearch, SearchResult
from gluomem.storage.db import MindDB

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class _Logits:
    """Fixed logit per doc keyword; negative by default like real cross-encoders."""

    def __init__(self, **by_word: float) -> None:
        self._by_word = by_word

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [
            next((v for w, v in self._by_word.items() if w in d), -9.0) for d in docs
        ]


def _cfg(**overrides: object) -> AppConfig:
    return load_config().with_overrides(overrides)


def _search(
    tmp_path: Path, cfg: AppConfig, reranker: Reranker | None, name: str
) -> tuple[MindDB, FrameStore, HybridSearch]:
    db = MindDB(tmp_path / f"{name}.mind", dim=cfg.storage.vector_dim)
    store = FrameStore(db, embedder=MockEmbedder(dim=db.dim), chunking=cfg.chunking)
    store.create_i_frame("g", "alpha deploy window is friday")
    store.create_i_frame("g", "beta rollback plan uses blue green")
    return db, store, HybridSearch(db, store, cfg, reranker=reranker)


def _both(
    tmp_path: Path, query: str, **overrides: object
) -> tuple[SearchResult, SearchResult]:
    """Run ``query`` with rerank off and on (negative logits) over equal stores."""
    off = _cfg(**{**overrides, "search.rerank.enabled": False})
    on = _cfg(**{**overrides, "search.rerank.enabled": True})
    db1, _s1, s_off = _search(tmp_path, off, None, "off")
    db2, _s2, s_on = _search(tmp_path, on, _Logits(alpha=-2.0, beta=-6.0), "on")
    try:
        return s_off.search(query), s_on.search(query)
    finally:
        db1.close()
        db2.close()


# --- supported / weak / no evidence, rerank on and off ----------------------


def test_supported_evidence_decides_the_same_with_rerank_on_and_off(
    tmp_path: Path,
) -> None:
    """Default contract: fusion score, threshold 0.0. Negative logits from the
    reranker reorder hits but cannot make supported evidence abstain."""
    off, on = _both(tmp_path, "alpha deploy window")
    for res in (off, on):
        assert res.abstained is False and res.hits
        d = res.abstention
        assert d is not None
        assert (d.reason, d.score, d.calibration) == ("sufficient", "fusion", "uncalibrated")
        assert d.score_identity == "fusion/rrf_k=60/profile=balanced"
    assert on.abstention.top_score == pytest.approx(off.abstention.top_score)  # type: ignore[union-attr]
    assert on.hits[0].rerank == pytest.approx(-2.0)
    assert on.hits[0].final == on.hits[0].rerank  # ranking uses the logit
    assert on.hits[0].fusion > 0.0  # abstention reads this, not final


def test_weak_evidence_abstains_on_both_paths_with_diagnostics(tmp_path: Path) -> None:
    off, on = _both(tmp_path, "alpha", **{"search.abstain.threshold": 10.0})
    for res in (off, on):
        assert res.abstained is True and res.hits == []
        d = res.abstention
        assert d is not None and d.reason == "below_threshold"
        assert d.top_score is not None and d.top_score < d.threshold == 10.0
        assert d.candidates >= 1


def test_no_eligible_evidence_is_a_recorded_abstention(tmp_path: Path) -> None:
    cfg = _cfg(**{"search.rerank.enabled": False})
    db = MindDB(tmp_path / "q.mind", dim=cfg.storage.vector_dim)
    store = FrameStore(db, embedder=MockEmbedder(dim=db.dim))
    store.create_i_frame("g", "alpha secret", trust="quarantined")
    res = HybridSearch(db, store, cfg).search("alpha")
    assert res.abstained is True
    assert res.abstention is not None
    assert (res.abstention.reason, res.abstention.candidates) == ("no_hits", 0)
    assert res.abstention.top_score is None

    # Gate off: still no hits, still diagnosed, not reported as an abstention.
    off = _cfg(**{"search.rerank.enabled": False, "search.abstain.enabled": False})
    res = HybridSearch(db, store, off).search("alpha")
    assert res.hits == [] and res.abstained is False
    assert res.abstention is not None and res.abstention.reason == "no_hits"
    db.close()


def test_disabled_gate_still_records_the_score(tmp_path: Path) -> None:
    off, on = _both(tmp_path, "alpha", **{"search.abstain.enabled": False})
    for res in (off, on):
        assert res.abstention is not None and res.abstention.reason == "disabled"
        assert res.abstention.top_score is not None and res.hits


# --- rerank-scale gate needs matching calibration ---------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"search.abstain.score": "rerank", "search.rerank.enabled": True}, "calibration"),
        (
            {
                "search.abstain.score": "rerank",
                "search.rerank.enabled": True,
                "search.abstain.calibration": "some-other-model",
            },
            "calibration",
        ),
        (
            {
                "search.abstain.score": "rerank",
                "search.rerank.enabled": False,
                "search.abstain.calibration": RERANKER_MODEL,
            },
            "rerank.enabled",
        ),
        (
            {
                "search.abstain.score": "rerank",
                "search.abstain.calibration": RERANKER_MODEL,
                "models.reranker.revision": "abc123",
            },
            "@abc123",
        ),
        ({"search.abstain.score": "logit"}, "invalid"),
    ],
    ids=["undeclared", "other-model", "rerank-off", "revision-not-declared", "unknown-score"],
)
def test_incompatible_calibration_is_rejected_at_load(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _cfg(**overrides)


def test_rerank_gate_with_matching_calibration_reads_logits(tmp_path: Path) -> None:
    contract = {
        "search.rerank.enabled": True,
        "search.abstain.score": "rerank",
        "search.abstain.calibration": RERANKER_MODEL,
        "search.abstain.threshold": -3.0,
    }
    cfg = _cfg(**contract)
    with pytest.raises(ValueError, match="needs a reranker"):
        _search(tmp_path, cfg, None, "none")

    db, _store, search = _search(tmp_path, cfg, _Logits(alpha=-2.0, beta=-6.0), "ok")
    res = search.search("alpha deploy window")
    d = res.abstention
    assert d is not None and not res.abstained
    assert (d.reason, d.score, d.calibration) == ("sufficient", "rerank", RERANKER_MODEL)
    assert d.score_identity == f"rerank/{RERANKER_MODEL}"
    assert d.top_score == pytest.approx(-2.0)
    db.close()

    db, _store, search = _search(tmp_path, cfg, _Logits(alpha=-4.0, beta=-6.0), "weak")
    res = search.search("alpha deploy window")
    assert res.abstained and res.abstention is not None
    assert res.abstention.reason == "below_threshold"
    db.close()


def test_pinned_revision_is_part_of_the_calibration_identity(tmp_path: Path) -> None:
    cfg = _cfg(
        **{
            "search.rerank.enabled": True,
            "models.reranker.revision": "abc123",
            "search.abstain.score": "rerank",
            "search.abstain.calibration": f"{RERANKER_MODEL}@abc123",
            "search.abstain.threshold": -3.0,
        }
    )
    db, _store, search = _search(tmp_path, cfg, _Logits(alpha=-2.0), "rev")
    d = search.search("alpha").abstention
    assert d is not None and d.score_identity == f"rerank/{RERANKER_MODEL}@abc123"
    db.close()
