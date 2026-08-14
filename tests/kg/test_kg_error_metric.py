"""KG-lane error metric tests.

The KG contextual lane is wrapped so that if it RAISES, the error is RECORDED
on an observable counter (``SearchResult.kg_lane_errors``) and the lane
degrades to no-contextual — NEVER a silent zero indistinguishable from a
legitimate zero-contextual result. A normal run with no KG error leaves the
counter at 0.

Drives the public ``HybridSearch`` surface with a wired ``KnowledgeGraph``.
All tests use a temp ``*.mind`` + the deterministic ``MockEmbedder`` — ZERO
network, ZERO API calls.
"""

from __future__ import annotations

from pathlib import Path

from gluomem.config.app_config import load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.hybrid import HybridSearch
from gluomem.storage.db import MindDB


def _build(
    tmp_path: Path, *, kg: KnowledgeGraph | None
) -> tuple[MindDB, FrameStore, HybridSearch, KnowledgeGraph]:
    """Build a temp .mind + FrameStore + HybridSearch with an optional KG."""
    cfg = load_config()
    dim = int(cfg.storage.vector_dim)
    db = MindDB(tmp_path / "test.mind", dim=dim)
    embedder = MockEmbedder(dim=dim)
    store = FrameStore(db, embedder=embedder, chunking=cfg.chunking)
    if kg is None:
        kg = KnowledgeGraph(db)
    search = HybridSearch(
        db,
        store,
        cfg,
        embedder=embedder,
        reranker=None,
        kg=kg,
    )
    return db, store, search, kg


class _RaisingKG(KnowledgeGraph):
    """A KG whose ``contextual`` lane always raises (simulates a KG failure)."""

    def contextual(  # type: ignore[override]
        self,
        query: str,
        kg_cfg: object,
        *,
        recent_entities: list[int] | None = None,
        depth: int,
    ) -> dict[int, int]:
        raise RuntimeError("simulated KG contextual lane failure")


# --- error counter ----------------------------------------------------------


def test_kg_error_increments_counter(tmp_path: Path) -> None:
    """Forcing the contextual lane to raise increments the surfaced KG-lane
    error counter; the search degrades to ranked hits without contextual
    (never a hard crash, never a silent zero)."""
    cfg = load_config().with_overrides(
        {"search.rerank.enabled": False, "search.abstain.enabled": False}
    )
    dim = int(cfg.storage.vector_dim)
    db = MindDB(tmp_path / "test.mind", dim=dim)
    embedder = MockEmbedder(dim=dim)
    store = FrameStore(db, embedder=embedder, chunking=cfg.chunking)
    raising_kg = _RaisingKG(db)
    try:
        store.create_i_frame("gop-1", "alpha beta")
        search = HybridSearch(
            db, store, cfg, embedder=embedder, reranker=None, kg=raising_kg
        )

        res = search.search("alpha beta", limit=10)
        # The error was recorded on the observable counter — distinguishable
        # from a legitimate zero-contextual result.
        assert res.kg_lane_errors >= 1, (
            "a KG-lane failure must increment the surfaced error counter"
        )
        # The lane degraded to no-contextual: ranked hits still return.
        assert not res.abstained
        assert res.hits, "degraded lane must still return ranked hits"
    finally:
        db.close()


def test_clean_run_keeps_counter_zero(tmp_path: Path) -> None:
    """A normal run with no KG error leaves the counter at 0, even when the
    contextual map is empty (legitimate zero-contextual)."""
    cfg = load_config().with_overrides(
        {"search.rerank.enabled": False, "search.abstain.enabled": False}
    )
    dim = int(cfg.storage.vector_dim)
    db = MindDB(tmp_path / "test.mind", dim=dim)
    embedder = MockEmbedder(dim=dim)
    store = FrameStore(db, embedder=embedder, chunking=cfg.chunking)
    kg = KnowledgeGraph(db)
    try:
        store.create_i_frame("gop-1", "alpha beta")
        search = HybridSearch(
            db, store, cfg, embedder=embedder, reranker=None, kg=kg
        )

        res = search.search("alpha beta", limit=10)
        # No KG error -> counter stays 0 (distinguishable from an error run).
        assert res.kg_lane_errors == 0, (
            "a clean run with no KG error must leave the counter at 0"
        )
        assert res.hits
    finally:
        db.close()


def test_clean_run_empty_contextual_keeps_counter_zero(tmp_path: Path) -> None:
    """A legitimate zero-contextual result (KG present, query seeds nothing)
    keeps the counter at 0 — the observable distinction from a KG failure."""
    cfg = load_config().with_overrides(
        {"search.rerank.enabled": False, "search.abstain.enabled": False}
    )
    dim = int(cfg.storage.vector_dim)
    db = MindDB(tmp_path / "test.mind", dim=dim)
    embedder = MockEmbedder(dim=dim)
    store = FrameStore(db, embedder=embedder, chunking=cfg.chunking)
    kg = KnowledgeGraph(db)
    try:
        store.create_i_frame("gop-1", "alpha beta")
        # An entity that the query does NOT mention -> empty contextual map,
        # but NO error (the legitimate zero-contextual case).
        kg.create_entity("concept", "zzz_unrelated")
        search = HybridSearch(
            db, store, cfg, embedder=embedder, reranker=None, kg=kg
        )

        res = search.search("alpha beta", limit=10)
        assert res.kg_lane_errors == 0
        assert res.hits
    finally:
        db.close()


def test_no_kg_wired_keeps_counter_zero(tmp_path: Path) -> None:
    """When no KG is wired into HybridSearch the counter stays 0 (no lane to
    fail). The contextual signal is simply absent (distance beyond depth)."""
    cfg = load_config().with_overrides(
        {"search.rerank.enabled": False, "search.abstain.enabled": False}
    )
    dim = int(cfg.storage.vector_dim)
    db = MindDB(tmp_path / "test.mind", dim=dim)
    embedder = MockEmbedder(dim=dim)
    store = FrameStore(db, embedder=embedder, chunking=cfg.chunking)
    try:
        store.create_i_frame("gop-1", "alpha beta")
        search = HybridSearch(
            db, store, cfg, embedder=embedder, reranker=None, kg=None
        )

        res = search.search("alpha beta", limit=10)
        assert res.kg_lane_errors == 0
        assert res.hits
    finally:
        db.close()
