"""Contextual KG boost integration tests.

The KG contextual signal raises a KG-linked frame's final score/rank above an
otherwise-identical unlinked frame under a profile with ``contextual`` weight >
0. The linked candidate (bfs_distance 0 -> contextual 1.0) outranks the
unlinked one (contextual 0.0).

Drives the public ``HybridSearch`` surface with a wired ``KnowledgeGraph`` over
a temp ``*.mind`` + the deterministic ``MockEmbedder`` — ZERO network.
"""

from __future__ import annotations

from pathlib import Path

from pytest import approx

from gluomem.config.app_config import load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.hybrid import HybridSearch
from gluomem.storage.db import MindDB


def _build(
    tmp_path: Path,
) -> tuple[MindDB, FrameStore, HybridSearch, KnowledgeGraph]:
    cfg = load_config()
    dim = int(cfg.storage.vector_dim)
    db = MindDB(tmp_path / "test.mind", dim=dim)
    embedder = MockEmbedder(dim=dim)
    store = FrameStore(db, embedder=embedder, chunking=cfg.chunking)
    kg = KnowledgeGraph(db)
    search = HybridSearch(
        db, store, cfg, embedder=embedder, reranker=None, kg=kg
    )
    return db, store, search, kg


# --- contextual boost -------------------------------------------------------


def test_contextual_boost_linked_outranks_unlinked(tmp_path: Path) -> None:
    """A KG-linked frame (distance 0 -> contextual 1.0) outranks an
    otherwise-identical unlinked frame (contextual 0.0) under a profile with
    contextual weight > 0.

    Two frames with identical content in DIFFERENT GOPs (so dedup keeps both),
    identical created_at / access_count / importance -> the ONLY relevance
    difference is the contextual signal. The linked frame's relevance (and
    final) exceeds the unlinked frame's, and it ranks first.
    """
    cfg = load_config().with_overrides(
        {"search.rerank.enabled": False, "search.abstain.enabled": False}
    )
    dim = int(cfg.storage.vector_dim)
    db = MindDB(tmp_path / "test.mind", dim=dim)
    embedder = MockEmbedder(dim=dim)
    store = FrameStore(db, embedder=embedder, chunking=cfg.chunking)
    kg = KnowledgeGraph(db)
    search = HybridSearch(
        db, store, cfg, embedder=embedder, reranker=None, kg=kg
    )
    try:
        # Two identical-content frames in different GOPs (dedup is per-gop).
        linked = store.create_i_frame("gop-linked", "alpha beta")
        unlinked = store.create_i_frame("gop-unlinked", "alpha beta")

        # Link ONLY the first frame to a KG entity seeded by the query token.
        entity = kg.create_entity("concept", "alpha")
        kg.link_entity_frame(entity.id, linked.id)

        # Balanced profile has contextual weight 0.2 (> 0).
        weights = cfg.scoring.profiles[cfg.search.default_profile].weights
        assert weights.contextual > 0.0, "balanced profile must have contextual>0"

        res = search.search("alpha", limit=10)
        assert not res.abstained
        assert len(res.hits) == 2, f"expected both frames, got {len(res.hits)}"

        by_id = {h.frame.id: h for h in res.hits}
        h_linked = by_id[linked.id]
        h_unlinked = by_id[unlinked.id]

        # The linked frame's relevance is higher (contextual 1.0 vs 0.0); all
        # other signals are equal (same content/created_at/access/importance).
        assert h_linked.relevance > h_unlinked.relevance, (
            f"linked relevance {h_linked.relevance} must exceed unlinked "
            f"{h_unlinked.relevance} under contextual weight>0"
        )
        # The linked frame's final exceeds the unlinked frame's.
        assert h_linked.final > h_unlinked.final
        # The linked frame ranks FIRST.
        assert res.hits[0].frame.id == linked.id, (
            "the KG-linked frame must outrank the identical unlinked frame"
        )
        # No KG error on a clean run.
        assert res.kg_lane_errors == 0
    finally:
        db.close()


def test_contextual_boost_zeroed_equalizes(tmp_path: Path) -> None:
    """Under a profile with contextual weight == 0 the boost vanishes: two
    otherwise-identical frames score EQUAL relevance (the ablation gate).

    The weights are a valid 4-key simplex (sum 1.0) with contextual set to 0,
    so the contextual signal contributes nothing and the linked/unlinked
    relevance values are equal."""
    cfg = load_config().with_overrides(
        {
            "search.rerank.enabled": False,
            "search.abstain.enabled": False,
            "scoring.profiles.balanced.weights.temporal": 0.5,
            "scoring.profiles.balanced.weights.popularity": 0.25,
            "scoring.profiles.balanced.weights.contextual": 0.0,
            "scoring.profiles.balanced.weights.importance": 0.25,
        }
    )
    dim = int(cfg.storage.vector_dim)
    db = MindDB(tmp_path / "test.mind", dim=dim)
    embedder = MockEmbedder(dim=dim)
    store = FrameStore(db, embedder=embedder, chunking=cfg.chunking)
    kg = KnowledgeGraph(db)
    search = HybridSearch(
        db, store, cfg, embedder=embedder, reranker=None, kg=kg
    )
    try:
        linked = store.create_i_frame("gop-linked", "alpha beta")
        unlinked = store.create_i_frame("gop-unlinked", "alpha beta")
        entity = kg.create_entity("concept", "alpha")
        kg.link_entity_frame(entity.id, linked.id)

        res = search.search("alpha", limit=10)
        assert not res.abstained
        by_id = {h.frame.id: h for h in res.hits}
        # With contextual weight 0 the boost is gone: relevance equal.
        assert by_id[linked.id].relevance == approx(by_id[unlinked.id].relevance)
    finally:
        db.close()


def test_contextual_map_distance_zero_for_linked(tmp_path: Path) -> None:
    """The KG contextual map returns distance 0 for a frame linked to a seeded
    entity (the seed itself), and omits the unlinked frame entirely."""
    cfg = load_config()
    dim = int(cfg.storage.vector_dim)
    db = MindDB(tmp_path / "test.mind", dim=dim)
    store = FrameStore(db, embedder=MockEmbedder(dim=dim), chunking=cfg.chunking)
    kg = KnowledgeGraph(db)
    try:
        linked = store.create_i_frame("gop-linked", "alpha beta")
        unlinked = store.create_i_frame("gop-unlinked", "alpha beta")
        entity = kg.create_entity("concept", "alpha")
        kg.link_entity_frame(entity.id, linked.id)

        ctx = kg.contextual("alpha", cfg.kg, depth=cfg.scoring.bfs_depth)
        # Linked frame is at distance 0 (seed entity linked to it).
        assert ctx[linked.id] == 0
        # Unlinked frame is absent from the contextual map.
        assert unlinked.id not in ctx
    finally:
        db.close()
