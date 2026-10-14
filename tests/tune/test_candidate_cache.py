"""Tests for CandidateCache, CandidateRow, and pure offline rescore (`DESIGN.md` §9)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from gluomem.config.app_config import load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.hybrid import HybridSearch
from gluomem.storage.db import MindDB
from gluomem.tune import (
    CandidateCache,
    CandidateRow,
    ParamVector,
    QuerySpec,
    build_cache_for_db,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    rescore,
    rescore_scores,
)


def test_rescore_matches_hybrid_search_ranking_and_scores(tmp_path: Path) -> None:
    cfg = load_config().with_overrides(
        {"embedding.model": "mock", "components.allow_fake": True}
    )
    dim = int(cfg.storage.vector_dim)
    embedder = MockEmbedder(dim=dim)
    db = MindDB(tmp_path / "tune.mind", dim=dim)
    fixed_now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=UTC)
    clock = lambda: fixed_now  # noqa: E731

    try:
        store = FrameStore(db, embedder=embedder, chunking=cfg.chunking)
        kg = KnowledgeGraph(db)
        f1 = store.create_i_frame(
            "s1",
            "PostgreSQL 18 production migration window is scheduled for Friday.",
            importance="critical",
        )
        f2 = store.create_i_frame(
            "s1",
            "PostgreSQL connection pool settings use pgbouncer in transaction mode.",
            importance="important",
        )
        f3 = store.create_i_frame(
            "s1",
            "Frontend CSS tokens for dark mode.",
            importance="temporary",
        )
        ent = kg.create_entity("technology", "postgresql")
        kg.link_entity_frame(ent.id, f1.id)

        hs = HybridSearch(
            db, store, cfg, embedder=embedder, kg=kg, clock=clock
        )
        query = "PostgreSQL migration window"
        search_res = hs.search(query, limit=10)

        spec = QuerySpec(
            query_id="q1",
            query=query,
            relevant_doc_ids=(f"frame:{f1.id}",),
        )
        cache = build_cache_for_db(
            db,
            store,
            cfg,
            [spec],
            embedder=embedder,
            kg=kg,
            lane_fetch=20,
            clock=clock,
        )
        p = ParamVector.from_config(cfg, pool_size=10)
        rescored = rescore_scores(cache.rows("q1"), p)
        assert rescore(cache.rows("q1"), p) == [doc_id for doc_id, _ in rescored]

        assert [doc_id for doc_id, _ in rescored] == [
            f"frame:{h.frame.id}" for h in search_res.hits
        ]
        for (_, r_score), hit in zip(rescored, search_res.hits, strict=True):
            assert abs(r_score - hit.fusion) < 1e-9
        assert f3.id is not None and f2.id is not None
    finally:
        db.close()


def test_candidate_cache_save_and_load_roundtrip(tmp_path: Path) -> None:
    rows = [
        CandidateRow(
            doc_id="D1:1",
            fts_rank=0,
            vec_rank=2,
            days_since_created=2.0,
            access_count=4,
            bfs_distance=0,
            importance_level="critical",
            grade=1,
            frame_id=10,
            dia_id="D1:1",
        ),
        CandidateRow(
            doc_id="D1:2",
            fts_rank=1,
            vec_rank=0,
            days_since_created=45.0,
            access_count=0,
            bfs_distance=4,
            importance_level="normal",
            grade=0,
            frame_id=11,
            dia_id="D1:2",
        ),
    ]
    cache = CandidateCache(
        {"q1": rows},
        qrels={"q1": {"D1:1": 1, "D1:99": 1}},
    )
    cache_dir = tmp_path / "cache"
    cache.save(cache_dir)

    loaded = CandidateCache.load(cache_dir)
    assert loaded.query_ids() == ["q1"]
    assert loaded.rows("q1") == rows
    # Unretrieved gold doc D1:99 is preserved in qrels for honest recall/nDCG
    assert loaded.qrel("q1") == {"D1:1": 1, "D1:99": 1}
    assert loaded.manifest.digest() == cache.manifest.digest()


def test_ir_metrics_ndcg_recall_hit_mrr() -> None:
    qrel = {"doc_a": 2, "doc_b": 1}
    perfect = ["doc_a", "doc_b", "doc_c"]
    reversed_order = ["doc_b", "doc_a", "doc_c"]
    miss = ["doc_x", "doc_y"]

    assert abs(ndcg_at_k(perfect, qrel, k=3) - 1.0) < 1e-9
    assert 0.0 < ndcg_at_k(reversed_order, qrel, k=3) < 1.0
    assert ndcg_at_k(miss, qrel, k=3) == 0.0

    assert recall_at_k(perfect, qrel, k=1) == 0.5
    assert recall_at_k(perfect, qrel, k=2) == 1.0
    assert hit_at_k(perfect, qrel, k=1) == 1.0
    assert hit_at_k(miss, qrel, k=2) == 0.0
    assert mrr_at_k(["doc_x", "doc_b"], qrel, k=2) == 0.5
