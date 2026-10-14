"""Tests for CandidateCache, CandidateRow, and pure offline rescore (`DESIGN.md` §9)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from gluomem.config.app_config import load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.hybrid import HybridSearch
from gluomem.storage.db import MindDB
from gluomem.storage.scope import MemoryScope
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
from gluomem.tune.rescore import rescore_array_scores, rescore_cache


def test_rescore_matches_hybrid_search_ranking_and_scores(tmp_path: Path) -> None:
    cfg = load_config().with_overrides({"embedding.model": "mock", "components.allow_fake": True})
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

        hs = HybridSearch(db, store, cfg, embedder=embedder, kg=kg, clock=clock)
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


@pytest.mark.parametrize("workspace", ["personal", "workspace"])
@pytest.mark.parametrize("filtered", [False, True])
def test_scoped_cache_uses_runtime_expiry_clock_and_time_filters(
    tmp_path: Path, workspace: str, filtered: bool
) -> None:
    cfg = load_config().with_overrides({"embedding.model": "mock", "components.allow_fake": True})
    scope = MemoryScope("org", "alice", workspace)
    other = MemoryScope("org", "bob", workspace)
    now = "2026-06-01T12:00:00"
    embedder = MockEmbedder(dim=cfg.storage.vector_dim)
    with MindDB(tmp_path / "scoped.mind", dim=cfg.storage.vector_dim) as db:
        store = FrameStore(db, embedder=embedder, scope=other)
        good = store.create_i_frame(
            "s1",
            "PostgreSQL production database",
            scope=scope,
            created_at="2026-05-01T12:00:00",
            event_time="2026-05-10",
        )
        out_of_range = store.create_i_frame(
            "s1",
            "PostgreSQL previous release",
            scope=scope,
            event_time="2026-04-01",
        )
        expired = store.create_i_frame(
            "s1",
            "PostgreSQL expired setting",
            scope=scope,
            expires_at="2026-05-31",
        )
        deleted = store.create_i_frame(
            "s1",
            "PostgreSQL deleted setting",
            scope=scope,
            status="forgotten",
        )
        store.create_i_frame("s1", "PostgreSQL private setting", scope=other)
        since, until = ("2026-05-01", "2026-05-31") if filtered else (None, None)
        search = HybridSearch(db, store, cfg, embedder=embedder, kg=KnowledgeGraph(db))
        result = search.search("PostgreSQL", scope=scope, now_iso=now, since=since, until=until)
        cache = build_cache_for_db(
            db,
            store,
            cfg,
            [
                QuerySpec(
                    "q",
                    "PostgreSQL",
                    (f"frame:{good.id}",),
                    since=since,
                    until=until,
                    clock_iso=now,
                )
            ],
            scope=scope,
            clock=lambda: datetime(2099, 1, 1),
        )
        rows = cache.rows("q")
        assert (
            rows and next(row for row in rows if row.frame_id == good.id).days_since_created == 31
        )
        assert not {expired.id, deleted.id} & {row.frame_id for row in rows}
        assert (out_of_range.id in {row.frame_id for row in rows}) is not filtered
        params = ParamVector.from_config(cfg, pool_size=10)
        scalar = rescore_scores(rows, params)
        assert [doc for doc, _ in scalar] == [f"frame:{hit.frame.id}" for hit in result.hits]
        assert [score for _, score in scalar] == pytest.approx([hit.fusion for hit in result.hits])
        assert result.lane_fetch == (100 if filtered else 20)
        assert rescore_cache(cache, params)["q"] == [doc for doc, _ in scalar]


@pytest.mark.parametrize(
    "use_fts,use_vec,use_relevance",
    [(True, True, True), (False, True, True), (True, False, True), (True, True, False)],
)
def test_vectorized_and_scalar_rescore_have_identical_scores_and_ties(
    use_fts: bool, use_vec: bool, use_relevance: bool
) -> None:
    rows = [
        CandidateRow("a", 0, 1, 17.4, 5, 0, "critical", frame_id=8),
        CandidateRow("b", 0, 1, 17.4, 5, 0, "critical", frame_id=3),
        CandidateRow("c", None, 2, 0.4, 0, 4, "temporary", frame_id=2),
        CandidateRow("d", 2, None, 20.0, 9, 2, "normal", frame_id=4),
    ]
    cache = CandidateCache({"q": rows, "short": rows[:1], "empty": []})
    p = ParamVector(
        half_life_days=17.4,
        flat_boost_days=0.4,
        use_fts=use_fts,
        use_vec=use_vec,
        use_relevance=use_relevance,
    )
    scores = rescore_array_scores(cache, p)
    rankings = rescore_cache(cache, p)
    for i, qid in enumerate(cache.query_ids()):
        scalar = rescore_scores(cache.rows(qid), p)
        assert rankings[qid] == [doc for doc, _ in scalar]
        array_scores = {
            row.doc_id: scores[i, j]
            for j, row in enumerate(cache.rows(qid))
            if np.isfinite(scores[i, j])
        }
        assert array_scores == pytest.approx(dict(scalar))


def test_manifest_digest_excludes_wall_clock_and_fractional_config_is_exact() -> None:
    cache = CandidateCache({})
    assert replace(cache.manifest, created_at="2099-01-01").digest() == cache.manifest.digest()
    p = ParamVector(half_life_days=17.4, flat_boost_days=0.4)
    cfg = load_config().with_overrides(p.to_config_overrides())
    assert ParamVector.from_config(cfg).half_life_days == 17.4
    assert ParamVector.from_config(cfg).flat_boost_days == 0.4
