"""HybridSearch, KG contextual boost, and abstention contract tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pytest import approx

from gluomem.config.app_config import AppConfig, load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.embedding.reranker import Reranker
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.hybrid import HybridSearch, SearchResult
from gluomem.storage.db import MindDB

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_FIXED_NOW = datetime(2025, 1, 20, 10, 0, tzinfo=UTC)


def _cfg(**overrides: object) -> AppConfig:
    return load_config().with_overrides(overrides)


def _build(
    tmp_path: Path,
    *,
    cfg: AppConfig | None = None,
    reranker: Reranker | None = None,
    with_kg: bool = False,
    name: str = "test.mind",
) -> tuple[MindDB, FrameStore, HybridSearch, KnowledgeGraph | None]:
    cfg = cfg if cfg is not None else load_config()
    dim = int(cfg.storage.vector_dim)
    db = MindDB(tmp_path / name, dim=dim)
    embedder = MockEmbedder(dim=dim)
    store = FrameStore(db, embedder=embedder, chunking=cfg.chunking)
    kg = KnowledgeGraph(db) if with_kg else None
    search = HybridSearch(db, store, cfg, embedder=embedder, reranker=reranker, kg=kg)
    return db, store, search, kg


class _ContentReranker:
    _map = {"one": 1.0, "two": 2.0, "three": 3.0}

    def __init__(self) -> None:
        self.seen: list[str] = []

    def score(self, query: str, docs: list[str]) -> list[float]:
        self.seen.extend(docs)
        return [
            next((v for k, v in self._map.items() if k in d.lower()), 0.0)
            for d in docs
        ]


class _Logits:
    def __init__(self, **by_word: float) -> None:
        self._by_word = by_word

    def score(self, query: str, docs: list[str]) -> list[float]:
        return [
            next((v for w, v in self._by_word.items() if w in d), -9.0) for d in docs
        ]


def test_rrf_final_relation_and_lane_fetch(tmp_path: Path) -> None:
    cfg = _cfg(**{"search.rerank.enabled": False})
    db, store, search, _ = _build(tmp_path, cfg=cfg)
    try:
        store.create_i_frame("gop-1", "alpha beta", event_time="2024-06-01")
        res = search.search("alpha beta", limit=5)
        assert not res.abstained and res.hits
        assert res.lane_fetch == 5 * cfg.search.lane_fetch_mult == 10

        hit = res.hits[0]
        expected_rrf = 2.0 / cfg.search.rrf_k
        assert hit.rrf == approx(expected_rrf)
        assert hit.final == approx(hit.rrf * hit.relevance)

        w = cfg.scoring.profiles[cfg.search.default_profile].weights
        expected_rel = w.temporal + w.popularity + w.importance
        assert hit.relevance == approx(expected_rel)

        res_time = search.search(
            "alpha beta", limit=5, since="2024-01-01", until="2024-12-31"
        )
        assert res_time.lane_fetch == 5 * cfg.search.lane_fetch_time_mult == 50
    finally:
        db.close()


@pytest.mark.parametrize(
    ("threshold", "expect_abstained"),
    [(999.0, True), (0.0, False)],
)
def test_abstain_threshold_gate(
    tmp_path: Path, threshold: float, expect_abstained: bool
) -> None:
    cfg = _cfg(
        **{
            "search.rerank.enabled": False,
            "search.abstain.enabled": True,
            "search.abstain.threshold": threshold,
        }
    )
    db, store, search, _ = _build(tmp_path, cfg=cfg)
    try:
        store.create_i_frame("gop-1", "alpha beta")
        res = search.search("alpha beta", limit=10)
        assert res.abstained is expect_abstained
        assert bool(res.hits) is (not expect_abstained)
    finally:
        db.close()


def test_event_time_filter_and_utc_normalization(tmp_path: Path) -> None:
    cfg = _cfg(**{"search.rerank.enabled": False, "search.abstain.enabled": False})
    db, store, search, _ = _build(tmp_path, cfg=cfg)
    try:
        f2020 = store.create_i_frame(
            "gop-1", "alpha event in 2020", event_time="2020-06-15T12:00:00"
        )
        f2024 = store.create_i_frame(
            "gop-1", "alpha event in 2024", event_time="2024-06-15T12:00:00"
        )
        res = search.search("alpha", limit=10, since="2024-01-01", until="2024-12-31")
        assert [h.frame.id for h in res.hits] == [f2024.id]
        assert f2020.id not in [h.frame.id for h in res.hits]

        before = store.create_i_frame(
            "gop-2", "beta before", event_time="2025-06-01T01:00:00+02:00"
        )
        inside = store.create_i_frame(
            "gop-3", "beta inside", event_time="2025-06-01T00:30:00Z"
        )
        assert before.event_time == "2025-05-31T23:00:00"
        assert {h.frame.id for h in search.search("beta", since="2025-06-01").hits} == {
            inside.id
        }
        assert {
            h.frame.id
            for h in search.search(
                "beta", since="2025-05-01", until="2025-06-01T02:30:00+02:00"
            ).hits
        } == {before.id, inside.id}
    finally:
        db.close()


def test_created_at_utc_decay_and_legacy_offset_rows(tmp_path: Path) -> None:
    cfg = _cfg(**{"search.rerank.enabled": False, "search.abstain.enabled": False})
    db, store, _, _ = _build(tmp_path, cfg=cfg)
    try:
        aware = store.create_i_frame(
            "gop-1", "alpha aware", created_at="2025-01-10T12:00:00+02:00"
        )
        naive = store.create_i_frame(
            "gop-2", "alpha naive", created_at="2025-01-10T10:00:00"
        )
        legacy = store.create_i_frame("gop-3", "alpha legacy")
        db.conn.execute(
            "UPDATE memory_frames SET created_at=? WHERE id=?",
            ("2025-01-10T12:00:00+02:00", legacy.id),
        )
        db.conn.commit()
        fixed_search = HybridSearch(
            db, store, cfg, embedder=store.embedder, clock=lambda: _FIXED_NOW
        )
        hits = {h.frame.id: h for h in fixed_search.search("alpha").hits}
        assert hits[aware.id].relevance == approx(hits[naive.id].relevance)
        assert hits[legacy.id].relevance == approx(hits[naive.id].relevance)
    finally:
        db.close()


def test_rerank_replaces_score_and_reorders(tmp_path: Path) -> None:
    cfg = _cfg(**{"search.rerank.enabled": True, "search.abstain.enabled": False})
    reranker = _ContentReranker()
    db, store, search, _ = _build(tmp_path, cfg=cfg, reranker=reranker)
    try:
        store.create_i_frame("gop-1", "alpha one")
        store.create_i_frame("gop-1", "alpha two")
        store.create_i_frame("gop-1", "alpha three")

        res = search.search("alpha", limit=10)
        assert [h.frame.content for h in res.hits] == [
            "alpha three",
            "alpha two",
            "alpha one",
        ]
        assert [h.final for h in res.hits] == [approx(3.0), approx(2.0), approx(1.0)]
        assert all(h.rrf > 0.0 and h.relevance > 0.0 for h in res.hits)
    finally:
        db.close()


def test_like_fallback_escapes_metacharacters(tmp_path: Path) -> None:
    db, store, search, _ = _build(tmp_path)
    try:
        with_pct = store.create_i_frame("gop-1", "confidence was 100% today")
        store.create_i_frame("gop-1", "confidence was one hundred today")
        assert search._fts_lane("%", lane_fetch=10) == [with_pct.id]

        with_us = store.create_i_frame("gop-1", "var snake_case here")
        store.create_i_frame("gop-1", "var snakeXcase here")
        assert search._fts_lane("_", lane_fetch=10) == [with_us.id]
    finally:
        db.close()


def test_quarantine_exclusion_across_lanes_rerank_and_kg(tmp_path: Path) -> None:
    cfg = _cfg(**{"search.rerank.enabled": True, "search.abstain.enabled": False})
    reranker = _ContentReranker()
    db, store, search, kg = _build(
        tmp_path, cfg=cfg, reranker=reranker, with_kg=True
    )
    assert kg is not None
    try:
        ok1 = store.create_i_frame("gop-1", "alpha one", trust="trusted")
        ok2 = store.create_i_frame("gop-1", "alpha two", trust="low")
        q = store.create_i_frame(
            "gop-1", "alpha three 100%", trust="quarantined"
        )
        entity = kg.create_entity("concept", "alpha")
        kg.link_entity_frame(entity.id, ok1.id)
        kg.link_entity_frame(entity.id, q.id)

        res = search.search("alpha", limit=10)
        assert [h.frame.id for h in res.hits] == [ok2.id, ok1.id]
        assert q.id not in {h.frame.id for h in res.hits}
        assert "alpha three 100%" not in reranker.seen
        assert search._fts_lane("%", lane_fetch=10) == []
        assert q.id not in search._vec_lane("alpha", lane_fetch=10)
        assert store.get_frame(q.id).trust == "quarantined"
    finally:
        db.close()


def test_contextual_boost_linked_outranks_unlinked_and_zero_weight_equalizes(
    tmp_path: Path,
) -> None:
    cfg = _cfg(**{"search.rerank.enabled": False, "search.abstain.enabled": False})
    db, store, search, kg = _build(tmp_path, cfg=cfg, with_kg=True)
    assert kg is not None
    try:
        linked = store.create_i_frame("gop-linked", "alpha beta")
        unlinked = store.create_i_frame("gop-unlinked", "alpha beta")
        entity = kg.create_entity("concept", "alpha")
        kg.link_entity_frame(entity.id, linked.id)

        ctx = kg.contextual("alpha", cfg.kg, depth=cfg.scoring.bfs_depth)
        assert ctx[linked.id] == 0 and unlinked.id not in ctx

        res = search.search("alpha", limit=10)
        assert res.hits[0].frame.id == linked.id
        by_id = {h.frame.id: h for h in res.hits}
        assert by_id[linked.id].relevance > by_id[unlinked.id].relevance

        zero_cfg = _cfg(
            **{
                "search.rerank.enabled": False,
                "search.abstain.enabled": False,
                "scoring.profiles.balanced.weights.temporal": 0.5,
                "scoring.profiles.balanced.weights.popularity": 0.25,
                "scoring.profiles.balanced.weights.contextual": 0.0,
                "scoring.profiles.balanced.weights.importance": 0.25,
            }
        )
        zero_search = HybridSearch(
            db, store, zero_cfg, embedder=store.embedder, kg=kg
        )
        res_zero = zero_search.search("alpha", limit=10)
        by_id_zero = {h.frame.id: h for h in res_zero.hits}
        assert by_id_zero[linked.id].relevance == approx(
            by_id_zero[unlinked.id].relevance
        )
    finally:
        db.close()


def _both(
    tmp_path: Path, query: str, **overrides: object
) -> tuple[SearchResult, SearchResult]:
    off = _cfg(**{**overrides, "search.rerank.enabled": False})
    on = _cfg(**{**overrides, "search.rerank.enabled": True})
    db1, store1, s_off, _ = _build(tmp_path, cfg=off, name="off.mind")
    db2, store2, s_on, _ = _build(
        tmp_path, cfg=on, reranker=_Logits(alpha=-2.0, beta=-6.0), name="on.mind"
    )
    try:
        for s in (store1, store2):
            s.create_i_frame("g", "alpha deploy window is friday")
            s.create_i_frame("g", "beta rollback plan uses blue green")
        return s_off.search(query), s_on.search(query)
    finally:
        db1.close()
        db2.close()


def test_abstention_contract_across_evidence_levels(tmp_path: Path) -> None:
    off, on = _both(tmp_path, "alpha deploy window")
    for res in (off, on):
        assert not res.abstained and res.hits
        d = res.abstention
        assert d is not None
        assert (d.reason, d.score, d.calibration) == (
            "sufficient",
            "fusion",
            "uncalibrated",
        )
        assert d.score_identity == "fusion/rrf_k=60/profile=balanced"
    assert on.abstention.top_score == approx(off.abstention.top_score)  # type: ignore[union-attr]
    assert on.hits[0].final == on.hits[0].rerank == approx(-2.0)
    assert on.hits[0].fusion > 0.0

    w_off, w_on = _both(
        tmp_path / "weak", "alpha", **{"search.abstain.threshold": 10.0}
    )
    for res in (w_off, w_on):
        assert res.abstained and res.hits == []
        assert res.abstention is not None and res.abstention.reason == "below_threshold"

    d_off, d_on = _both(
        tmp_path / "dis", "alpha", **{"search.abstain.enabled": False}
    )
    for res in (d_off, d_on):
        assert res.abstention is not None and res.abstention.reason == "disabled"
        assert res.abstention.top_score is not None and res.hits


def test_no_eligible_evidence_records_no_hits(tmp_path: Path) -> None:
    cfg = _cfg(**{"search.rerank.enabled": False})
    db, store, search, _ = _build(tmp_path, cfg=cfg, name="q.mind")
    try:
        store.create_i_frame("g", "alpha secret", trust="quarantined")
        res = search.search("alpha")
        assert res.abstained and res.abstention is not None
        assert (res.abstention.reason, res.abstention.candidates) == ("no_hits", 0)
        assert res.abstention.top_score is None
    finally:
        db.close()


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
)
def test_incompatible_calibration_is_rejected_at_load(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _cfg(**overrides)


def test_rerank_gate_with_matching_calibration_and_revision(tmp_path: Path) -> None:
    cfg = _cfg(
        **{
            "search.rerank.enabled": True,
            "models.reranker.revision": "abc123",
            "search.abstain.score": "rerank",
            "search.abstain.calibration": f"{RERANKER_MODEL}@abc123",
            "search.abstain.threshold": -3.0,
        }
    )
    with pytest.raises(ValueError, match="needs a reranker"):
        _build(tmp_path, cfg=cfg, reranker=None, name="none.mind")

    db, store, search, _ = _build(
        tmp_path, cfg=cfg, reranker=_Logits(alpha=-2.0, beta=-6.0), name="ok.mind"
    )
    try:
        store.create_i_frame("g", "alpha deploy window")
        res = search.search("alpha deploy window")
        d = res.abstention
        assert d is not None and not res.abstained
        assert d.score_identity == f"rerank/{RERANKER_MODEL}@abc123"
        assert d.top_score == approx(-2.0)
    finally:
        db.close()
