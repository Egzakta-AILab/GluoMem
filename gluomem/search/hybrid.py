"""Hybrid BM25 + vector search with RRF fusion, relevance scoring, reranking, and abstention."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime

from gluomem.config.app_config import (
    AppConfig,
    ProfileWeights,
    reranker_calibration_id,
)
from gluomem.embedding.base import Embedder
from gluomem.embedding.compat import bind_embedder
from gluomem.embedding.reranker import Reranker
from gluomem.frames.eligibility import is_recall_eligible
from gluomem.frames.store import Frame, FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.sanitize import sanitize
from gluomem.search.scoring import relevance
from gluomem.storage.backend import FtsStore, VectorStore
from gluomem.storage.db import MindDB
from gluomem.timeutil import Clock, parse_utc, utc_key, utc_now

__all__ = ["AbstentionDecision", "HybridSearch", "SearchHit", "SearchResult"]


@dataclass
class SearchHit:
    """A single ranked search hit."""

    frame: Frame
    rrf: float
    relevance: float
    final: float
    rank: int = 0
    fusion: float = 0.0
    rerank: float | None = None


@dataclass(frozen=True)
class AbstentionDecision:
    """Why a search did or did not abstain, on which scale."""

    enabled: bool
    abstained: bool
    reason: str
    score: str
    score_identity: str
    threshold: float
    calibration: str
    top_score: float | None
    candidates: int

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass
class SearchResult:
    """The outcome of a :meth:`HybridSearch.search` call."""

    hits: list[SearchHit] = field(default_factory=list)
    abstained: bool = False
    lane_fetch: int = 0
    profile: str = "balanced"
    kg_lane_errors: int = 0
    abstention: AbstentionDecision | None = None


class HybridSearch:
    """Config-driven hybrid BM25 + vector search."""

    def __init__(
        self,
        db: MindDB,
        store: FrameStore,
        cfg: AppConfig,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        kg: KnowledgeGraph | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._db: MindDB = db
        self._clock: Clock = clock or utc_now
        self._conn: sqlite3.Connection = db.conn
        self._store: FrameStore = store
        self._cfg: AppConfig = cfg
        if embedder is None:
            embedder = store.embedder
        elif embedder is not store.embedder:
            bind_embedder(db, embedder)
        self._embedder: Embedder = embedder
        abstain = cfg.search.abstain
        if abstain.enabled and abstain.score == "rerank" and reranker is None:
            raise ValueError(
                "search.abstain.score='rerank' needs a reranker; none was given, "
                "so the gate would have no logit to read"
            )
        self._reranker: Reranker | None = reranker
        self._kg: KnowledgeGraph | None = kg
        self._fts: FtsStore = db.fts
        self._frames_vec: VectorStore = db.frame_vec
        self._chunks_vec: VectorStore = db.chunk_vec

    def search(
        self,
        query: str,
        *,
        profile: str | None = None,
        limit: int = 10,
        since: str | None = None,
        until: str | None = None,
    ) -> SearchResult:
        """Run a hybrid BM25 + vector search and return a :class:`SearchResult`."""
        cfg = self._cfg
        prof_name = profile or cfg.search.default_profile
        weights = cfg.scoring.profiles[prof_name].weights

        time_filter = since is not None or until is not None
        mult = (
            cfg.search.lane_fetch_time_mult
            if time_filter
            else cfg.search.lane_fetch_mult
        )
        lane_fetch = max(1, limit) * int(mult)

        fts_ids = self._fts_lane(query, lane_fetch)
        vec_ids = self._vec_lane(query, lane_fetch)

        rrf_k = int(cfg.search.rrf_k)
        rrf: dict[int, float] = {}
        for rank, fid in enumerate(fts_ids):
            rrf[fid] = rrf.get(fid, 0.0) + 1.0 / (rrf_k + rank)
        for rank, fid in enumerate(vec_ids):
            rrf[fid] = rrf.get(fid, 0.0) + 1.0 / (rrf_k + rank)

        kg_lane_errors = 0
        ctx_map: dict[int, int] = {}
        if self._kg is not None:
            try:
                ctx_map = self._kg.contextual(
                    query, cfg.kg, depth=cfg.scoring.bfs_depth
                )
            except Exception:
                kg_lane_errors += 1
                ctx_map = {}

        beyond_depth = cfg.scoring.bfs_depth + 1
        now = self._clock()

        frames_by_id = self._store.get_frames(list(rrf.keys()))
        hits: list[SearchHit] = []
        for fid, rrf_score in rrf.items():
            frame = frames_by_id.get(fid)
            if frame is None:
                continue
            if not is_recall_eligible(frame.trust):
                continue
            if not _event_time_in_range(frame.event_time, since, until):
                continue
            bfs_distance = ctx_map.get(fid, beyond_depth)
            rel = relevance(
                days_since_created=_days_since_created(frame.created_at, now),
                access_count=frame.access_count,
                bfs_distance=bfs_distance,
                importance_level=frame.importance,
                scoring=cfg.scoring,
                weights=weights,
            )
            fused = rrf_score * rel
            hits.append(
                SearchHit(
                    frame=frame, rrf=rrf_score, relevance=rel, final=fused, fusion=fused
                )
            )

        hits.sort(key=lambda h: (h.final, -h.frame.id), reverse=True)

        if cfg.search.rerank.enabled and self._reranker is not None and hits:
            hits = self._rerank_pool(query, hits, weights)

        for i, h in enumerate(hits):
            h.rank = i

        decision = self._abstention(hits, prof_name)
        abstained = decision.abstained
        if abstained:
            hits = []
        else:
            hits = hits[: max(0, limit)]

        return SearchResult(
            hits=hits,
            abstained=abstained,
            lane_fetch=lane_fetch,
            profile=prof_name,
            kg_lane_errors=kg_lane_errors,
            abstention=decision,
        )

    def _abstention(self, hits: list[SearchHit], profile: str) -> AbstentionDecision:
        cfg = self._cfg
        abstain = cfg.search.abstain
        if abstain.score == "rerank":
            identity = f"rerank/{reranker_calibration_id(cfg)}"
            scores = [h.rerank for h in hits if h.rerank is not None]
        else:
            identity = f"fusion/rrf_k={cfg.search.rrf_k}/profile={profile}"
            scores = [h.fusion for h in hits]
        top = max(scores) if scores else None
        if top is None:
            reason = "no_hits"
        elif not abstain.enabled:
            reason = "disabled"
        elif top < abstain.threshold:
            reason = "below_threshold"
        else:
            reason = "sufficient"
        return AbstentionDecision(
            enabled=abstain.enabled,
            abstained=abstain.enabled and reason in ("no_hits", "below_threshold"),
            reason=reason,
            score=abstain.score,
            score_identity=identity,
            threshold=abstain.threshold,
            calibration=abstain.calibration or "uncalibrated",
            top_score=top,
            candidates=len(hits),
        )

    def _fts_lane(self, query: str, lane_fetch: int) -> list[int]:
        """BM25 lane: sanitize query and run FTS MATCH (with LIKE fallback)."""
        sanitized = sanitize(query)
        if not sanitized:
            return self._fts.search_like(query, int(lane_fetch))
        return self._fts.search(sanitized, int(lane_fetch))

    def _vec_lane(self, query: str, lane_fetch: int) -> list[int]:
        """Vector lane: chunk-level KNN primary, whole-frame KNN fallback."""
        qvec = self._embedder.embed(query)
        chunk_hits = self._chunks_vec.search(qvec, int(lane_fetch))
        if chunk_hits:
            chunk_ids = [int(cid) for cid, _dist in chunk_hits]
            chunk_to_frame = self._store.resolve_chunk_frames(chunk_ids)
            ids: list[int] = []
            seen: set[int] = set()
            for cid in chunk_ids:
                fid = chunk_to_frame.get(cid)
                if fid is not None and fid not in seen:
                    seen.add(fid)
                    ids.append(fid)
            if ids:
                return self._eligible(ids)
        frame_hits = self._frames_vec.search(qvec, int(lane_fetch))
        return self._eligible([int(fid) for fid, _dist in frame_hits])

    def _eligible(self, ids: list[int]) -> list[int]:
        """Drop recall-ineligible frame ids, preserving lane order."""
        eligible = self._store.filter_eligible_ids(ids)
        return [fid for fid in ids if fid in eligible]

    def _rerank_pool(
        self,
        query: str,
        hits: list[SearchHit],
        _weights: ProfileWeights,
    ) -> list[SearchHit]:
        """Rerank the top ``rerank.pool`` hits with the cross-encoder."""
        pool_size = int(self._cfg.search.rerank.pool)
        if pool_size <= 0 or not hits:
            return hits
        pooled = hits[:pool_size]
        rest = hits[pool_size:]

        docs = [h.frame.indexable_text for h in pooled]
        logits = self._reranker.score(query, docs) if self._reranker else []
        for h, logit in zip(pooled, logits, strict=True):
            h.rerank = float(logit)
            h.final = h.rerank

        pooled.sort(key=lambda h: (h.final, -h.frame.id), reverse=True)
        return pooled + rest


def _days_since_created(created_at: str, now: datetime) -> float:
    try:
        dt = parse_utc(created_at)
    except ValueError:
        return 0.0
    delta = (parse_utc(now) - dt).total_seconds() / 86400.0
    return max(0.0, delta)


def _event_time_in_range(
    event_time: str | None, since: str | None, until: str | None
) -> bool:
    if since is None and until is None:
        return True
    if event_time is None:
        return False
    et = utc_key(event_time)
    if since is not None and et < utc_key(since):
        return False
    if until is not None and et > utc_key(until):
        return False
    return True
