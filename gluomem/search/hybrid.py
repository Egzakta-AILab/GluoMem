"""Hybrid BM25 + vector search with RRF fusion, relevance scoring, reranking, and abstention."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from gluomem.config.app_config import (
    AppConfig,
    reranker_calibration_id,
)
from gluomem.embedding.base import Embedder, embed_query_text
from gluomem.embedding.compat import bind_embedder
from gluomem.embedding.reranker import Reranker
from gluomem.frames.eligibility import is_recall_eligible
from gluomem.frames.store import Frame, FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.sanitize import sanitize
from gluomem.search.scoring import relevance, rrf
from gluomem.storage.backend import FtsStore, VectorStore
from gluomem.storage.db import MindDB
from gluomem.storage.scope import MemoryScope, resolve_scope
from gluomem.timeutil import Clock, now_iso, parse_utc, utc_key, utc_now

__all__ = [
    "AbstentionDecision",
    "CandidatePool",
    "HybridSearch",
    "SearchCandidate",
    "SearchHit",
    "SearchResult",
]


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


@dataclass(frozen=True)
class SearchCandidate:
    """Eligible frame and raw features from the runtime retrieval lanes."""

    frame: Frame
    fts_rank: int | None
    vec_rank: int | None
    days_since_created: float
    access_count: int
    bfs_distance: int
    importance_level: str


@dataclass
class CandidatePool:
    """Raw candidates and lane diagnostics before fusion or reranking."""

    rows: list[SearchCandidate]
    lane_fetch: int
    kg_lane_errors: int


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
        scope: MemoryScope | None = None,
    ) -> None:
        self._db: MindDB = db
        self._clock: Clock = clock or utc_now
        self._store: FrameStore = store
        self._cfg: AppConfig = cfg
        self._scope: MemoryScope = resolve_scope(scope) if scope is not None else store.scope
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

    @property
    def scope(self) -> MemoryScope:
        return self._scope

    def search(
        self,
        query: str,
        *,
        profile: str | None = None,
        limit: int = 10,
        since: str | None = None,
        until: str | None = None,
        scope: MemoryScope | None = None,
        now_iso: str | None = None,
    ) -> SearchResult:
        """Run a hybrid BM25 + vector search and return a :class:`SearchResult`."""
        cfg = self._cfg
        s = resolve_scope(scope, default=self._scope)
        prof_name = profile or cfg.search.default_profile
        weights = cfg.scoring.profiles[prof_name].weights

        now = parse_utc(now_iso) if now_iso is not None else self._clock()
        pool = self.candidates(
            query,
            scope=s,
            now=now,
            limit=limit,
            since=since,
            until=until,
        )
        hits: list[SearchHit] = []
        for row in pool.rows:
            rrf_score = rrf(row.fts_rank, row.vec_rank, cfg.search.rrf_k)
            rel = relevance(
                days_since_created=row.days_since_created,
                access_count=row.access_count,
                bfs_distance=row.bfs_distance,
                importance_level=row.importance_level,
                scoring=cfg.scoring,
                weights=weights,
            )
            fused = rrf_score * rel
            hits.append(
                SearchHit(frame=row.frame, rrf=rrf_score, relevance=rel, final=fused, fusion=fused)
            )

        hits.sort(key=lambda h: (h.final, -h.frame.id), reverse=True)

        if cfg.search.rerank.enabled and self._reranker is not None and hits:
            hits = self._rerank_pool(query, hits)

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
            lane_fetch=pool.lane_fetch,
            profile=prof_name,
            kg_lane_errors=pool.kg_lane_errors,
            abstention=decision,
        )

    def candidates(
        self,
        query: str,
        *,
        scope: MemoryScope,
        now: datetime,
        lane_fetch: int | None = None,
        limit: int = 10,
        since: str | None = None,
        until: str | None = None,
    ) -> CandidatePool:
        """Retrieve raw features using runtime scope, eligibility and logical time."""
        cfg = self._cfg
        mult = (
            cfg.search.lane_fetch_time_mult
            if since is not None or until is not None
            else cfg.search.lane_fetch_mult
        )
        fetch = lane_fetch if lane_fetch is not None else max(1, limit) * int(mult)
        now_str = now_iso(now)
        fts_ids = self._fts_lane(query, fetch, scope=scope, now_iso=now_str)
        vec_ids = self._vec_lane(query, fetch, scope=scope, now_iso=now_str)
        fts_ranks = {fid: rank for rank, fid in enumerate(fts_ids)}
        vec_ranks = {fid: rank for rank, fid in enumerate(vec_ids)}
        ids = list(dict.fromkeys([*fts_ids, *vec_ids]))
        errors = 0
        ctx: dict[int, int] = {}
        if self._kg is not None:
            try:
                ctx = self._kg.contextual(query, cfg.kg, depth=cfg.scoring.bfs_depth, scope=scope)
            except Exception:
                errors += 1
        frames = self._store.get_frames(ids, scope=scope)
        rows = []
        for fid in ids:
            frame = frames.get(fid)
            if frame is None or not is_recall_eligible(frame, now=now):
                continue
            if not _event_time_in_range(frame.event_time, since, until):
                continue
            rows.append(
                SearchCandidate(
                    frame,
                    fts_ranks.get(fid),
                    vec_ranks.get(fid),
                    _days_since_created(frame.created_at, now),
                    frame.access_count,
                    ctx.get(fid, cfg.scoring.bfs_depth + 1),
                    frame.importance,
                )
            )
        return CandidatePool(rows, fetch, errors)

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

    def _fts_lane(
        self,
        query: str,
        lane_fetch: int,
        *,
        scope: MemoryScope,
        now_iso: str | None = None,
    ) -> list[int]:
        """BM25 lane: sanitize query and run FTS MATCH (with LIKE fallback)."""
        s = resolve_scope(scope, default=self._scope)
        sanitized = sanitize(query)
        if not sanitized:
            return self._fts.search_like(query, int(lane_fetch), scope=s, now_iso=now_iso)
        return self._fts.search(sanitized, int(lane_fetch), scope=s, now_iso=now_iso)

    def _vec_lane(
        self,
        query: str,
        lane_fetch: int,
        *,
        scope: MemoryScope,
        now_iso: str | None = None,
    ) -> list[int]:
        """Vector lane: chunk-level KNN primary, whole-frame KNN fallback."""
        s = resolve_scope(scope, default=self._scope)
        qvec = embed_query_text(self._embedder, query)
        chunk_hits = self._chunks_vec.search(qvec, int(lane_fetch), scope=s, now_iso=now_iso)
        if chunk_hits:
            chunk_ids = [int(cid) for cid, _dist in chunk_hits]
            chunk_to_frame = self._store.resolve_chunk_frames(chunk_ids, scope=s)
            ids: list[int] = []
            seen: set[int] = set()
            for cid in chunk_ids:
                fid = chunk_to_frame.get(cid)
                if fid is not None and fid not in seen:
                    seen.add(fid)
                    ids.append(fid)
            if ids:
                return self._eligible(ids, scope=s, now_iso=now_iso)
        frame_hits = self._frames_vec.search(qvec, int(lane_fetch), scope=s, now_iso=now_iso)
        return self._eligible([int(fid) for fid, _dist in frame_hits], scope=s, now_iso=now_iso)

    def _eligible(
        self,
        ids: list[int],
        *,
        scope: MemoryScope,
        now_iso: str | None = None,
    ) -> list[int]:
        """Drop recall-ineligible frame ids, preserving lane order."""
        eligible = self._store.filter_eligible_ids(ids, scope=scope, now_iso=now_iso)
        return [fid for fid in ids if fid in eligible]

    def _rerank_pool(
        self,
        query: str,
        hits: list[SearchHit],
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


def _event_time_in_range(event_time: str | None, since: str | None, until: str | None) -> bool:
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
