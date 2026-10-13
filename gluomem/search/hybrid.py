"""gluomem.search.hybrid — HybridSearch.

Ties the BM25 (FTS5) and vector (chunk-level KNN with whole-frame fallback)
lanes together with:

- **RRF fusion** — ``rrf[id] += 1/(rrf_k + rank)`` per lane (``rrf_k`` from
  config, default 60).
- **event_time range filter** — frames are filtered by ``event_time`` (DISTINCT
  from ``created_at``) when ``since``/``until`` are supplied. The decay anchor
  stays ``created_at``; only the range filter anchors on ``event_time``.
  All timestamps compare as UTC instants (:mod:`gluomem.timeutil`), and
  recency decay reads "now" from an injectable clock.
- **relevance multiplier** — ``final = rrf * relevance`` where ``relevance`` is
  the active profile's weighted sum of temporal/popularity/contextual/importance
  (:mod:`gluomem.search.scoring`). Every constant comes from
  ``AppConfig``; each signal is independently zeroable via a profile weight of 0.
- **lane fetch** — ``lane_fetch = limit*lane_fetch_mult`` normally, but
  ``limit*lane_fetch_time_mult`` when a since/until time filter is set. The
  effective ``lane_fetch`` is EXPOSED on the result so it is observable.
- **rerank pool** — when ``search.rerank.enabled`` and a reranker is configured,
  the top ``rerank.pool`` hits are re-scored by the cross-encoder; each pooled
  hit's ``rerank`` holds the logit and its ranking score ``final`` becomes the
  logit; the pool is then re-sorted by logit descending. ``fusion`` keeps
  ``rrf * relevance`` for every hit.
- **abstain gate** — ranking and sufficiency are separate. The gate reads the
  score ``search.abstain.score`` names (``fusion``: top ``fusion`` among
  eligible hits, identical with rerank on or off; ``rerank``: top logit, only
  with a matching declared calibration, validated in config) and compares it
  with ``abstain.threshold``. Every result carries an
  :class:`AbstentionDecision` (reason, score identity, top score, threshold,
  calibration), including when the gate is disabled or nothing matched.

- **eligibility** — recall-ineligible frames (quarantined; see
  :mod:`gluomem.frames.eligibility`) are dropped inside each lane and again at
  frame load, so they never take a lane rank, receive a KG contextual score,
  enter the rerank pool or reach the caller. Inspect them by id through
  ``FrameStore.get_frame``.

When no ``KnowledgeGraph`` is wired, the contextual distance defaults to the
no-link value (distance beyond ``bfs_depth`` -> ``contextual == 0.0``).

Every constant flows from the passed ``AppConfig`` — no literals beyond the
documented RRF / scoring formula shape. Deterministic and offline: tests use
the ``MockEmbedder`` and a stub reranker (no network, no model downloads).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime

from gluomem.config.app_config import AppConfig, ProfileWeights, reranker_calibration_id
from gluomem.embedding.base import Embedder
from gluomem.embedding.compat import bind_embedder
from gluomem.embedding.reranker import Reranker
from gluomem.frames.eligibility import (
    INELIGIBLE_PARAMS,
    INELIGIBLE_PLACEHOLDERS,
    is_recall_eligible,
)
from gluomem.frames.store import Frame, FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.sanitize import sanitize
from gluomem.search.scoring import relevance
from gluomem.storage.db import MindDB
from gluomem.storage.sqlite_vec import SqliteVecStore
from gluomem.timeutil import Clock, parse_utc, utc_key, utc_now

__all__ = ["AbstentionDecision", "HybridSearch", "SearchHit", "SearchResult"]


# ---------------------------------------------------------------------------
# Result types.
# ---------------------------------------------------------------------------


@dataclass
class SearchHit:
    """A single ranked search hit.

    ``rrf`` and ``relevance`` are the pre-rerank fusion / personalization
    scores and ``fusion == rrf * relevance``. ``rerank`` is the cross-encoder
    logit for pooled hits (``None`` otherwise). ``final`` is the ranking
    score: the logit for reranked hits, else ``fusion``. Abstention reads
    ``fusion`` or ``rerank`` explicitly, never ``final``.
    """

    frame: Frame
    rrf: float
    relevance: float
    final: float
    rank: int = 0
    fusion: float = 0.0
    rerank: float | None = None


@dataclass(frozen=True)
class AbstentionDecision:
    """Why a search did or did not abstain, on which scale.

    ``reason`` is ``no_hits`` (no eligible evidence), ``below_threshold``,
    ``sufficient`` or ``disabled`` (gate off, evidence present).
    ``score_identity`` names the scale (``fusion/rrf_k=60/profile=balanced``
    or ``rerank/<model>``); ``calibration`` is the declared threshold source,
    ``"uncalibrated"`` when none was declared.
    """

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
    """The outcome of a :meth:`HybridSearch.search` call.

    ``abstained`` is True when the abstain gate fired (no eligible evidence or
    insufficient top score); in that case ``hits`` is empty. ``abstention``
    records the decision and its score contract for every search.
    ``lane_fetch`` is the effective per-lane fetch used (exposed for
    observability).
    ``kg_lane_errors`` counts KG contextual-lane failures observed during the
    search: a normal run leaves it at 0; a KG-lane exception increments it
    (distinguishable from a legitimate zero-contextual result) and the lane
    degrades to no-contextual. NEVER a silent zero.
    """

    hits: list[SearchHit] = field(default_factory=list)
    abstained: bool = False
    lane_fetch: int = 0
    profile: str = "balanced"
    kg_lane_errors: int = 0
    abstention: AbstentionDecision | None = None


# ---------------------------------------------------------------------------
# HybridSearch.
# ---------------------------------------------------------------------------


class HybridSearch:
    """Config-driven hybrid BM25 + vector search with RRF, relevance, rerank,
    and a wired abstain gate.

    Parameters
    ----------
    db:
        An open ``MindDB`` (WAL, FK, sqlite-vec loaded, schema applied). The
        search reuses its connection for FTS / vec / frame reads.
    store:
        The ``FrameStore`` over ``db``; used to load typed frames by id.
    cfg:
        The ``AppConfig`` carrying every search/scoring constant (rrf_k,
        lane_fetch multipliers, rerank pool, abstain threshold, scoring
        profile weights, …). Nothing is hardcoded.
    embedder:
        The embedder used to embed the query for the vector lane. Defaults to
        the store's embedder. A different embedder must carry the fingerprint
        recorded in the file, or construction raises
        :class:`~gluomem.embedding.compat.EmbeddingMismatchError`.
    reranker:
        Optional cross-encoder reranker. Used only when
        ``cfg.search.rerank.enabled`` is True. When ``None`` the rerank step is
        skipped even if enabled (no reranker to score with).
    clock:
        Returns "now" for recency decay. Defaults to the UTC wall clock;
        evaluation injects a fixed clock so scores are reproducible.
    kg:
        Optional ``KnowledgeGraph`` providing the contextual KG signal. When
        wired, the search computes ``kg.contextual(query, ...)`` and feeds
        each candidate frame's BFS distance into ``scoring.contextual``. The
        lane is wrapped: a KG exception is RECORDED on
        ``SearchResult.kg_lane_errors`` (never a silent zero) and the lane
        degrades to no-contextual. When ``None`` the contextual signal is
        simply absent (distance beyond ``bfs_depth`` -> 0.0).
    """

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
        self._frames_vec: SqliteVecStore = SqliteVecStore(
            db.conn, table="memory_frames_vec", dim=db.dim
        )
        self._chunks_vec: SqliteVecStore = SqliteVecStore(
            db.conn, table="memory_frame_chunks_vec", dim=db.dim
        )

    # --- public API --------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        profile: str | None = None,
        limit: int = 10,
        since: str | None = None,
        until: str | None = None,
    ) -> SearchResult:
        """Run a hybrid BM25 + vector search and return a :class:`SearchResult`.

        Parameters
        ----------
        query:
            The user query. Sanitized for the FTS5 MATCH path (token-preserving,
            LIKE fallback when the sanitizer signals it).
        profile:
            The scoring profile name (e.g. ``"balanced"``). Defaults to
            ``cfg.search.default_profile``.
        limit:
            Maximum number of hits to return.
        since, until:
            Optional ``event_time`` range bounds (ISO ``YYYY-MM-DD`` or
            ``YYYY-MM-DDTHH:MM:SS``). When either is set, the per-lane fetch
            uses ``lane_fetch_time_mult`` (10×) instead of ``lane_fetch_mult``
            (2×), and frames whose ``event_time`` falls outside the range are
            filtered out. The decay anchor stays ``created_at``; only the range
            filter anchors on ``event_time``.
        """
        cfg = self._cfg
        prof_name = profile or cfg.search.default_profile
        weights = cfg.scoring.profiles[prof_name].weights

        # (1) Effective per-lane fetch: limit*mult, or limit*time_mult when a
        # time filter is set. Exposed on the result so it is observable.
        time_filter = since is not None or until is not None
        mult = (
            cfg.search.lane_fetch_time_mult
            if time_filter
            else cfg.search.lane_fetch_mult
        )
        lane_fetch = max(1, limit) * int(mult)

        # (2) FTS / BM25 lane: sanitize then FTS5 MATCH (LIKE fallback).
        fts_ids = self._fts_lane(query, lane_fetch)
        # (3) Vector lane: embed the query, chunk-level KNN primary, whole-frame
        # KNN fallback.
        vec_ids = self._vec_lane(query, lane_fetch)

        # (4) RRF fusion: rrf[id] += 1/(rrf_k + rank) per lane.
        rrf_k = int(cfg.search.rrf_k)
        rrf: dict[int, float] = {}
        for rank, fid in enumerate(fts_ids):
            rrf[fid] = rrf.get(fid, 0.0) + 1.0 / (rrf_k + rank)
        for rank, fid in enumerate(vec_ids):
            rrf[fid] = rrf.get(fid, 0.0) + 1.0 / (rrf_k + rank)

        # (5) KG contextual signal: map each candidate frame to its minimum BFS
        # distance from the query. The lane is WRAPPED — a KG exception is
        # RECORDED on the observable error counter (never a silent zero) and
        # the lane degrades to no-contextual (distance beyond bfs_depth).
        # A clean run (KG present, contextual succeeds) leaves the counter 0;
        # when no KG is wired the counter stays 0 and the signal is absent.
        kg_lane_errors = 0
        ctx_map: dict[int, int] = {}
        if self._kg is not None:
            try:
                ctx_map = self._kg.contextual(
                    query, cfg.kg, depth=cfg.scoring.bfs_depth
                )
            except Exception:
                # Record the KG-lane failure on the observable counter and
                # degrade to no-contextual — NEVER a silent zero
                # indistinguishable from a legitimate zero-contextual result.
                kg_lane_errors += 1
                ctx_map = {}
        # Frames absent from the contextual map are "beyond depth" -> contextual 0.0.
        beyond_depth = cfg.scoring.bfs_depth + 1
        now = self._clock()

        # (6) Load frames and apply the event_time since/until range filter.
        hits: list[SearchHit] = []
        for fid, rrf_score in rrf.items():
            try:
                frame = self._store.get_frame(fid)
            except KeyError:
                # Stale id (frame deleted since indexing). Skip it.
                continue
            if not is_recall_eligible(frame.trust):
                continue
            if not _event_time_in_range(frame.event_time, since, until):
                continue
            # (7) relevance with the active profile's weights. The contextual
            # distance comes from the KG map; unlinked/unreached frames fall
            # back to "beyond depth" -> contextual 0.0.
            bfs_distance = ctx_map.get(fid, beyond_depth)
            rel = relevance(
                days_since_created=_days_since_created(frame.created_at, now),
                access_count=frame.access_count,
                bfs_distance=bfs_distance,
                importance_level=frame.importance,
                scoring=cfg.scoring,
                weights=weights,
            )
            # (8) final = fusion = rrf * relevance (until rerank).
            fused = rrf_score * rel
            hits.append(
                SearchHit(
                    frame=frame, rrf=rrf_score, relevance=rel, final=fused, fusion=fused
                )
            )

        # Sort by final descending (stable on equal finals by frame id).
        hits.sort(key=lambda h: (h.final, -h.frame.id), reverse=True)

        # (9) Rerank pool: replace the top pool's scores with cross-encoder
        # logits and re-sort the pool by logit.
        if cfg.search.rerank.enabled and self._reranker is not None and hits:
            hits = self._rerank_pool(query, hits, weights)

        # Assign ranks (post-sort).
        for i, h in enumerate(hits):
            h.rank = i

        # (10) Sufficiency gate on its declared scale, never on ``final``.
        decision = self._abstention(hits, prof_name)
        abstained = decision.abstained
        if abstained:
            hits = []

        # Truncate to the requested limit (after abstain so an abstained result
        # stays empty).
        if not abstained:
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

    # --- lanes --------------------------------------------------------------

    def _fts_lane(self, query: str, lane_fetch: int) -> list[int]:
        """BM25 lane: sanitize the query, FTS5 MATCH (LIKE fallback)."""
        sanitized = sanitize(query)
        if not sanitized:
            # LIKE fallback for empty / too-short / no-usable-token queries.
            # LIKE metacharacters in the raw query are escaped so e.g. '%'
            # matches frames containing a literal '%', not every frame.
            escaped = (
                query.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
            )
            cur = self._conn.execute(
                "SELECT id FROM memory_frames WHERE content LIKE ? ESCAPE '\\' "
                f"AND trust NOT IN ({INELIGIBLE_PLACEHOLDERS}) "
                "ORDER BY id ASC LIMIT ?",
                (f"%{escaped}%", *INELIGIBLE_PARAMS, int(lane_fetch)),
            )
            return [int(r[0]) for r in cur.fetchall()]
        cur = self._conn.execute(
            "SELECT f.rowid FROM memory_frames_fts f "
            "JOIN memory_frames m ON m.id = f.rowid "
            f"WHERE memory_frames_fts MATCH ? AND m.trust NOT IN ({INELIGIBLE_PLACEHOLDERS}) "
            "ORDER BY f.rank LIMIT ?",
            (sanitized, *INELIGIBLE_PARAMS, int(lane_fetch)),
        )
        return [int(r[0]) for r in cur.fetchall()]

    def _vec_lane(self, query: str, lane_fetch: int) -> list[int]:
        """Vector lane: chunk-level KNN primary, whole-frame KNN fallback.

        Returns the deduplicated frame-id ranking (a frame appears once, at its
        best chunk distance).
        """
        qvec = self._embedder.embed(query)
        # Primary: chunk-level KNN. Map chunk rowids back to frame ids, keeping
        # the first (nearest) occurrence of each frame.
        chunk_hits = self._chunks_vec.search(qvec, int(lane_fetch))
        if chunk_hits:
            ids: list[int] = []
            seen: set[int] = set()
            for chunk_rowid, _dist in chunk_hits:
                row = self._conn.execute(
                    "SELECT frame_id FROM memory_frame_chunks WHERE id=?",
                    (int(chunk_rowid),),
                ).fetchone()
                if row is None:
                    continue
                fid = int(row[0])
                if fid not in seen:
                    seen.add(fid)
                    ids.append(fid)
            if ids:
                return self._eligible(ids)
        # Fallback: whole-frame KNN.
        frame_hits = self._frames_vec.search(qvec, int(lane_fetch))
        return self._eligible([int(fid) for fid, _dist in frame_hits])

    def _eligible(self, ids: list[int]) -> list[int]:
        """Drop recall-ineligible frame ids, preserving lane order."""
        if not ids:
            return ids
        marks = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT id FROM memory_frames WHERE id IN ({marks}) "
            f"AND trust NOT IN ({INELIGIBLE_PLACEHOLDERS})",
            (*ids, *INELIGIBLE_PARAMS),
        ).fetchall()
        keep = {int(r[0]) for r in rows}
        return [i for i in ids if i in keep]

    # --- rerank -------------------------------------------------------------

    def _rerank_pool(
        self,
        query: str,
        hits: list[SearchHit],
        _weights: ProfileWeights,
    ) -> list[SearchHit]:
        """Rerank the top ``rerank.pool`` hits with the cross-encoder.

        Each pooled hit's ``rerank`` and ranking score ``final`` become the
        cross-encoder logit (``fusion`` is kept); the pool is then re-sorted by
        logit descending. Hits beyond the pool keep their fusion score and order.
        """
        pool_size = int(self._cfg.search.rerank.pool)
        if pool_size <= 0 or not hits:
            return hits
        pooled = hits[:pool_size]
        rest = hits[pool_size:]

        docs = [_indexable_text(h.frame) for h in pooled]
        logits = self._reranker.score(query, docs) if self._reranker else []
        for h, logit in zip(pooled, logits, strict=True):
            h.rerank = float(logit)
            h.final = h.rerank

        # Re-sort the pool by logit descending (stable on equal logits).
        pooled.sort(key=lambda h: (h.final, -h.frame.id), reverse=True)
        return pooled + rest


# ---------------------------------------------------------------------------
# Helpers (pure).
# ---------------------------------------------------------------------------


def _indexable_text(frame: Frame) -> str:
    """Return the human-readable text to rerank for a frame.

    B-frames carry a JSON correction payload; the description is the
    human-readable text (mirrors :meth:`FrameStore._indexable_text`).
    """
    if frame.frame_type == "B":
        import json

        try:
            payload = json.loads(frame.content)
            return str(payload.get("description", frame.content))
        except (json.JSONDecodeError, TypeError):
            return frame.content
    return frame.content


def _days_since_created(created_at: str, now: datetime) -> float:
    """Days between ``created_at`` and ``now``, both read as UTC.

    ``created_at`` may be SQLite ``YYYY-MM-DD HH:MM:SS`` (UTC), naive ISO
    (UTC by convention), date-only (midnight UTC) or offset-aware. Returns a
    non-negative float (clamped at 0 for future or backdated-ahead values);
    an unparsable legacy value counts as age 0 rather than failing recall.
    """
    try:
        dt = parse_utc(created_at)
    except ValueError:
        return 0.0
    delta = (parse_utc(now) - dt).total_seconds() / 86400.0
    return max(0.0, delta)


def _event_time_in_range(
    event_time: str | None, since: str | None, until: str | None
) -> bool:
    """True when ``event_time`` falls within the inclusive ``[since, until]``.

    A frame with no ``event_time`` (unknown world time) is excluded ONLY when
    a range is set. All three values compare as UTC instants
    (:func:`gluomem.timeutil.utc_key`): naive is UTC, aware is converted,
    and a date-only bound means midnight UTC of that day.
    """
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
