"""Pure offline rescoring and IR metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from gluomem.search.scoring import contextual, importance, popularity, rrf, temporal
from gluomem.tune.candidate_cache import CandidateCache, CandidateRow

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig

__all__ = [
    "DEFAULT_IMPORTANCE_MULT",
    "ParamVector",
    "evaluate_cache",
    "hit_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "rescore",
    "rescore_scores",
]

DEFAULT_IMPORTANCE_MULT: dict[str, float] = {
    "critical": 2.0,
    "important": 1.5,
    "normal": 1.0,
    "temporary": 0.7,
    "deprecated": 0.3,
}


@dataclass(frozen=True)
class ParamVector:
    """Cheap retrieval and relevance scoring parameters."""

    w_t: float = 0.4
    w_p: float = 0.2
    w_c: float = 0.2
    w_i: float = 0.2
    rrf_k: int = 60
    half_life_days: float = 30.0
    flat_boost_days: float = 7.0
    log_coef: float = 0.1
    bfs_steps: tuple[float, ...] = (1.0, 0.7, 0.4, 0.2, 0.0)
    bfs_depth: int = 3
    importance_mult: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_IMPORTANCE_MULT)
    )
    pool_size: int = 100
    use_fts: bool = True
    use_vec: bool = True
    use_relevance: bool = True

    @classmethod
    def from_config(
        cls,
        cfg: AppConfig,
        profile: str | None = None,
        *,
        pool_size: int | None = None,
    ) -> ParamVector:
        prof_name = profile or cfg.search.default_profile
        weights = cfg.scoring.profiles[prof_name].weights
        scoring = cfg.scoring
        resolved_pool = (
            int(pool_size) if pool_size is not None else int(cfg.search.rerank.pool or 100)
        )
        return cls(
            w_t=float(weights.temporal),
            w_p=float(weights.popularity),
            w_c=float(weights.contextual),
            w_i=float(weights.importance),
            rrf_k=int(cfg.search.rrf_k),
            half_life_days=float(scoring.half_life_days),
            flat_boost_days=float(scoring.flat_boost_days),
            log_coef=float(scoring.log_coef),
            bfs_steps=tuple(float(x) for x in scoring.bfs_steps),
            bfs_depth=int(scoring.bfs_depth),
            importance_mult=dict(scoring.importance_mult),
            pool_size=resolved_pool,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_config_overrides(self, profile: str = "balanced") -> dict[str, Any]:
        """Return dotted `AppConfig` overrides corresponding to this parameter vector."""
        return {
            "search.rrf_k": int(self.rrf_k),
            "scoring.half_life_days": self.half_life_days,
            "scoring.flat_boost_days": self.flat_boost_days,
            "scoring.log_coef": float(self.log_coef),
            "scoring.bfs_steps": [float(x) for x in self.bfs_steps],
            "scoring.bfs_depth": int(self.bfs_depth),
            "scoring.importance_mult": dict(self.importance_mult),
            f"scoring.profiles.{profile}.weights.temporal": float(self.w_t),
            f"scoring.profiles.{profile}.weights.popularity": float(self.w_p),
            f"scoring.profiles.{profile}.weights.contextual": float(self.w_c),
            f"scoring.profiles.{profile}.weights.importance": float(self.w_i),
        }


def rescore_scores(rows: Sequence[CandidateRow], p: ParamVector) -> list[tuple[str, float]]:
    """Compute `(doc_id, final_score)` sorted descending for `rows` under `p`."""
    scored: list[tuple[str, float, int]] = []
    for idx, r in enumerate(rows):
        fusion = rrf(r.fts_rank if p.use_fts else None, r.vec_rank if p.use_vec else None, p.rrf_k)
        if fusion <= 0.0:
            continue

        if p.use_relevance:
            rel = (
                p.w_t * temporal(r.days_since_created, p)
                + p.w_p * popularity(r.access_count, p)
                + p.w_c * contextual(r.bfs_distance, p)
                + p.w_i * importance(r.importance_level, p)
            )
        else:
            rel = 1.0

        tie_id = r.frame_id if r.frame_id is not None else idx
        scored.append((r.doc_id, fusion * rel, -int(tie_id)))

    scored.sort(key=lambda item: (item[1], item[2]), reverse=True)
    limit = max(0, int(p.pool_size))
    return [(doc_id, score) for doc_id, score, _tie in scored[:limit]]


def rescore(rows: Sequence[CandidateRow], p: ParamVector) -> list[str]:
    """Return ranked `doc_id`s for `rows` under `p` (`DESIGN.md` §9)."""
    return [doc_id for doc_id, _score in rescore_scores(rows, p)]


def _unique_top_k(ranked_doc_ids: Sequence[str], k: int) -> list[str]:
    if k <= 0:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for doc_id in ranked_doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
            if len(out) >= k:
                break
    return out


def ndcg_at_k(ranked_doc_ids: Sequence[str], qrel: Mapping[str, int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank `k`."""
    positive_grades = sorted((int(g) for g in qrel.values() if int(g) > 0), reverse=True)
    if not positive_grades or k <= 0:
        return 0.0

    top_docs = _unique_top_k(ranked_doc_ids, k)
    dcg = 0.0
    for i, doc_id in enumerate(top_docs):
        grade = int(qrel.get(doc_id, 0))
        if grade > 0:
            dcg += (2.0**grade - 1.0) / math.log2(i + 2.0)

    idcg = 0.0
    for i, grade in enumerate(positive_grades[:k]):
        idcg += (2.0**grade - 1.0) / math.log2(i + 2.0)

    return dcg / idcg if idcg > 0.0 else 0.0


def recall_at_k(ranked_doc_ids: Sequence[str], qrel: Mapping[str, int], k: int) -> float:
    """Fraction of positive-grade documents retrieved in top `k`."""
    relevant = {doc_id for doc_id, grade in qrel.items() if int(grade) > 0}
    if not relevant or k <= 0:
        return 0.0
    top_docs = set(_unique_top_k(ranked_doc_ids, k))
    return len(top_docs & relevant) / len(relevant)


def hit_at_k(ranked_doc_ids: Sequence[str], qrel: Mapping[str, int], k: int) -> float:
    """1.0 if at least one positive-grade document appears in top `k`, else 0.0."""
    relevant = {doc_id for doc_id, grade in qrel.items() if int(grade) > 0}
    if not relevant or k <= 0:
        return 0.0
    top_docs = set(_unique_top_k(ranked_doc_ids, k))
    return 1.0 if (top_docs & relevant) else 0.0


def mrr_at_k(ranked_doc_ids: Sequence[str], qrel: Mapping[str, int], k: int) -> float:
    """Mean Reciprocal Rank at `k`."""
    relevant = {doc_id for doc_id, grade in qrel.items() if int(grade) > 0}
    if not relevant or k <= 0:
        return 0.0
    top_docs = _unique_top_k(ranked_doc_ids, k)
    for i, doc_id in enumerate(top_docs):
        if doc_id in relevant:
            return 1.0 / (i + 1.0)
    return 0.0


def rescore_array_scores(cache: CandidateCache, p: ParamVector) -> NDArray[np.float64]:
    """Score precomputed query arrays using the shared runtime signals."""
    a = cache.arrays()
    absent = np.full_like(a.fts, np.inf)
    fusion = rrf(a.fts if p.use_fts else absent, a.vec if p.use_vec else absent, p.rrf_k)
    if p.use_relevance and a.importance_levels:
        imp = np.asarray([importance(level, p) for level in a.importance_levels])[a.importance]
        rel = (
            p.w_t * temporal(a.days, p)
            + p.w_p * popularity(a.access, p)
            + p.w_c * contextual(a.distance, p)
            + p.w_i * imp
        )
        fusion = fusion * rel
    present = ((a.fts < np.inf) & p.use_fts) | ((a.vec < np.inf) & p.use_vec)
    return np.where(a.valid & present, fusion, -np.inf)


def rescore_cache(cache: CandidateCache, p: ParamVector) -> dict[str, list[str]]:
    """Rank all cached query arrays in one vectorized pass."""
    scores = rescore_array_scores(cache, p)
    order = np.lexsort((cache.arrays().ties, -scores), axis=1)
    out = {}
    for i, qid in enumerate(cache.query_ids()):
        indices = order[i, : max(0, p.pool_size)]
        indices = indices[np.isfinite(scores[i, indices])]
        out[qid] = cache.arrays().doc_ids[i, indices].tolist()

    return out


def evaluate_cache(
    cache: CandidateCache,
    p: ParamVector,
    *,
    k: int = 10,
    query_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute mean `ndcg@k`, `recall@k`, `hit@k`, and `mrr@k` across `cache`."""
    qids = list(query_ids) if query_ids is not None else cache.query_ids()
    if not qids:
        return {"ndcg": 0.0, "recall": 0.0, "hit": 0.0, "mrr": 0.0}
    ndcgs: list[float] = []
    recalls: list[float] = []
    hits: list[float] = []
    mrrs: list[float] = []
    rankings = rescore_cache(cache, p)
    for qid in qids:
        qrel = cache.qrel(qid)
        if not qrel:
            continue
        ranked = rankings[qid]
        ndcgs.append(ndcg_at_k(ranked, qrel, k))
        recalls.append(recall_at_k(ranked, qrel, k))
        hits.append(hit_at_k(ranked, qrel, k))
        mrrs.append(mrr_at_k(ranked, qrel, k))
    n = len(ndcgs)
    if n == 0:
        return {"ndcg": 0.0, "recall": 0.0, "hit": 0.0, "mrr": 0.0}
    return {
        "ndcg": sum(ndcgs) / n,
        "recall": sum(recalls) / n,
        "hit": sum(hits) / n,
        "mrr": sum(mrrs) / n,
    }
