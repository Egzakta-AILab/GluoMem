"""Offline retrieval parameter tuning and ablation engine (`DESIGN.md` §9)."""

from __future__ import annotations

from gluomem.tune.ablate import (
    AblationReport,
    AblationRow,
    AblationVariant,
    default_ablation_variants,
    format_ablation_table,
    run_ablation,
)
from gluomem.tune.candidate_cache import (
    CacheManifest,
    CandidateCache,
    CandidateRow,
    QuerySpec,
    build_cache_for_benchmark,
    build_cache_for_db,
)
from gluomem.tune.objective import (
    TrialRecord,
    TuneConfig,
    TuneResult,
    TuneSpace,
    run_optuna_tune,
)
from gluomem.tune.rescore import (
    DEFAULT_IMPORTANCE_MULT,
    ParamVector,
    evaluate_cache,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    rescore,
    rescore_scores,
)

__all__ = [
    "DEFAULT_IMPORTANCE_MULT",
    "AblationReport",
    "AblationRow",
    "AblationVariant",
    "CacheManifest",
    "CandidateCache",
    "CandidateRow",
    "ParamVector",
    "QuerySpec",
    "TrialRecord",
    "TuneConfig",
    "TuneResult",
    "TuneSpace",
    "build_cache_for_benchmark",
    "build_cache_for_db",
    "default_ablation_variants",
    "evaluate_cache",
    "format_ablation_table",
    "hit_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "rescore",
    "rescore_scores",
    "run_ablation",
    "run_optuna_tune",
]
