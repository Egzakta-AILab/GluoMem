"""Seeded offline tuning on development queries with a held-out promotion gate."""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gluomem import extras
from gluomem.tune.candidate_cache import CandidateCache
from gluomem.tune.rescore import (
    ParamVector,
    evaluate_cache,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
)

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig

__all__ = [
    "TrialRecord",
    "TuneConfig",
    "TuneResult",
    "TuneSpace",
    "run_optuna_tune",
]

METRIC_FUNCTIONS = {"ndcg": ndcg_at_k, "recall": recall_at_k, "hit": hit_at_k}
VALID_TUNE_METRICS = tuple(METRIC_FUNCTIONS)


@dataclass(frozen=True)
class TuneSpace:
    """Search space bounds for cheap retrieval/relevance parameters."""

    rrf_k: tuple[int, int] = (10, 120)
    half_life_days: tuple[float, float] = (3.0, 120.0)
    flat_boost_days: tuple[float, float] = (0.0, 21.0)
    log_coef: tuple[float, float] = (0.01, 0.5)
    raw_weights: tuple[float, float] = (1e-3, 1.0)
    bfs_step_1: tuple[float, float] = (0.1, 1.0)
    imp_critical: tuple[float, float] = (1.2, 3.0)
    imp_important: tuple[float, float] = (1.0, 2.2)
    imp_temporary: tuple[float, float] = (0.3, 1.0)
    imp_deprecated: tuple[float, float] = (0.05, 0.6)
    tune_bfs_steps: bool = True
    tune_importance_mult: bool = True


@dataclass(frozen=True)
class TuneConfig:
    """Configuration for a Tier-1 Optuna tuning run."""

    n_trials: int = 200
    k: int = 10
    metric: str = "ndcg"
    profile: str = "balanced"
    seed: int = 42
    held_out_fraction: float = 0.3
    space: TuneSpace = field(default_factory=TuneSpace)

    def __post_init__(self) -> None:
        if not 0.0 < self.held_out_fraction < 1.0:
            raise ValueError("held_out_fraction must be between 0 and 1")
        if self.metric not in VALID_TUNE_METRICS:
            raise ValueError(
                f"tune.metric must be one of {VALID_TUNE_METRICS}, got {self.metric!r}"
            )
        if self.n_trials < 1:
            raise ValueError(f"tune.n_trials must be >= 1, got {self.n_trials}")
        if self.k < 1:
            raise ValueError(f"tune.k must be >= 1, got {self.k}")


@dataclass(frozen=True)
class TrialRecord:
    """Summary of one evaluated Optuna trial."""

    trial_number: int
    value: float
    ndcg: float
    recall: float
    hit: float
    mrr: float
    is_incumbent: bool
    state: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TuneResult:
    """Outcome of :func:`run_optuna_tune`."""

    metric: str
    k: int
    n_trials: int
    seed: int
    profile: str
    incumbent_score: float
    best_score: float
    improvement: float
    best_trial_number: int
    incumbent_params: ParamVector
    best_params: ParamVector
    trials: list[TrialRecord]
    elapsed_ms: float
    cache_manifest_digest: str
    development_query_ids: tuple[str, ...]
    held_out_query_ids: tuple[str, ...]
    incumbent_dev_score: float
    best_dev_score: float
    promoted: bool

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "k": self.k,
            "n_trials": self.n_trials,
            "completed_trials": sum(1 for t in self.trials if t.state == "COMPLETE"),
            "pruned_trials": sum(1 for t in self.trials if t.state == "PRUNED"),
            "seed": self.seed,
            "profile": self.profile,
            "incumbent_score": self.incumbent_score,
            "best_score": self.best_score,
            "improvement": self.improvement,
            "best_trial_number": self.best_trial_number,
            "development_query_ids": list(self.development_query_ids),
            "held_out_query_ids": list(self.held_out_query_ids),
            "incumbent_dev_score": self.incumbent_dev_score,
            "best_dev_score": self.best_dev_score,
            "promoted": self.promoted,
            "split_unit": "query (within-dataset, not independent-history generalization)",
            "model_calls": 0,
            "cache_manifest_digest": self.cache_manifest_digest,
        }


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def split_queries(qids: list[str], fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Partition query IDs by seeded hash, ensuring nonempty dev and held-out sets."""
    if len(qids) < 2:
        raise ValueError("held-out tuning requires at least two labeled queries")
    ordered = sorted(qids, key=lambda qid: hashlib.sha256(f"{seed}:{qid}".encode()).digest())
    count = max(1, min(len(qids) - 1, round(len(qids) * fraction)))
    return sorted(ordered[count:]), sorted(ordered[:count])


def _incumbent_trial_params(incumbent: ParamVector, space: TuneSpace) -> dict[str, Any]:
    params: dict[str, Any] = {
        f"raw_w_{signal}": _clamp(getattr(incumbent, f"w_{signal}"), space.raw_weights)
        for signal in ("t", "p", "c", "i")
    }
    for name in ("rrf_k", "half_life_days", "flat_boost_days", "log_coef"):
        params[name] = _clamp(getattr(incumbent, name), getattr(space, name))
    if space.tune_bfs_steps:
        steps = list(incumbent.bfs_steps) + [0.0] * 4
        params["bfs_step_1"] = _clamp(steps[1], space.bfs_step_1)
        for i in (2, 3):
            params[f"bfs_step_{i}"] = _clamp(steps[i], (0.0, params[f"bfs_step_{i - 1}"]))
    if space.tune_importance_mult:
        for level in ("critical", "important", "temporary", "deprecated"):
            params[f"imp_{level}"] = _clamp(
                incumbent.importance_mult[level], getattr(space, f"imp_{level}")
            )
    return params


def _sample_param_vector(trial: Any, incumbent: ParamVector, space: TuneSpace) -> ParamVector:
    raw = np.array(
        [
            trial.suggest_float("raw_w_t", *space.raw_weights),
            trial.suggest_float("raw_w_p", *space.raw_weights),
            trial.suggest_float("raw_w_c", *space.raw_weights),
            trial.suggest_float("raw_w_i", *space.raw_weights),
        ],
        dtype=np.float64,
    )
    weights = raw / float(raw.sum())
    rrf_k = int(trial.suggest_int("rrf_k", space.rrf_k[0], space.rrf_k[1]))
    half_life = float(
        trial.suggest_float("half_life_days", space.half_life_days[0], space.half_life_days[1])
    )
    flat_boost = float(
        trial.suggest_float("flat_boost_days", space.flat_boost_days[0], space.flat_boost_days[1])
    )
    log_coef = float(trial.suggest_float("log_coef", space.log_coef[0], space.log_coef[1]))

    if space.tune_bfs_steps:
        s1 = float(trial.suggest_float("bfs_step_1", *space.bfs_step_1))
        s2 = float(trial.suggest_float("bfs_step_2", 0.0, s1))
        s3 = float(trial.suggest_float("bfs_step_3", 0.0, s2))
        bfs_steps: tuple[float, ...] = (1.0, s1, s2, s3, 0.0)
    else:
        bfs_steps = incumbent.bfs_steps

    if space.tune_importance_mult:
        imp_critical = float(trial.suggest_float("imp_critical", *space.imp_critical))
        imp_important = float(trial.suggest_float("imp_important", *space.imp_important))
        imp_temporary = float(trial.suggest_float("imp_temporary", *space.imp_temporary))
        imp_deprecated = float(trial.suggest_float("imp_deprecated", *space.imp_deprecated))
        importance_mult: dict[str, float] = {
            "critical": imp_critical,
            "important": imp_important,
            "normal": 1.0,
            "temporary": imp_temporary,
            "deprecated": imp_deprecated,
        }
    else:
        importance_mult = dict(incumbent.importance_mult)

    return ParamVector(
        w_t=float(weights[0]),
        w_p=float(weights[1]),
        w_c=float(weights[2]),
        w_i=float(weights[3]),
        rrf_k=rrf_k,
        half_life_days=half_life,
        flat_boost_days=flat_boost,
        log_coef=log_coef,
        bfs_steps=bfs_steps,
        bfs_depth=incumbent.bfs_depth,
        importance_mult=importance_mult,
        pool_size=incumbent.pool_size,
    )


def run_optuna_tune(
    cache: CandidateCache,
    cfg: AppConfig,
    *,
    tune_cfg: TuneConfig | None = None,
    out_dir: Path | str | None = None,
) -> TuneResult:
    """Optimize on development queries, then gate the winner on held-out queries."""
    extras.require("tune", "Offline retrieval parameter tuning")
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    resolved_tune = tune_cfg or TuneConfig(profile=cfg.search.default_profile)
    incumbent = ParamVector.from_config(cfg, profile=resolved_tune.profile)
    qids = [qid for qid in cache.query_ids() if cache.qrel(qid)]
    dev, held_out = split_queries(qids, resolved_tune.held_out_fraction, resolved_tune.seed)
    k = resolved_tune.k
    metric = resolved_tune.metric
    incumbent_dev = evaluate_cache(cache, incumbent, k=k, query_ids=dev)[metric]
    incumbent_score = evaluate_cache(cache, incumbent, k=k, query_ids=held_out)[metric]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=resolved_tune.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="gluomem-offline")
    study.enqueue_trial(_incumbent_trial_params(incumbent, resolved_tune.space))
    trial_params_map: dict[int, ParamVector] = {}
    start = time.perf_counter()

    def objective(trial: Any) -> float:
        sampled = _sample_param_vector(trial, incumbent, resolved_tune.space)
        p = incumbent if trial.number == 0 else sampled
        trial_params_map[trial.number] = p
        metrics = evaluate_cache(cache, p, k=k, query_ids=dev)
        trial.set_user_attr("metrics", metrics)
        return metrics[metric]

    study.optimize(objective, n_trials=resolved_tune.n_trials)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    winner = study.best_trial
    candidate = trial_params_map[winner.number]
    candidate_score = evaluate_cache(cache, candidate, k=k, query_ids=held_out)[metric]
    promoted = (
        winner.value is not None
        and winner.value > incumbent_dev
        and candidate_score >= incumbent_score
    )
    best_score = candidate_score if promoted else incumbent_score
    best_params = candidate if promoted else incumbent
    best_trial_num = winner.number if promoted else 0
    trial_records = [
        TrialRecord(
            trial_number=t.number,
            value=float(t.value or 0.0),
            ndcg=t.user_attrs["metrics"]["ndcg"],
            recall=t.user_attrs["metrics"]["recall"],
            hit=t.user_attrs["metrics"]["hit"],
            mrr=t.user_attrs["metrics"]["mrr"],
            is_incumbent=t.number == 0,
            state=t.state.name,
            params=trial_params_map[t.number].to_dict(),
        )
        for t in study.trials
    ]

    result = TuneResult(
        metric=resolved_tune.metric,
        k=k,
        n_trials=resolved_tune.n_trials,
        seed=resolved_tune.seed,
        profile=resolved_tune.profile,
        incumbent_score=incumbent_score,
        best_score=best_score,
        improvement=max(0.0, best_score - incumbent_score),
        best_trial_number=best_trial_num,
        incumbent_params=incumbent,
        best_params=best_params,
        trials=trial_records,
        elapsed_ms=elapsed_ms,
        cache_manifest_digest=cache.manifest.digest(),
        development_query_ids=tuple(dev),
        held_out_query_ids=tuple(held_out),
        incumbent_dev_score=incumbent_dev,
        best_dev_score=float(winner.value or 0.0),
        promoted=promoted,
    )

    if out_dir is not None:
        _write_tune_artifacts(Path(out_dir), result)

    return result


def _write_tune_artifacts(out_dir: Path, result: TuneResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(result.to_manifest_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "trials.jsonl").open("w", encoding="utf-8") as fh:
        for rec in result.trials:
            fh.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")
    best_payload = {
        "best_trial_number": result.best_trial_number,
        "metric": result.metric,
        "k": result.k,
        "incumbent_score": result.incumbent_score,
        "best_score": result.best_score,
        "improvement": result.improvement,
        "params": result.best_params.to_dict(),
        "config_overrides": result.best_params.to_config_overrides(profile=result.profile),
    }
    (out_dir / "best_params.json").write_text(
        json.dumps(best_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
