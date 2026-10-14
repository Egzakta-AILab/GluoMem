"""Optuna Tier-1 parameter optimization over a cached candidate pool (`DESIGN.md` §9).

Requires the `tune` optional extra (`optuna>=4.0`). Evaluates the incumbent
`AppConfig` parameter vector as Trial 0 so tuning never regresses below the
starting configuration.
"""

from __future__ import annotations

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
    rescore,
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

VALID_TUNE_METRICS: tuple[str, ...] = ("ndcg", "recall", "hit")


@dataclass(frozen=True)
class TuneSpace:
    """Search space bounds for cheap retrieval/relevance parameters."""

    rrf_k: tuple[int, int] = (10, 120)
    half_life_days: tuple[float, float] = (3.0, 120.0)
    flat_boost_days: tuple[float, float] = (0.0, 21.0)
    log_coef: tuple[float, float] = (0.01, 0.5)
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
    enable_pruner: bool = True
    space: TuneSpace = field(default_factory=TuneSpace)

    def __post_init__(self) -> None:
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
            "elapsed_ms": self.elapsed_ms,
            "model_calls": 0,
            "cache_manifest_digest": self.cache_manifest_digest,
        }


def _metric_fn(name: str):  # noqa: ANN202
    if name == "ndcg":
        return ndcg_at_k
    if name == "recall":
        return recall_at_k
    if name == "hit":
        return hit_at_k
    raise ValueError(f"unknown tune metric: {name!r}")


def _incumbent_trial_params(incumbent: ParamVector, space: TuneSpace) -> dict[str, Any]:
    params: dict[str, Any] = {
        "raw_w_t": max(1e-3, min(1.0, float(incumbent.w_t))),
        "raw_w_p": max(1e-3, min(1.0, float(incumbent.w_p))),
        "raw_w_c": max(1e-3, min(1.0, float(incumbent.w_c))),
        "raw_w_i": max(1e-3, min(1.0, float(incumbent.w_i))),
        "rrf_k": int(
            max(space.rrf_k[0], min(space.rrf_k[1], int(incumbent.rrf_k)))
        ),
        "half_life_days": float(
            max(
                space.half_life_days[0],
                min(space.half_life_days[1], float(incumbent.half_life_days)),
            )
        ),
        "flat_boost_days": float(
            max(
                space.flat_boost_days[0],
                min(space.flat_boost_days[1], float(incumbent.flat_boost_days)),
            )
        ),
        "log_coef": float(
            max(
                space.log_coef[0],
                min(space.log_coef[1], float(incumbent.log_coef)),
            )
        ),
    }
    if space.tune_bfs_steps:
        steps = list(incumbent.bfs_steps) + [0.0] * 5
        params["bfs_step_1"] = max(0.1, min(1.0, float(steps[1])))
        params["bfs_step_2"] = max(0.0, min(float(params["bfs_step_1"]), float(steps[2])))
        params["bfs_step_3"] = max(0.0, min(float(params["bfs_step_2"]), float(steps[3])))
    if space.tune_importance_mult:
        im = incumbent.importance_mult
        params["imp_critical"] = max(1.2, min(3.0, float(im.get("critical", 2.0))))
        params["imp_important"] = max(1.0, min(2.2, float(im.get("important", 1.5))))
        params["imp_temporary"] = max(0.3, min(1.0, float(im.get("temporary", 0.7))))
        params["imp_deprecated"] = max(0.05, min(0.6, float(im.get("deprecated", 0.3))))
    return params


def _sample_param_vector(
    trial: Any, incumbent: ParamVector, space: TuneSpace
) -> ParamVector:
    raw = np.array(
        [
            trial.suggest_float("raw_w_t", 1e-3, 1.0),
            trial.suggest_float("raw_w_p", 1e-3, 1.0),
            trial.suggest_float("raw_w_c", 1e-3, 1.0),
            trial.suggest_float("raw_w_i", 1e-3, 1.0),
        ],
        dtype=np.float64,
    )
    weights = raw / float(raw.sum())
    rrf_k = int(trial.suggest_int("rrf_k", space.rrf_k[0], space.rrf_k[1]))
    half_life = float(
        trial.suggest_float(
            "half_life_days", space.half_life_days[0], space.half_life_days[1]
        )
    )
    flat_boost = float(
        trial.suggest_float(
            "flat_boost_days", space.flat_boost_days[0], space.flat_boost_days[1]
        )
    )
    log_coef = float(
        trial.suggest_float("log_coef", space.log_coef[0], space.log_coef[1])
    )

    if space.tune_bfs_steps:
        s1 = float(trial.suggest_float("bfs_step_1", 0.1, 1.0))
        s2 = float(trial.suggest_float("bfs_step_2", 0.0, s1))
        s3 = float(trial.suggest_float("bfs_step_3", 0.0, s2))
        bfs_steps: tuple[float, ...] = (1.0, s1, s2, s3, 0.0)
    else:
        bfs_steps = incumbent.bfs_steps

    if space.tune_importance_mult:
        imp_critical = float(trial.suggest_float("imp_critical", 1.2, 3.0))
        imp_important = float(trial.suggest_float("imp_important", 1.0, 2.2))
        imp_temporary = float(trial.suggest_float("imp_temporary", 0.3, 1.0))
        imp_deprecated = float(trial.suggest_float("imp_deprecated", 0.05, 0.6))
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
    """Run Tier-1 Optuna TPE + Hyperband tuning over `cache` with 0 model calls."""
    extras.require("tune", "Offline retrieval parameter tuning")
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    resolved_tune = tune_cfg or TuneConfig(profile=cfg.search.default_profile)
    incumbent = ParamVector.from_config(cfg, profile=resolved_tune.profile)

    qids = [qid for qid in cache.query_ids() if cache.qrel(qid)]
    if not qids:
        raise ValueError("CandidateCache has no queries with non-empty qrels")

    score_fn = _metric_fn(resolved_tune.metric)
    k = int(resolved_tune.k)

    incumbent_metrics = evaluate_cache(cache, incumbent, k=k)
    incumbent_score = float(incumbent_metrics[resolved_tune.metric])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            group=True,
            seed=resolved_tune.seed,
        )
        pruner: Any = (
            optuna.pruners.HyperbandPruner(
                min_resource=max(1, len(qids) // 4),
                max_resource=max(2, len(qids)),
                reduction_factor=3,
            )
            if resolved_tune.enable_pruner and len(qids) >= 4
            else optuna.pruners.NopPruner()
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.enqueue_trial(_incumbent_trial_params(incumbent, resolved_tune.space))

    trial_params_map: dict[int, ParamVector] = {}
    trial_records: list[TrialRecord] = []
    start = time.perf_counter()

    def objective(trial: Any) -> float:
        if trial.number == 0:
            # Evaluate the exact incumbent parameter vector on Trial 0
            _ = _sample_param_vector(trial, incumbent, resolved_tune.space)
            p = incumbent
        else:
            p = _sample_param_vector(trial, incumbent, resolved_tune.space)
        trial_params_map[trial.number] = p

        running = 0.0
        for step, qid in enumerate(qids, start=1):
            ranked = rescore(cache.rows(qid), p)
            running += float(score_fn(ranked, cache.qrel(qid), k))
            if trial.number > 0 and resolved_tune.enable_pruner and len(qids) >= 4:
                trial.report(running / step, step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return running / len(qids)

    study.optimize(objective, n_trials=resolved_tune.n_trials)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    best_score = incumbent_score
    best_params = incumbent
    best_trial_num = 0

    for t in study.trials:
        p = trial_params_map.get(t.number, incumbent)
        state_name = t.state.name
        if state_name == "COMPLETE" and t.value is not None:
            metrics = evaluate_cache(cache, p, k=k)
            val = float(t.value)
            if val > best_score:
                best_score = val
                best_params = p
                best_trial_num = int(t.number)
        else:
            metrics = {"ndcg": 0.0, "recall": 0.0, "hit": 0.0, "mrr": 0.0}
            val = float(t.value) if t.value is not None else 0.0

        trial_records.append(
            TrialRecord(
                trial_number=int(t.number),
                value=val,
                ndcg=float(metrics["ndcg"]),
                recall=float(metrics["recall"]),
                hit=float(metrics["hit"]),
                mrr=float(metrics["mrr"]),
                is_incumbent=(t.number == 0),
                state=state_name,
                params=p.to_dict(),
            )
        )

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
        "config_overrides": result.best_params.to_config_overrides(
            profile=result.profile
        ),
    }
    (out_dir / "best_params.json").write_text(
        json.dumps(best_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
