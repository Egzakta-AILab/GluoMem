"""Config-driven, ablatable relevance scoring signals."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol, overload

import numpy as np
from numpy.typing import NDArray

from gluomem.config.app_config import ProfileWeights


class ScoringParams(Protocol):
    """Signal parameters shared by runtime and offline rescoring."""

    @property
    def flat_boost_days(self) -> float: ...
    @property
    def half_life_days(self) -> float: ...
    @property
    def log_coef(self) -> float: ...
    @property
    def bfs_depth(self) -> int: ...
    @property
    def bfs_steps(self) -> Sequence[float]: ...
    @property
    def importance_mult(self) -> Mapping[str, float]: ...


@overload
def temporal(days: float, c: ScoringParams) -> float: ...
@overload
def temporal(days: NDArray[np.float64], c: ScoringParams) -> NDArray[np.float64]: ...
def temporal(days: float | NDArray[np.float64], c: ScoringParams) -> float | NDArray[np.float64]:
    """Recency decay anchored on ``created_at``."""
    if isinstance(days, np.ndarray):
        return np.where(days <= c.flat_boost_days, 1.0, 0.5 ** (days / c.half_life_days))
    return 1.0 if days <= c.flat_boost_days else 0.5 ** (days / c.half_life_days)


@overload
def popularity(n: int, c: ScoringParams) -> float: ...
@overload
def popularity(n: NDArray[np.float64], c: ScoringParams) -> NDArray[np.float64]: ...
def popularity(n: int | NDArray[np.float64], c: ScoringParams) -> float | NDArray[np.float64]:
    """Access-count popularity boost."""
    if isinstance(n, np.ndarray):
        return 1.0 + np.log10(1 + n) * c.log_coef
    return 1.0 + math.log10(1 + n) * c.log_coef


@overload
def contextual(distance: int, c: ScoringParams) -> float: ...
@overload
def contextual(distance: NDArray[np.int64], c: ScoringParams) -> NDArray[np.float64]: ...
def contextual(distance: int | NDArray[np.int64], c: ScoringParams) -> float | NDArray[np.float64]:
    """BFS contextual step vector score."""
    if isinstance(distance, np.ndarray):
        steps = np.asarray(c.bfs_steps, dtype=np.float64)
        if not len(steps):
            return np.zeros_like(distance, dtype=np.float64)
        valid = (distance >= 0) & (distance <= c.bfs_depth) & (distance < len(steps))
        return np.where(valid, steps[np.clip(distance, 0, len(steps) - 1)], 0.0)
    return (
        c.bfs_steps[distance]
        if 0 <= distance <= c.bfs_depth and distance < len(c.bfs_steps)
        else 0.0
    )


def importance(level: str, c: ScoringParams) -> float:
    """Importance multiplier for a frame importance level."""
    return c.importance_mult[level]


@overload
def rrf(fts_rank: int | None, vec_rank: int | None, k: int) -> float: ...
@overload
def rrf(
    fts_rank: NDArray[np.float64], vec_rank: NDArray[np.float64], k: int
) -> NDArray[np.float64]: ...
def rrf(
    fts_rank: int | None | NDArray[np.float64], vec_rank: int | None | NDArray[np.float64], k: int
) -> float | NDArray[np.float64]:
    """Fuse zero-based lane ranks; absent array ranks are infinity."""
    if isinstance(fts_rank, np.ndarray) and isinstance(vec_rank, np.ndarray):
        return 1.0 / (k + fts_rank) + 1.0 / (k + vec_rank)
    if isinstance(fts_rank, np.ndarray) or isinstance(vec_rank, np.ndarray):
        raise TypeError("lane ranks must both be scalar or arrays")
    return (1.0 / (k + fts_rank) if fts_rank is not None else 0.0) + (
        1.0 / (k + vec_rank) if vec_rank is not None else 0.0
    )


def relevance(
    *,
    days_since_created: float,
    access_count: int,
    bfs_distance: int,
    importance_level: str,
    scoring: ScoringParams,
    weights: ProfileWeights,
) -> float:
    """Compute the active-profile weighted sum of the four relevance signals."""
    return (
        weights.temporal * temporal(days_since_created, scoring)
        + weights.popularity * popularity(access_count, scoring)
        + weights.contextual * contextual(bfs_distance, scoring)
        + weights.importance * importance(importance_level, scoring)
    )
