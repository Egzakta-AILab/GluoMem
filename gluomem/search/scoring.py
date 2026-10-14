"""Config-driven, ablatable relevance scoring signals."""

from __future__ import annotations

import math

from gluomem.config.app_config import ProfileWeights, ScoringConfig


def temporal(days: float, c: ScoringConfig) -> float:
    """Recency decay anchored on ``created_at``."""
    if days <= c.flat_boost_days:
        return 1.0
    return 0.5 ** (days / c.half_life_days)


def popularity(n: int, c: ScoringConfig) -> float:
    """Access-count popularity: ``1.0 + log10(1 + n) * log_coef``."""
    return 1.0 + math.log10(1 + n) * c.log_coef


def contextual(distance: int, c: ScoringConfig) -> float:
    """BFS contextual step vector score."""
    if distance <= c.bfs_depth and distance < len(c.bfs_steps):
        return c.bfs_steps[distance]
    return 0.0


def importance(level: str, c: ScoringConfig) -> float:
    """Importance multiplier for a frame importance level."""
    return c.importance_mult[level]


def relevance(
    *,
    days_since_created: float,
    access_count: int,
    bfs_distance: int,
    importance_level: str,
    scoring: ScoringConfig,
    weights: ProfileWeights,
) -> float:
    """Compute the active-profile weighted sum of the four relevance signals."""
    return (
        weights.temporal * temporal(days_since_created, scoring)
        + weights.popularity * popularity(access_count, scoring)
        + weights.contextual * contextual(bfs_distance, scoring)
        + weights.importance * importance(importance_level, scoring)
    )
