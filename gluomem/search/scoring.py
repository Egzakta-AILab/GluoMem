"""gluomem.search.scoring — config-driven, ablatable scoring.

Pure functions. Every constant comes from the passed ``ScoringConfig`` — no
literals beyond the documented formula shape (the structural 1.0 flat-boost
anchor, 0.5 half-life base, and ``log10``). ``temporal`` anchors on
``created_at``; the caller computes ``days`` from ``created_at`` (NOT
``event_time``).

Each signal is independently zeroable: a profile weight of 0 removes that
signal's contribution entirely (used for ablations). ``relevance`` is the
active-profile weighted sum.
"""

from __future__ import annotations

import math

from gluomem.config.app_config import ProfileWeights, ScoringConfig


def temporal(days: float, c: ScoringConfig) -> float:
    """Recency decay anchored on ``created_at``.

    ``1.0`` within the flat-boost window (``days <= flat_boost_days``), then
    exponential half-life decay: ``0.5 ** (days / half_life_days)``.
    """
    if days <= c.flat_boost_days:
        return 1.0
    return 0.5 ** (days / c.half_life_days)


def popularity(n: int, c: ScoringConfig) -> float:
    """Access-count popularity: ``1.0 + log10(1 + n) * log_coef``."""
    return 1.0 + math.log10(1 + n) * c.log_coef


def contextual(distance: int, c: ScoringConfig) -> float:
    """BFS contextual step vector.

    Returns ``bfs_steps[distance]`` when ``distance <= bfs_depth`` and
    ``distance < len(bfs_steps)``; otherwise ``0.0``.
    """
    if distance <= c.bfs_depth and distance < len(c.bfs_steps):
        return c.bfs_steps[distance]
    return 0.0


def importance(level: str, c: ScoringConfig) -> float:
    """Importance multiplier for a documented level (critical … deprecated)."""
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
    """Active-profile weighted sum of the four signals.

    ``relevance = w_t*temporal + w_p*popularity + w_c*contextual + w_i*importance``.
    A weight of 0 zeroes that signal's contribution (ablation).
    """
    return (
        weights.temporal * temporal(days_since_created, scoring)
        + weights.popularity * popularity(access_count, scoring)
        + weights.contextual * contextual(bfs_distance, scoring)
        + weights.importance * importance(importance_level, scoring)
    )
