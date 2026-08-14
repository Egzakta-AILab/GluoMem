"""Scoring tests.

The scoring module is PURE: deterministic, no I/O, every constant read from the
passed ``ScoringConfig`` (no literals beyond the documented formula shape:
1.0 flat boost, 0.5 half-life base, log10). Each signal is independently
zeroable via a profile weight of 0 (ablation gate).
"""

from __future__ import annotations

import math

from pytest import approx

from gluomem.config.app_config import AppConfig, ProfileWeights, load_config
from gluomem.search.scoring import (
    contextual,
    importance,
    popularity,
    relevance,
    temporal,
)


def _cfg() -> AppConfig:
    return load_config()


# --- temporal ---------------------------------------------------------------


def test_temporal_flat_boost_and_halflife() -> None:
    """temporal(0)==1.0, temporal(7)==1.0 (boundary),
    temporal(37)==approx(0.5**(37/30))."""
    cfg = _cfg().scoring
    assert cfg.flat_boost_days == 7
    assert cfg.half_life_days == 30

    assert temporal(0, cfg) == 1.0
    # Boundary: days == flat_boost_days is still inside the flat boost.
    assert temporal(7, cfg) == 1.0
    # Beyond the flat boost: half-life decay anchored on created_at.
    assert temporal(37, cfg) == approx(0.5 ** (37 / 30))


def test_temporal_just_above_flat_boost() -> None:
    """days == flat_boost_days + 1 begins decay (not 1.0)."""
    cfg = _cfg().scoring
    assert temporal(8, cfg) == approx(0.5 ** (8 / 30))
    assert temporal(8, cfg) < 1.0


# --- popularity -------------------------------------------------------------


def test_popularity_log10() -> None:
    """popularity(0)==1.0, popularity(9)==approx(1.1)."""
    cfg = _cfg().scoring
    assert cfg.log_coef == 0.1

    assert popularity(0, cfg) == 1.0
    # log10(1 + 9) = log10(10) = 1.0; 1.0 + 1.0 * 0.1 = 1.1
    assert popularity(9, cfg) == approx(1.1)


def test_popularity_monotonic() -> None:
    """popularity is monotonically non-decreasing in n."""
    cfg = _cfg().scoring
    prev = popularity(0, cfg)
    for n in range(1, 100):
        val = popularity(n, cfg)
        assert val >= prev
        prev = val


# --- contextual -------------------------------------------------------------


def test_contextual_step_vector() -> None:
    """bfs_steps=[1.0,0.7,0.4,0.2,0.0], bfs_depth=3:
    contextual(0)==1.0, contextual(1)==0.7, contextual(4)==0.0 (beyond depth)."""
    cfg = _cfg().scoring
    assert cfg.bfs_steps == [1.0, 0.7, 0.4, 0.2, 0.0]
    assert cfg.bfs_depth == 3

    assert contextual(0, cfg) == 1.0
    assert contextual(1, cfg) == approx(0.7)
    # d=4 > bfs_depth(3) -> 0.0 even though bfs_steps[4] exists.
    assert contextual(4, cfg) == 0.0


def test_contextual_beyond_steps_length() -> None:
    """d >= len(bfs_steps) -> 0.0 regardless of depth."""
    cfg = _cfg().scoring
    assert contextual(5, cfg) == 0.0
    assert contextual(100, cfg) == 0.0


# --- importance -------------------------------------------------------------


def test_importance_multipliers() -> None:
    """critical 2.0, normal 1.0, deprecated 0.3 (from config)."""
    cfg = _cfg().scoring
    assert cfg.importance_mult == {
        "critical": 2.0,
        "important": 1.5,
        "normal": 1.0,
        "temporary": 0.7,
        "deprecated": 0.3,
    }

    assert importance("critical", cfg) == 2.0
    assert importance("normal", cfg) == 1.0
    assert importance("deprecated", cfg) == 0.3


def test_importance_all_levels() -> None:
    """Every documented level maps to its config multiplier."""
    cfg = _cfg().scoring
    for level, expected in cfg.importance_mult.items():
        assert importance(level, cfg) == expected


# --- relevance --------------------------------------------------------------


def test_relevance_weighted_sum() -> None:
    """Balanced profile: relevance == approx(0.4t + 0.2p + 0.2c + 0.2i)."""
    cfg = _cfg()
    scoring = cfg.scoring
    weights = scoring.profiles["balanced"].weights
    assert (weights.temporal, weights.popularity, weights.contextual, weights.importance) == (
        0.4,
        0.2,
        0.2,
        0.2,
    )

    # Fixed signal inputs.
    t = temporal(37, scoring)
    p = popularity(9, scoring)
    c = contextual(1, scoring)
    i = importance("critical", scoring)

    expected = (
        weights.temporal * t
        + weights.popularity * p
        + weights.contextual * c
        + weights.importance * i
    )
    result = relevance(
        days_since_created=37,
        access_count=9,
        bfs_distance=1,
        importance_level="critical",
        scoring=scoring,
        weights=weights,
    )
    assert result == approx(expected)


def test_relevance_matches_direct_formula() -> None:
    """relevance with balanced weights == 0.4*temporal + 0.2*popularity
    + 0.2*contextual + 0.2*importance."""
    cfg = _cfg()
    scoring = cfg.scoring
    w = scoring.profiles["balanced"].weights

    result = relevance(
        days_since_created=100,
        access_count=99,
        bfs_distance=2,
        importance_level="important",
        scoring=scoring,
        weights=w,
    )
    direct = (
        0.4 * temporal(100, scoring)
        + 0.2 * popularity(99, scoring)
        + 0.2 * contextual(2, scoring)
        + 0.2 * importance("important", scoring)
    )
    assert result == approx(direct)


# --- signal zeroability -----------------------------------------------------


def test_signal_zeroable_ablation() -> None:
    """Two frames differing ONLY in contextual distance score EQUAL
    relevance under a contextual=0 profile, and DIFFER under contextual=0.2.

    The other three signals (temporal, popularity, importance) are held
    identical so that any relevance difference is attributable solely to the
    contextual signal.
    """
    cfg = _cfg()
    scoring = cfg.scoring

    # Two frames identical except in BFS contextual distance.
    common = dict(
        days_since_created=30,
        access_count=5,
        importance_level="normal",
        scoring=scoring,
    )
    frame_near = dict(**common, bfs_distance=0)  # contextual(0) = 1.0
    frame_far = dict(**common, bfs_distance=4)  # contextual(4) = 0.0

    # Profile A: contextual weight = 0 (ablation). The remaining weights must
    # still form a valid simplex, so redistribute the 0.2 to temporal.
    w_zero = ProfileWeights(temporal=0.6, popularity=0.2, contextual=0.0, importance=0.2)
    # sanity: still a simplex
    assert math.isclose(
        w_zero.temporal + w_zero.popularity + w_zero.contextual + w_zero.importance,
        1.0,
        abs_tol=1e-6,
    )

    rel_near_a = relevance(weights=w_zero, **frame_near)
    rel_far_a = relevance(weights=w_zero, **frame_far)
    # With contextual weight 0, the contextual signal contributes nothing, so
    # the two frames (differing only in contextual distance) score equally.
    assert rel_near_a == approx(rel_far_a), (
        f"contextual=0 profile must zero the signal: {rel_near_a} != {rel_far_a}"
    )

    # Profile B: contextual weight = 0.2 (the balanced default). Now the
    # contextual signal differs (1.0 vs 0.0), so relevance must differ.
    w_active = ProfileWeights(
        temporal=0.4, popularity=0.2, contextual=0.2, importance=0.2
    )
    rel_near_b = relevance(weights=w_active, **frame_near)
    rel_far_b = relevance(weights=w_active, **frame_far)
    assert rel_near_b != approx(rel_far_b), (
        f"contextual=0.2 profile must NOT zero the signal: "
        f"{rel_near_b} == {rel_far_b}"
    )
    assert rel_near_b > rel_far_b, "nearer frame (contextual 1.0) must outrank far one (0.0)"


def test_each_signal_independently_zeroable() -> None:
    """Every signal can be zeroed independently: setting any single weight to 0
    removes that signal's contribution while the others still apply."""
    cfg = _cfg()
    scoring = cfg.scoring

    base = dict(
        days_since_created=30,
        access_count=9,
        bfs_distance=1,
        importance_level="critical",
        scoring=scoring,
    )

    # Full profile (all signals active) for reference.
    w_full = ProfileWeights(
        temporal=0.25, popularity=0.25, contextual=0.25, importance=0.25
    )
    full = relevance(weights=w_full, **base)

    # Zero temporal: the temporal term disappears.
    w_no_t = ProfileWeights(
        temporal=0.0, popularity=1 / 3, contextual=1 / 3, importance=1 / 3
    )
    no_t = relevance(weights=w_no_t, **base)
    expected_no_t = (
        w_no_t.popularity * popularity(9, scoring)
        + w_no_t.contextual * contextual(1, scoring)
        + w_no_t.importance * importance("critical", scoring)
    )
    assert no_t == approx(expected_no_t)
    assert no_t != approx(full)

    # Zero popularity.
    w_no_p = ProfileWeights(
        temporal=1 / 3, popularity=0.0, contextual=1 / 3, importance=1 / 3
    )
    no_p = relevance(weights=w_no_p, **base)
    expected_no_p = (
        w_no_p.temporal * temporal(30, scoring)
        + w_no_p.contextual * contextual(1, scoring)
        + w_no_p.importance * importance("critical", scoring)
    )
    assert no_p == approx(expected_no_p)

    # Zero importance.
    w_no_i = ProfileWeights(
        temporal=1 / 3, popularity=1 / 3, contextual=1 / 3, importance=0.0
    )
    no_i = relevance(weights=w_no_i, **base)
    expected_no_i = (
        w_no_i.temporal * temporal(30, scoring)
        + w_no_i.popularity * popularity(9, scoring)
        + w_no_i.contextual * contextual(1, scoring)
    )
    assert no_i == approx(expected_no_i)
