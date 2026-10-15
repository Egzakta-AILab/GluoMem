"""Pure scoring formulas and ablation weight tests."""

from __future__ import annotations

from pytest import approx

from gluomem.config.app_config import ProfileWeights, load_config
from gluomem.search.scoring import (
    contextual,
    importance,
    popularity,
    relevance,
    temporal,
)


def test_temporal_flat_boost_and_halflife() -> None:
    cfg = load_config().scoring
    assert temporal(0, cfg) == 1.0
    assert temporal(7, cfg) == 1.0
    assert temporal(8, cfg) == approx(0.5 ** (8 / 30))
    assert temporal(8, cfg) < 1.0
    assert temporal(37, cfg) == approx(0.5 ** (37 / 30))


def test_popularity_log10_and_monotonic() -> None:
    cfg = load_config().scoring
    assert popularity(0, cfg) == 1.0
    assert popularity(9, cfg) == approx(1.1)
    vals = [popularity(n, cfg) for n in range(25)]
    assert vals == sorted(vals)


def test_contextual_step_vector_and_bounds() -> None:
    cfg = load_config().scoring
    assert contextual(0, cfg) == 1.0
    assert contextual(1, cfg) == approx(0.7)
    assert contextual(4, cfg) == 0.0
    assert contextual(100, cfg) == 0.0


def test_importance_all_levels() -> None:
    cfg = load_config().scoring
    for level, expected in cfg.importance_mult.items():
        assert importance(level, cfg) == expected


def test_relevance_weighted_sum_and_signal_zeroability() -> None:
    cfg = load_config()
    scoring = cfg.scoring
    w = scoring.profiles["balanced"].weights

    res = relevance(
        days_since_created=37,
        access_count=9,
        bfs_distance=1,
        importance_level="critical",
        scoring=scoring,
        weights=w,
    )
    expected = (
        0.4 * temporal(37, scoring)
        + 0.2 * popularity(9, scoring)
        + 0.2 * contextual(1, scoring)
        + 0.2 * importance("critical", scoring)
    )
    assert res == approx(expected)

    common = dict(
        days_since_created=30,
        access_count=5,
        importance_level="normal",
        scoring=scoring,
    )
    w_zero_c = ProfileWeights(temporal=0.6, popularity=0.2, contextual=0.0, importance=0.2)
    assert relevance(weights=w_zero_c, bfs_distance=0, **common) == approx(
        relevance(weights=w_zero_c, bfs_distance=4, **common)
    )
    assert relevance(weights=w, bfs_distance=0, **common) > relevance(
        weights=w, bfs_distance=4, **common
    )
