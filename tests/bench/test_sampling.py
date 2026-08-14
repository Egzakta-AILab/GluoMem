"""Stratified sampling — determinism, short buckets, manifest round-trip."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gluomem.bench.adapters.locomo import (
    DatasetError,
    LoCoMoAdapter,
    _xorshift32,
    load_sample_manifest,
    stratified_sample,
    write_sample_manifest,
)
from gluomem.bench.types import Sample
from gluomem.config.app_config import load_config

FIXTURE = Path(__file__).parent.parent / "fixtures" / "locomo_tiny.json"
FIXTURE_SHA = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()


def _samples(category: int, n: int) -> list[Sample]:
    return [
        Sample(
            id=f"c1:qa:{category * 100 + i}",
            conv_id="c1",
            question="q",
            gold="g",
            category=category,
            category_name=str(category),
        )
        for i in range(n)
    ]


def test_xorshift32_known_vector() -> None:
    """Reference xorshift32 sequence from seed 42 (<<13, >>17, <<5 masks)."""
    state = 42
    seq = []
    for _ in range(3):
        state = _xorshift32(state)
        seq.append(state)
    assert seq == [11355432, 2836018348, 476557059]


def test_same_seed_same_sample() -> None:
    pool = _samples(1, 50) + _samples(2, 50)
    a = stratified_sample(pool, per_category=10, categories=[1, 2], seed=42)
    b = stratified_sample(pool, per_category=10, categories=[1, 2], seed=42)
    assert [s.id for s in a] == [s.id for s in b]
    assert len(a) == 20
    assert [s.category for s in a] == [1] * 10 + [2] * 10


def test_different_seed_different_order() -> None:
    pool = _samples(1, 50)
    a = stratified_sample(pool, per_category=10, categories=[1], seed=42)
    b = stratified_sample(pool, per_category=10, categories=[1], seed=43)
    assert [s.id for s in a] != [s.id for s in b]


def test_bucket_independent_of_category_order() -> None:
    """Per-bucket seeding: category 2's draw is identical whether categories
    are configured [1, 2] or [2, 1]."""
    pool = _samples(1, 50) + _samples(2, 50)
    ab = stratified_sample(pool, per_category=5, categories=[1, 2], seed=42)
    ba = stratified_sample(pool, per_category=5, categories=[2, 1], seed=42)
    assert [s.id for s in ab if s.category == 2] == [
        s.id for s in ba if s.category == 2
    ]


def test_short_bucket_fails_loud() -> None:
    pool = _samples(1, 3)
    with pytest.raises(DatasetError, match="only 3 eligible"):
        stratified_sample(pool, per_category=10, categories=[1], seed=42)


def test_manifest_round_trip(tmp_path: Path) -> None:
    cfg = load_config().with_overrides(
        {
            "benchmark.datasets.locomo.sha256": FIXTURE_SHA,
            "benchmark.sample.per_category": 1,
        }
    )
    adapter = LoCoMoAdapter(cfg, path=FIXTURE)
    picked = stratified_sample(
        adapter.load(), per_category=1, categories=[1, 2, 3, 4], seed=42
    )
    manifest_path = tmp_path / "sample.json"
    write_sample_manifest(picked, adapter=adapter, cfg=cfg, path=manifest_path)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["dataset_sha"] == FIXTURE_SHA
    assert data["seed"] == 42
    assert len(data["ids"]) == 4

    loaded = load_sample_manifest(manifest_path, adapter=adapter)
    assert [s.id for s in loaded] == [s.id for s in picked]


def test_manifest_sha_drift_fails(tmp_path: Path) -> None:
    cfg = load_config().with_overrides(
        {"benchmark.datasets.locomo.sha256": FIXTURE_SHA}
    )
    adapter = LoCoMoAdapter(cfg, path=FIXTURE)
    picked = stratified_sample(
        adapter.load(), per_category=1, categories=[4], seed=42
    )
    manifest_path = tmp_path / "sample.json"
    write_sample_manifest(picked, adapter=adapter, cfg=cfg, path=manifest_path)
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["dataset_sha"] = "0" * 64
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(DatasetError, match="dataset changed"):
        load_sample_manifest(manifest_path, adapter=adapter)


def test_manifest_unknown_id_fails(tmp_path: Path) -> None:
    cfg = load_config().with_overrides(
        {"benchmark.datasets.locomo.sha256": FIXTURE_SHA}
    )
    adapter = LoCoMoAdapter(cfg, path=FIXTURE)
    manifest_path = tmp_path / "sample.json"
    manifest_path.write_text(
        json.dumps(
            {"dataset_sha": adapter.sha, "ids": ["conv-tiny-1:qa:999"]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="qa:999"):
        load_sample_manifest(manifest_path, adapter=adapter)
