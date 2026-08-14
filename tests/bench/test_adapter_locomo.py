"""LoCoMo adapter — qa parse, eligibility, SHA pinning, dia bridges, fetch."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from gluomem.bench.adapters import make_adapter
from gluomem.bench.adapters.locomo import DatasetError, LoCoMoAdapter
from gluomem.config.app_config import AppConfig, load_config

FIXTURE = Path(__file__).parent.parent / "fixtures" / "locomo_tiny.json"
FIXTURE_SHA = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()


@pytest.fixture
def cfg() -> AppConfig:
    return load_config().with_overrides(
        {"benchmark.datasets.locomo.sha256": FIXTURE_SHA}
    )


@pytest.fixture
def adapter(cfg: AppConfig) -> LoCoMoAdapter:
    return LoCoMoAdapter(cfg, path=FIXTURE)


def test_load_eligibility_and_counts(adapter: LoCoMoAdapter) -> None:
    samples = adapter.load()
    # 6 qa rows in the fixture; the category-5 adversarial row (no "answer"
    # key, only "adversarial_answer") is ineligible.
    assert len(samples) == 5
    assert adapter.eligible_counts() == {1: 1, 2: 1, 3: 1, 4: 2}
    assert all(s.category != 5 for s in samples)


def test_category_normalization_and_names(adapter: LoCoMoAdapter) -> None:
    by_id = {s.id: s for s in adapter.load()}
    # Fixture stores category "3" as a STRING (real LoCoMo mixes types).
    open_ended = by_id["conv-tiny-1:qa:2"]
    assert open_ended.category == 3
    assert open_ended.category_name == "open-ended"
    assert by_id["conv-tiny-1:qa:0"].category_name == "multi-hop"
    assert by_id["conv-tiny-1:qa:1"].category_name == "temporal"


def test_sample_ids_and_evidence(adapter: LoCoMoAdapter) -> None:
    first = adapter.load()[0]
    assert first.id == "conv-tiny-1:qa:0"
    assert first.conv_id == "conv-tiny-1"
    assert first.evidence == ("D1:2", "D3:1")


def test_sha_verify_pass_and_fail(cfg: AppConfig) -> None:
    LoCoMoAdapter(cfg, path=FIXTURE).verify()  # pinned sha matches
    bad = cfg.with_overrides({"benchmark.datasets.locomo.sha256": "0" * 64})
    with pytest.raises(DatasetError, match="sha256 mismatch"):
        LoCoMoAdapter(bad, path=FIXTURE).verify()


def test_missing_file_names_fetch(cfg: AppConfig, tmp_path: Path) -> None:
    adapter = LoCoMoAdapter(cfg, path=tmp_path / "absent.json")
    with pytest.raises(DatasetError, match="bench fetch"):
        adapter.load()


def test_dia_content_matches_ingested_raw_turns(
    cfg: AppConfig, adapter: LoCoMoAdapter, tmp_path: Path
) -> None:
    """The dia bridge must reproduce EXACTLY what ingest writes as raw-turn
    frame content — including blip-caption folding — else evidence metrics
    silently miss."""
    from gluomem.llm import NullExtractors
    from gluomem.storage.db import MindDB
    from gluomem.write.ingest import ingest_path

    ingest_cfg = cfg.with_overrides(
        {"write.extraction_backend": "none", "embedding.model": "mock"}
    )
    db_path = tmp_path / "tiny.mind"
    ingest_path(
        FIXTURE, ingest_cfg, facade=NullExtractors(), db_path=db_path
    )
    with MindDB(db_path, dim=ingest_cfg.storage.vector_dim) as db:
        rows = db.conn.execute(
            "SELECT content FROM memory_frames WHERE gop_id LIKE '%:session:%'"
        ).fetchall()
    contents = {r[0] for r in rows}
    assert contents  # raw turns were admitted
    reverse = adapter.content_to_dias()
    missing = contents - set(reverse)
    assert not missing, f"ingested raw-turn content not in dia map: {missing}"
    # Caption folding round-trips.
    assert any("[shares photo:" in c for c in contents)


def test_content_to_dias_collision_aware(adapter: LoCoMoAdapter) -> None:
    reverse = adapter.content_to_dias()
    assert reverse["Alice: I adopted a cat yesterday."] == ("D1:3", "D2:3")


def test_fetch_file_url_end_to_end(cfg: AppConfig, tmp_path: Path) -> None:
    """fetch() works offline via a file:// source and verifies the pin."""
    src = tmp_path / "upstream.json"
    shutil.copyfile(FIXTURE, src)
    target = tmp_path / "data" / "locomo10.json"
    fetch_cfg = cfg.with_overrides(
        {"benchmark.datasets.locomo.url": src.as_uri()}
    )
    adapter = LoCoMoAdapter(fetch_cfg, path=target)
    got = adapter.fetch()
    assert got == target and target.is_file()
    assert adapter.sha == FIXTURE_SHA
    # Tampered upstream -> loud failure, no file installed.
    src.write_text(src.read_text() + "\n", encoding="utf-8")
    adapter2 = LoCoMoAdapter(fetch_cfg, path=tmp_path / "data2" / "x.json")
    with pytest.raises(DatasetError, match="sha256 mismatch"):
        adapter2.fetch()
    assert not (tmp_path / "data2" / "x.json").exists()


def test_make_adapter_unknown_name_lists_known(cfg: AppConfig) -> None:
    with pytest.raises(ValueError, match="locomo"):
        make_adapter("membench", cfg)
    assert make_adapter("locomo", cfg).name == "locomo"
