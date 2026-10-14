"""Tests for the Serbian/English KVARK replay fixture (sr_en_kvark_v1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gluomem.bench.replay import (
    REPLAY_FIXTURES,
    ReplayFixtureError,
    load_registered_fixture,
    run_replay_fixture,
)
from gluomem.config.app_config import AppConfig, load_config
from gluomem.embedding.providers import MockEmbedder


@pytest.fixture
def replay_cfg() -> AppConfig:
    return load_config().with_overrides(
        {
            "embedding.model": "mock",
            "components.allow_fake": True,
            "search.rerank.enabled": False,
        }
    )


def test_sr_en_kvark_fixture_registration_and_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry, histories, sha = load_registered_fixture("sr_en_kvark_v1")
    assert sha == entry.sha256
    assert entry.readiness == "fixture"
    competencies = {p.competency for h in histories for p in h.probes}
    assert competencies == {
        "accurate_retrieval",
        "conflict_resolution",
        "cross_script_multilingual",
    }

    tampered = tmp_path / "sr_en_tampered.json"
    tampered.write_text(
        REPLAY_FIXTURES["sr_en_kvark_v1"].read_text(encoding="utf-8").replace("Avala-9", "Avala-0"),
        encoding="utf-8",
    )
    monkeypatch.setitem(REPLAY_FIXTURES, "sr_en_kvark_v1", tampered)
    with pytest.raises(ReplayFixtureError, match="registry pin"):
        load_registered_fixture("sr_en_kvark_v1")


def test_sr_en_kvark_replay_cross_script_and_conflict_resolution(
    replay_cfg: AppConfig, tmp_path: Path
) -> None:
    embedder = MockEmbedder(dim=replay_cfg.storage.vector_dim)
    receipt, digest = run_replay_fixture(
        replay_cfg,
        benchmark_id="sr_en_kvark_v1",
        embedder=embedder,
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
    )
    assert digest
    assert receipt["denominators"]["histories"] == 2
    assert receipt["denominators"]["probes"] == 6
    assert receipt["denominators"]["quarantined_events"] == 0

    by_family_scope = {
        (s["tag"]["metric_family"], s["scope"]): s["value"] for s in receipt["scores"]
    }
    assert by_family_scope[("evidence_hit@5", "overall")] == 1.0
    assert by_family_scope[("context_substring_match@5", "overall")] == 1.0
    assert by_family_scope[("current_over_stale@5", "overall")] == 1.0
    assert (
        by_family_scope[("evidence_hit@5", "competency:cross_script_multilingual")]
        == 1.0
    )
