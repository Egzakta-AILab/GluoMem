"""Tests for the Serbian/English KVARK replay fixture (sr_en_kvark_v1)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from gluomem.bench.replay import (
    ReplayRunner,
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


def test_sr_en_kvark_fixture_registration() -> None:
    entry, histories, sha = load_registered_fixture("sr_en_kvark_v1")
    assert sha == entry.sha256
    assert entry.readiness == "fixture"
    competencies = {p.competency for h in histories for p in h.probes}
    assert competencies == {
        "accurate_retrieval",
        "conflict_resolution",
        "cross_script_multilingual",
    }


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
    assert "MemoryAgentBench" not in receipt["label"]
    assert by_family_scope[("evidence_hit@1", "overall")] == 0.5
    assert by_family_scope[("evidence_hit@3", "overall")] == 1.0
    assert by_family_scope[("context_substring_match@5", "overall")] == 1.0
    assert by_family_scope[("current_over_stale@5", "overall")] == 1.0
    assert by_family_scope[("evidence_hit@5", "competency:cross_script_multilingual")] == 1.0


def test_gibberish_queries_score_below_real_queries(replay_cfg: AppConfig, tmp_path: Path) -> None:
    entry, histories, sha = load_registered_fixture("sr_en_kvark_v1")
    assert all(len(h.events) >= 20 for h in histories)
    corrupted = [
        replace(h, probes=tuple(replace(p, question="zqxj vbnm fghj") for p in h.probes))
        for h in histories
    ]

    def score(histories, name: str) -> dict[str, float]:
        receipt = ReplayRunner(
            replay_cfg,
            embedder=MockEmbedder(dim=replay_cfg.storage.vector_dim),
            work_dir=tmp_path / name,
            ks=(1, 3),
        ).run(histories, entry=entry, dataset_sha=sha)
        return {
            s["tag"]["metric_family"]: s["value"]
            for s in receipt["scores"]
            if s["scope"] == "overall"
        }

    real = score(histories, "real")
    gibberish = score(corrupted, "gibberish")
    assert real["evidence_hit@1"] >= gibberish["evidence_hit@1"] + 0.3
    assert real["evidence_hit@3"] >= gibberish["evidence_hit@3"] + 0.5
