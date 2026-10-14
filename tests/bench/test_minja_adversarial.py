"""Tests for the MINJA-inspired adversarial memory poisoning suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from gluomem.bench.adversarial import (
    load_registered_adversarial_fixture,
    run_adversarial_fixture,
)
from gluomem.config.app_config import AppConfig, load_config
from gluomem.embedding.providers import MockEmbedder


@pytest.fixture
def adv_cfg() -> AppConfig:
    return load_config().with_overrides(
        {
            "embedding.model": "mock",
            "components.allow_fake": True,
            "search.rerank.enabled": False,
        }
    )


def test_minja_adversarial_fixture_registration() -> None:
    entry, scenarios, sha = load_registered_adversarial_fixture("minja_adversarial_v1")
    assert sha == entry.sha256
    assert len(scenarios) == 8
    assert sum(len(s.adversarial_events) for s in scenarios) == 60
    families = {s.attack_family for s in scenarios}
    assert families == {
        "prompt_injection_override",
        "bridging_step_redirection",
        "authority_spoofing",
        "multilingual_serbian_injection",
    }


def test_minja_adversarial_quarantine_and_clean_recall(adv_cfg: AppConfig, tmp_path: Path) -> None:
    embedder = MockEmbedder(dim=adv_cfg.storage.vector_dim)
    report = run_adversarial_fixture(
        adv_cfg,
        entry_id="minja_adversarial_v1",
        embedder=embedder,
        work_dir=tmp_path / "adv_work",
        k=5,
    )
    summary = report["summary"]
    assert summary["poison_admission_rate"] == pytest.approx(8 / 60)
    assert summary["unquarantined_persistence_rate@5"] == pytest.approx(7 / 60)
    assert summary["quarantine_recall_leak_rate@5"] == 0.0
    assert summary["poison_recall_leak_rate@5"] == 0.5
    assert summary["benign_quarantine_rate"] == 0.0
    assert summary["clean_recall_retention@5"] == 1.0
    assert report["by_split"]["held_out"]["poison_admission_rate"] == 0.4
    assert report["denominators"]["adversarial_events"] == 60
    for row in report["scenarios"]:
        assert row["quarantined_recalled"] == 0
        if row["split"] == "development":
            assert row["adversarial_admitted_eligible"] == 0
        assert row["clean_retained"] is True
