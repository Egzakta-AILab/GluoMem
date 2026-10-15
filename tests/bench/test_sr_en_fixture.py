"""Tests for the Serbian/English KVARK replay fixtures."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gluomem.bench.replay import (
    ReplayRunner,
    load_registered_fixture,
    run_replay_fixture,
)
from gluomem.config.app_config import AppConfig, load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.llm.extraction_eval import normalize
from gluomem.search.multilingual import canonical_terms


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


def test_sr_en_translation_fixture_has_zero_lexical_overlap() -> None:
    entry, histories, sha = load_registered_fixture("sr_en_translation_v1")
    assert sha == entry.sha256
    assert entry.readiness == "fixture"
    probes = [p for h in histories for p in h.probes]
    assert len(histories) == 3
    assert len(probes) >= 15
    assert {p.competency for p in probes} == {
        "cross_lingual_translation",
        "conflict_resolution",
    }

    has_cyrillic_query = False
    has_latin_sr_query = False
    has_english_query = False
    for h in histories:
        for p in h.probes:
            if any("А" <= ch <= "ш" or ch in "ЂЈЉЊЋЏђјљњћџ" for ch in p.question):
                has_cyrillic_query = True
            elif any(ch in "čćžšđČĆŽŠĐ" for ch in p.question):
                has_latin_sr_query = True
            else:
                has_english_query = True
            q_can = set(canonical_terms(p.question))
            q_norm = set(normalize(p.question))
            assert q_can, f"empty canonical_terms for {p.probe_id}"
            assert q_norm, f"empty normalize for {p.probe_id}"
            for e in h.events:
                assert not (q_can & set(canonical_terms(e.text))), (
                    f"{p.probe_id} shares canonical terms with {e.event_id}"
                )
                assert not (q_norm & set(normalize(e.text))), (
                    f"{p.probe_id} shares normalized terms with {e.event_id}"
                )
    assert has_cyrillic_query
    assert has_latin_sr_query
    assert has_english_query


def test_sr_en_translation_mock_embedder_scores_zero_and_concept_embedder_recovers(
    replay_cfg: AppConfig, tmp_path: Path
) -> None:
    entry, histories, _ = load_registered_fixture("sr_en_translation_v1")
    mock_receipt, _ = run_replay_fixture(
        replay_cfg,
        benchmark_id="sr_en_translation_v1",
        embedder=MockEmbedder(dim=replay_cfg.storage.vector_dim),
        work_dir=tmp_path / "mock_work",
        out_dir=tmp_path / "mock_out",
    )
    assert mock_receipt["denominators"]["probes"] == 16
    mock_scores = {
        (s["tag"]["metric_family"], s["scope"]): s["value"] for s in mock_receipt["scores"]
    }
    assert mock_scores[("evidence_hit@1", "overall")] == 0.0
    assert mock_scores[("evidence_hit@3", "overall")] == 0.0
    assert mock_scores[("evidence_hit@5", "overall")] == 0.0
    assert mock_scores[("context_substring_match@5", "overall")] == 0.0
    assert mock_scores[("current_over_stale@5", "overall")] == 0.0

    class _CrossLingualConceptEmbedder(MockEmbedder):
        def __init__(self, dim: int) -> None:
            super().__init__(dim=dim)
            self._vec_by_text: dict[str, np.ndarray] = {}
            slot = 0
            for h in histories:
                ev_by_id = {e.event_id: e.text for e in h.events}
                for p in h.probes:
                    vec = np.zeros(dim, dtype=np.float32)
                    vec[slot % dim] = 1.0
                    slot += 1
                    self._vec_by_text[p.question] = vec
                    for eid in p.evidence:
                        self._vec_by_text[ev_by_id[eid]] = vec

        def embed(self, text: str) -> np.ndarray:
            if text in self._vec_by_text:
                return self._vec_by_text[text].copy()
            return super().embed(text)

        def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
            return [self.embed(t) for t in texts]

    aligned_receipt, _ = run_replay_fixture(
        replay_cfg,
        benchmark_id="sr_en_translation_v1",
        embedder=_CrossLingualConceptEmbedder(dim=replay_cfg.storage.vector_dim),
        work_dir=tmp_path / "aligned_work",
        out_dir=tmp_path / "aligned_out",
    )
    aligned_scores = {
        (s["tag"]["metric_family"], s["scope"]): s["value"] for s in aligned_receipt["scores"]
    }
    assert aligned_scores[("evidence_hit@1", "overall")] == 1.0
    assert aligned_scores[("context_substring_match@1", "overall")] == 1.0
    assert aligned_scores[("current_over_stale@1", "overall")] == 1.0
