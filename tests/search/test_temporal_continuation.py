"""Equal-length, equal-vector distractors isolate temporal continuation control flow."""

import numpy as np
import pytest

from gluomem.client.sdk import EmbeddedMemoryClient
from gluomem.embedding.fingerprint import EmbedderFingerprint
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.search.hybrid import HybridSearch


class ConstantEmbedder(MockEmbedder):
    @property
    def fingerprint(self):
        return EmbedderFingerprint(
            implementation="authored.constant",
            model="constant",
            revision=None,
            dim=self.dim,
            preprocessing="none",
        )

    def _vector(self, text):
        return np.ones(self.dim, dtype=np.float32)


@pytest.mark.parametrize("lane", ["fts", "vector"])
def test_temporal_match_beyond_early_equal_length_distractors(mind_db, cfg, monkeypatch, lane):
    cfg = cfg.with_overrides({"search.abstain.enabled": False})
    store = FrameStore(mind_db, embedder=ConstantEmbedder())
    for i in range(25):
        store.create_i_frame(f"d{i}", f"alpha evidence {i:02d}", event_time="2025-01-01")
    target = store.create_i_frame("target", "alpha evidence 25", event_time="2026-10-04")
    search = HybridSearch(mind_db, store, cfg)
    if lane == "fts":
        monkeypatch.setattr(mind_db.chunk_vec, "search", lambda *args, **kwargs: [])
        monkeypatch.setattr(mind_db.frame_vec, "search", lambda *args, **kwargs: [])
        assert target.id not in mind_db.fts.search("alpha", 10, scope=store.scope)
    else:
        monkeypatch.setattr(search, "_fts_lane", lambda *args, **kwargs: [])
        chunk_ids = [
            cid
            for cid, _ in mind_db.chunk_vec.search(
                np.ones(store.embedder.dim, dtype=np.float32), 10, scope=store.scope
            )
        ]
        assert target.id not in store.resolve_chunk_frames(chunk_ids).values()
    result = EmbeddedMemoryClient(mind_db, store, cfg=cfg, search=search).recall(
        "alpha",
        k=1,
        since="2026-10-01",
        until="2026-10-31",
    )
    assert [hit.frame.id for hit in result.hits] == [target.id]
    assert result.retrieval.quota_satisfied
    diagnostic = result.retrieval.lanes[lane]
    assert diagnostic.scanned > 10
    assert not diagnostic.budget_exhausted


def test_candidate_budget_does_not_claim_lane_exhaustion(mind_db, cfg, monkeypatch):
    cfg = cfg.with_overrides({"search.abstain.enabled": False})
    store = FrameStore(mind_db, embedder=ConstantEmbedder())
    for i in range(12):
        store.create_i_frame(f"d{i}", f"alpha evidence {i:02d}", event_time="2025-01-01")
    target = store.create_i_frame("target", "alpha evidence 12", event_time="2026-10-04")
    search = HybridSearch(mind_db, store, cfg)
    monkeypatch.setattr(mind_db.chunk_vec, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(mind_db.frame_vec, "search", lambda *args, **kwargs: [])
    client = EmbeddedMemoryClient(mind_db, store, cfg=cfg, search=search)
    capped = client.recall("alpha", k=1, since="2026-10-01", candidate_budget=8)
    assert not capped.hits
    assert capped.retrieval.stop_reason == "budget_exhausted"
    assert capped.retrieval.candidate_budget == 8
    assert capped.retrieval.lanes["fts"].scanned == 8
    assert capped.retrieval.lanes["fts"].budget_exhausted
    assert not capped.retrieval.lanes["fts"].exhausted
    found = client.recall("alpha", k=1, since="2026-10-01", candidate_budget=16)
    assert [hit.frame.id for hit in found.hits] == [target.id]
    assert found.retrieval.stop_reason == "quota_satisfied"
    exhausted = client.recall("alpha", k=1, since="2027-01-01", candidate_budget=16)
    assert not exhausted.hits
    assert exhausted.retrieval.stop_reason == "lane_exhausted"
    assert all(lane.exhausted for lane in exhausted.retrieval.lanes.values())


def test_score_abstention_is_not_reported_as_lane_exhaustion(mind_db, cfg):
    cfg = cfg.with_overrides({"search.abstain.enabled": True, "search.abstain.threshold": 1e9})
    store = FrameStore(mind_db, embedder=ConstantEmbedder())
    for i in range(8):
        store.create_i_frame(str(i), f"alpha evidence {i:02d}")
    result = EmbeddedMemoryClient(mind_db, store, cfg=cfg).recall("alpha", k=1)
    assert result.abstained and not result.hits
    assert not result.retrieval.lanes["fts"].exhausted
    assert result.retrieval.stop_reason == "abstained"
