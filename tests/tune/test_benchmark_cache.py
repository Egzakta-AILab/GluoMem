"""Benchmark cache logical-clock and sampling-error contracts."""

from datetime import datetime
from pathlib import Path

import pytest

from gluomem.bench.types import Sample
from gluomem.config.app_config import load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.storage.db import MindDB
from gluomem.tune import build_cache_for_benchmark
from gluomem.write.ingest import Conversation, Session


def test_benchmark_cache_anchors_recency_to_conversation_not_wall_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config().with_overrides({"embedding.model": "mock", "components.allow_fake": True})
    embedder = MockEmbedder(dim=cfg.storage.vector_dim)
    with MindDB(tmp_path / "c1.mind", dim=cfg.storage.vector_dim) as db:
        FrameStore(db, embedder=embedder).create_i_frame(
            "c1:session:1",
            "PostgreSQL production database",
            source="import",
            created_at="2026-05-01T12:00:00",
        )

    class Adapter:
        sha = "0" * 64

        def load(self):
            return [Sample("q1", "c1", "PostgreSQL", "answer", 1, "single", ("D1",))]

        def content_to_dias(self):
            return {"PostgreSQL production database": ("D1",)}

        def conversations(self):
            return [
                Conversation(
                    "c1",
                    ("Alice", "Bob"),
                    [
                        Session(1, datetime(2026, 6, 1, 12), []),
                    ],
                )
            ]

    monkeypatch.setattr("gluomem.bench.adapters.make_adapter", lambda *args: Adapter())
    monkeypatch.setattr(
        "gluomem.bench.adapters.locomo.stratified_sample", lambda samples, **kwargs: samples
    )
    cache = build_cache_for_benchmark(
        cfg,
        db_dir=tmp_path,
        embedder=embedder,
        clock=lambda: datetime(2099, 1, 1),
    )
    assert cache.rows("q1")[0].days_since_created == 31
    assert cache.rows("q1")[0].doc_id == "D1"
    assert cache.manifest.lane_fetch == 20

    def fail_sampling(*args, **kwargs):
        raise ValueError("invalid sample categories")

    monkeypatch.setattr("gluomem.bench.adapters.locomo.stratified_sample", fail_sampling)
    with pytest.raises(ValueError, match="invalid sample categories"):
        build_cache_for_benchmark(cfg, db_dir=tmp_path, embedder=embedder)
