"""Latency and multi-principal isolation benchmark for recall (< 300 ms p95)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from gluomem.config.app_config import AppConfig
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from gluomem.search.hybrid import HybridSearch
from gluomem.storage.db import MindDB
from gluomem.storage.scope import MemoryScope


def test_multi_principal_recall_p95_under_300ms(tmp_path: Path) -> None:
    cfg = AppConfig()
    db = MindDB(tmp_path / "latency.mind", vector_dim=32)
    embedder = MockEmbedder(dim=32)
    target_scope = MemoryScope("egzakta", "user_target", "personal")
    store = FrameStore(db, embedder=embedder, scope=target_scope)
    search = HybridSearch(
        db, store, cfg, embedder=embedder, scope=target_scope
    )

    # Seed 500 frames across multiple principals + 50 frames for target_scope
    noise_scope = MemoryScope("egzakta", "user_noise", "personal")
    for idx in range(200):
        store.create_i_frame(
            "gop-noise",
            f"Noise frame {idx} about distributed consensus and database sharding.",
            scope=noise_scope,
        )
    for idx in range(50):
        store.create_i_frame(
            "gop-target",
            f"Target memory {idx}: deployment window on Friday at {idx % 24:02d}:00 UTC.",
            scope=target_scope,
        )

    latencies_ms: list[float] = []
    for idx in range(25):
        t0 = time.perf_counter()
        res = search.search(
            f"deployment window Friday {idx % 24:02d}:00",
            limit=10,
            scope=target_scope,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        assert len(res.hits) > 0
        assert all(h.frame.scope == target_scope for h in res.hits)

    p95_ms = float(np.percentile(latencies_ms, 95))
    assert p95_ms < 300.0, f"expected p95 < 300ms, got {p95_ms:.2f}ms"
    db.close()
