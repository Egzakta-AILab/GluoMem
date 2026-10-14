"""Live scoped HNSW completeness and p95 recall latency on 10,000 frames."""

from __future__ import annotations

import time

import numpy as np
import pytest

from gluomem.config.app_config import AppConfig
from gluomem.frames.store import FrameStore
from gluomem.search.hybrid import HybridSearch
from gluomem.storage.db import MindDB
from gluomem.storage.scope import MemoryScope

pytestmark = pytest.mark.postgres


def test_multi_principal_recall_p95_under_300ms(
    mind_db: MindDB, frame_store: FrameStore, cfg: AppConfig
) -> None:
    target = MemoryScope("egzakta", "target", "personal")
    scopes = [target] + [MemoryScope("egzakta", f"noise_{i}", "personal") for i in range(8)]
    with mind_db.transaction():
        mind_db.executemany(
            "INSERT INTO sessions(tenant_id, principal_id, namespace, gop_id) "
            "VALUES (?, ?, ?, 'gop')",
            [scope.as_tuple() for scope in scopes],
        )
        with mind_db.conn.cursor() as cursor:
            with cursor.copy(
                "COPY memory_frames (id, tenant_id, principal_id, namespace, gop_id, "
                "frame_type, content, folded_content) FROM STDIN"
            ) as copy:
                for idx in range(10000):
                    scope = target if idx < 1500 else scopes[1 + idx % 8]
                    content = f"deployment window Friday memory {idx}"
                    copy.write_row((idx + 1, *scope.as_tuple(), "gop", "I", content, content))
            with cursor.copy("COPY memory_frames_vec (rowid, embedding) FROM STDIN") as copy:
                for idx in range(10000):
                    vec = frame_store.embedder.embed(f"deployment window Friday memory {idx}")
                    copy.write_row((idx + 1, vec))
        mind_db.execute(
            "INSERT INTO memory_frames_fts(rowid, content, folded_content) "
            "SELECT id, content, folded_content FROM memory_frames"
        )
    mind_db.execute("ANALYZE")
    assert mind_db.scalar("SHOW hnsw.iterative_scan") == "strict_order"
    query = frame_store.embedder.embed("deployment window Friday")
    # Force the ANN path so an exact sort cannot mask filtered HNSW under-fetch.
    with mind_db.transaction():
        mind_db.execute("SET LOCAL enable_seqscan=off")
        mind_db.execute("SET LOCAL enable_sort=off")
        plan = mind_db.fetchall(
            "EXPLAIN SELECT v.rowid FROM memory_frames_vec v "
            "JOIN memory_frames m ON m.id=v.rowid WHERE m.tenant_id=? AND m.principal_id=? "
            "AND m.namespace=? ORDER BY v.embedding <-> ?, v.rowid LIMIT 100",
            (*target.as_tuple(), query),
        )
        assert "idx_frames_vec_hnsw" in "\n".join(str(row["QUERY PLAN"]) for row in plan)
        print(
            "\n".join(
                str(row["QUERY PLAN"]) for row in plan if "Index Scan using" in row["QUERY PLAN"]
            )
        )
        mind_db.execute("SET LOCAL hnsw.iterative_scan=off")
        before = len(mind_db.frame_vec.search(query, 100, scope=target))
        mind_db.execute("SET LOCAL hnsw.iterative_scan=strict_order")
        hits = mind_db.frame_vec.search(query, 100, scope=target)
        assert len(hits) == 100
        assert all(1 <= rid <= 1500 for rid, _ in hits)
        assert [distance for _, distance in hits] == sorted(distance for _, distance in hits)
        print(f"Filtered HNSW k=100: iterative_scan=off {before} -> strict_order {len(hits)}")

    search = HybridSearch(mind_db, frame_store, cfg, embedder=frame_store.embedder, scope=target)
    search.search("deployment window Friday", limit=10)
    latencies: list[float] = []
    for idx in range(25):
        start = time.perf_counter()
        result = search.search(f"deployment window Friday memory {idx}", limit=10)
        latencies.append((time.perf_counter() - start) * 1000)
        assert result.hits
        assert all(hit.frame.scope == target for hit in result.hits)
    p95 = float(np.percentile(latencies, 95))
    print(f"PostgreSQL 10000 frames, target=1500, k=100; recall p95={p95:.2f}ms")
    assert p95 < 300, f"recall p95={p95:.2f}ms"
