"""Live PostgreSQL persistence and scoped projection contracts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.sanitize import sanitize
from gluomem.storage.backend import IncompatibleSchemaError, VectorDimMismatchError
from gluomem.storage.db import MindDB
from gluomem.storage.postgres_backend import PostgresBackend
from gluomem.storage.scope import MemoryScope

pytestmark = pytest.mark.postgres


def test_committed_writes_survive_reads_and_nested_transactions(
    mind_db: MindDB, kg: KnowledgeGraph, pg_dsn: str
) -> None:
    schema = mind_db.backend.postgres_schema  # type: ignore[attr-defined]
    observer = PostgresBackend(pg_dsn, postgres_schema=schema)
    try:
        assert mind_db.scalar("SELECT ?::integer + length('?50%') /* ? 10% */", (1,)) == 5
        kg.create_entity("tool", "git")
        kg.create_entity("tool", "svn")
        assert observer.scalar("SELECT COUNT(*) FROM knowledge_entities") == 2
        with mind_db.transaction():
            mind_db.meta_set("outer", "1")
            assert observer.scalar("SELECT COUNT(*) FROM meta WHERE key='outer'") == 0
            with pytest.raises(ValueError), mind_db.transaction():
                mind_db.meta_set("nested", "1")
                raise ValueError("rollback savepoint")
        assert observer.scalar("SELECT COUNT(*) FROM meta WHERE key='outer'") == 1
        assert mind_db.meta_get("nested") is None
        with pytest.raises(ValueError), mind_db.transaction():
            mind_db.meta_set("failed", "1")
            raise ValueError("rollback transaction")
        assert observer.scalar("SELECT COUNT(*) FROM meta WHERE key='failed'") == 0
        mind_db.commit()
        mind_db.rollback()
        assert not mind_db.in_transaction
    finally:
        observer.close()


@pytest.mark.parametrize(
    "other",
    [
        MemoryScope("other", "default", "personal"),
        MemoryScope("local", "other", "personal"),
        MemoryScope("local", "default", "workspace"),
    ],
)
def test_vector_fts_like_scope_and_eligibility(
    mind_db: MindDB, frame_store: FrameStore, other: MemoryScope
) -> None:
    content = "hyperion deployment 100% complete"
    own = frame_store.create_i_frame("gop", content)
    frame_store.create_i_frame("gop", content, scope=other)
    frame_store.create_i_frame("gop", content + " quarantine", trust="quarantined")
    frame_store.create_i_frame("gop", content + " expired", expires_at="2000-01-01")
    stale = frame_store.create_i_frame("gop", content + " stale")
    frame_store.set_status(stale.id, "superseded")
    assert mind_db.fts.search("hyperion", 100) == [own.id]
    assert mind_db.fts.search_like("100%", 100) == [own.id]
    vec = frame_store.embedder.embed(content)
    assert [rid for rid, _ in mind_db.frame_vec.search(vec, 100)] == [own.id]
    own_chunk = mind_db.scalar("SELECT id FROM memory_frame_chunks WHERE frame_id=?", (own.id,))
    assert [rid for rid, _ in mind_db.chunk_vec.search(vec, 100)] == [own_chunk]
    with pytest.raises(VectorDimMismatchError):
        mind_db.frame_vec.add(own.id, vec[:4])


def test_fts_matches_script_variants_and_survives_operator_input(
    mind_db: MindDB, frame_store: FrameStore
) -> None:
    serbian = frame_store.create_i_frame("gop", "Извештај за Београд је послао Ђорђе.")
    english = frame_store.create_i_frame("gop", "The deployments were running smoothly.")
    for query in ("Београд извештај", "Beogradu izveštaj", "Beograd izvestaj Djordje"):
        assert mind_db.fts.search(sanitize(query), 10) == [serbian.id]
    assert mind_db.fts.search(sanitize("deployment run"), 10) == [english.id]
    for hostile in ("or -x", "a & b | !c", "O'Brien:*", "C++ (v1.2.3)", "<-> ''", "!!!"):
        assert mind_db.fts.search(sanitize(hostile), 10) == []


def test_concurrent_opens_do_not_execute_ddl(mind_db: MindDB, pg_dsn: str) -> None:
    schema = mind_db.backend.postgres_schema  # type: ignore[attr-defined]

    def open_backend(_: int) -> int:
        backend = PostgresBackend(pg_dsn, postgres_schema=schema)
        try:
            return backend.scalar("SELECT COUNT(*) FROM memory_frames")
        finally:
            backend.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(open_backend, range(8))) == [0] * 8


@pytest.mark.parametrize("version", [None, "0001_initial_scoped_schema"])
def test_unmigrated_postgres_is_rejected(mind_db: MindDB, pg_dsn: str, version: str | None) -> None:
    schema = mind_db.backend.postgres_schema  # type: ignore[attr-defined]
    if version is None:
        mind_db.execute("DROP TABLE alembic_version")
    else:
        mind_db.execute("UPDATE alembic_version SET version_num=?", (version,))
    with pytest.raises(IncompatibleSchemaError, match="gluomem migrate upgrade"):
        PostgresBackend(pg_dsn, postgres_schema=schema)
