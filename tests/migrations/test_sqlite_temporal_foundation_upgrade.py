"""Upgrade existing scoped SQLite stores without inventing historical timestamps."""

from __future__ import annotations

import pytest

from gluomem.storage.db import MindDB


@pytest.mark.parametrize("old_version", [1, 2, 3, 4, 5])
def test_temporal_upgrade_preserves_frames_and_does_not_backfill_dates(tmp_path, old_version):
    path = tmp_path / "prior.mind"
    db = MindDB(path, dim=4)
    with db.transaction():
        db.execute("INSERT INTO sessions(gop_id) VALUES ('prior')")
        db.execute(
            "INSERT INTO memory_frames(frame_type, gop_id, content, event_time) "
            "VALUES ('I', 'prior', 'prior authorized truth', '2025-01-01')"
        )
        db.execute(
            "INSERT INTO memory_provenance(frame_id, source_type, source_ref) "
            "SELECT id, 'chat', 'prior:unknown' FROM memory_frames"
        )
        columns = {r["name"] for r in db.fetchall("PRAGMA table_info(memory_provenance)")}
        for name in () if old_version >= 4 else ("message_role", "observed_at"):
            if name in columns:
                db.execute(f"ALTER TABLE memory_provenance DROP COLUMN {name}")
        if old_version < 3:
            db.execute("DROP TABLE IF EXISTS memory_frame_temporal")
        if old_version == 1:
            db.execute("DROP TABLE source_retractions")
        db.execute("DROP TABLE memory_claim_semantics")
        if old_version < 5:
            db.execute("DROP TABLE memory_source_support")
        db.execute(f"PRAGMA user_version={old_version}")
    db.close()
    for _ in range(2):
        db = MindDB(path, dim=4)
        try:
            assert db.scalar("PRAGMA user_version") == 6
            assert db.scalar("SELECT COUNT(*) FROM memory_source_support") == 0
            assert db.scalar("SELECT COUNT(*) FROM memory_claim_semantics") == 0
            assert db.scalar("SELECT COUNT(*) FROM memory_frame_temporal") == 0
            assert db.scalar("SELECT content FROM memory_frames") == "prior authorized truth"
            assert db.scalar("SELECT COUNT(*) FROM source_retractions") == 0
            provenance = db.fetchone("SELECT message_role, observed_at FROM memory_provenance")
            assert provenance is not None
            assert dict(provenance) == {
                "message_role": None,
                "observed_at": None,
            }
            assert db.fetchall("PRAGMA foreign_key_check") == []
        finally:
            db.close()


def test_composed_migration_graph_and_offline_downgrade_guards():
    from gluomem.migrations import get_heads, get_revisions, run_downgrade, run_upgrade

    chain = [
        "0001",
        "7e2b094eb309",
        "0003_temporal_state",
        "0004_source_attribution",
        "0005_source_support",
        "0006_semantic_bindings",
    ]
    revisions = get_revisions()
    assert get_heads() == [chain[-1]]
    assert [r["revision"] for r in revisions] == chain
    assert [r["down_revision"] for r in revisions] == [None, *chain[:-1]]
    dsn = "postgresql+psycopg://localhost/gluomem"
    upgrade = run_upgrade(dsn, "head", sql=True)
    for table in ("memory_frame_temporal", "memory_source_support", "memory_claim_semantics"):
        assert "CREATE TABLE " + table in upgrade
    assert "SET LOCAL search_path TO gluomem, public" in upgrade
    assert "INSERT INTO gluomem.memory_claim_semantics" not in upgrade
    downgrade = run_downgrade(dsn, chain[-1] + ":0004_source_attribution", sql=True)
    assert "Cannot downgrade while semantic source bindings exist" in downgrade
    assert "Cannot downgrade while retained source support exists" in downgrade
    assert downgrade.index("DROP TABLE memory_claim_semantics") < downgrade.index(
        "DROP TABLE memory_source_support"
    )
