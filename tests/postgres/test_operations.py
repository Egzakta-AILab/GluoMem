"""Live operational probes and a real pg_dump/pg_restore deletion-replay drill."""
# ruff: noqa: F401 -- imported tests deliberately run the shared contracts on PostgreSQL

import os
import subprocess
from uuid import uuid4

import pytest

from tests.contracts.test_worker_liveness import test_worker_pulses_during_blocked_extraction
from tests.service.test_readiness import (
    test_idle_worker_liveness_is_independent_of_api_and_database,
    test_ingestion_backlog_reports_old_pending_jobs,
    test_ingestion_readiness_does_not_treat_fake_components_as_ready,
    test_paused_worker_does_not_announce_liveness,
    test_probe_cleanup_failure_cannot_leave_worker_ready,
    test_stale_future_or_corrupt_worker_is_not_ready,
)
from tests.service.test_restore_drill import (
    test_blocked_restore_rejects_api_reads_and_worker_polls,
    test_invalid_or_uncovered_journal_keeps_restore_blocked,
    test_missing_deletion_journal_blocks_restored_store,
    test_replay_failure_rolls_back_but_leaves_target_blocked,
    test_restore_replays_authorized_scope_erasure,
)
from tests.service.test_restore_drill import (
    test_restore_replays_later_source_deletion as _restore_contract,
)

pytestmark = pytest.mark.postgres


def test_postgres_real_backup_restore_deletion_replay(
    mind_db, frame_store, cfg, tmp_path, monkeypatch, assert_no_recall
):
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    from gluomem.storage.db import MindDB

    container = os.environ.get("GLUOMEM_TEST_PG_CONTAINER")
    if not container:
        pytest.skip("GLUOMEM_TEST_PG_CONTAINER required for actual pg_dump/pg_restore drill")
    assert container is not None
    backend = mind_db.backend
    database = f"restore_drill_{uuid4().hex}"
    with psycopg.connect(backend.dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    target_dsn = make_conninfo(backend.dsn, dbname=database)
    try:
        with psycopg.connect(target_dsn, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION vector")

        def backup_restore(after_backup):
            dump = subprocess.run(
                [
                    "docker",
                    "exec",
                    container,
                    "pg_dump",
                    "-U",
                    "postgres",
                    "-d",
                    "postgres",
                    "-n",
                    backend.postgres_schema,
                    "-Fc",
                ],
                capture_output=True,
                check=True,
            ).stdout
            (tmp_path / "before-delete.dump").write_bytes(dump)
            after_backup()
            subprocess.run(
                [
                    "docker",
                    "exec",
                    "-i",
                    container,
                    "pg_restore",
                    "-U",
                    "postgres",
                    "-d",
                    database,
                    "--no-owner",
                    "--exit-on-error",
                ],
                input=dump,
                capture_output=True,
                check=True,
            )
            return MindDB(
                dim=mind_db.dim,
                backend_type="postgres",
                dsn=target_dsn,
                postgres_schema=backend.postgres_schema,
            )

        monkeypatch.setattr(mind_db, "_restore_drill", backup_restore, raising=False)
        _restore_contract(mind_db, frame_store, cfg, tmp_path, assert_no_recall)
    finally:
        with psycopg.connect(backend.dsn, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))
