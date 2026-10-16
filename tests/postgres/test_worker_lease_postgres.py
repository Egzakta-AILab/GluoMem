"""Live PostgreSQL lease renewal while extraction is blocked."""

import pytest

from tests.contracts.test_worker_heartbeat import (
    test_slow_extraction_cannot_be_reclaimed as _slow_extraction_contract,
)

pytestmark = pytest.mark.postgres


def test_postgres_slow_extraction_keeps_lease(mind_db, frame_store, cfg) -> None:
    _slow_extraction_contract(mind_db, frame_store, cfg)
