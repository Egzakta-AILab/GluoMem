"""Gate A scope/export behavior against a migrated live PostgreSQL database."""

import pytest

from gluomem.client.sdk import EmbeddedMemoryClient
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.store import FrameStore
from tests.test_gate_a_boundaries import (
    BOB,
)
from tests.test_gate_a_boundaries import (
    test_agent_correction_is_a_proposal_not_a_user_receipt as _correction,
)
from tests.test_gate_a_boundaries import (
    test_agent_scope_cannot_override_host as _scope,
)
from tests.test_gate_a_boundaries import (
    test_export_lineage_uses_requested_scope_not_store_default as _export,
)
from tests.test_harnesses_and_mcp import _test_cfg

pytestmark = pytest.mark.postgres


@pytest.fixture
def boundary_client(mind_db):
    cfg = _test_cfg().with_overrides({"models.mock.dim": 384, "storage.vector_dim": 384})
    store = FrameStore(mind_db, embedder=MockEmbedder(dim=384), scope=BOB)
    with EmbeddedMemoryClient(mind_db, store, cfg=cfg, default_scope=BOB) as client:
        yield client


@pytest.mark.parametrize("transport", ["pack", "mcp", "http"])
def test_host_scope_is_authoritative(boundary_client, transport):
    _scope(boundary_client, transport)


@pytest.mark.parametrize("transport", ["embedded", "http"])
def test_scoped_export_lineage(boundary_client, transport):
    _export(boundary_client, transport)


def test_correction_does_not_claim_user_confirmation(boundary_client):
    _correction(boundary_client)
