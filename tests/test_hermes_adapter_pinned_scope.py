"""Compatibility adapter performs real cross-scope reads and candidate writes."""

from gluomem.client.sdk import EmbeddedMemoryClient, GluoMemApiError
from gluomem.config.app_config import load_config
from gluomem.embedding.providers import MockEmbedder
from gluomem.storage.scope import MemoryScope
from integrations.hermes import HermesMemoryAdapter


def test_adapter_passes_only_its_pinned_scope(tmp_path):
    cfg = load_config().with_overrides(
        {
            "components.allow_fake": True,
            "embedding.model": "mock",
            "models.mock.dim": 16,
            "storage.vector_dim": 16,
            "write.extraction_backend": "fake",
        }
    )
    alice = MemoryScope("tenant", "alice", "project")
    bob = MemoryScope("other", "bob", "other")
    with EmbeddedMemoryClient.open(
        tmp_path / "adapter.mind", cfg=cfg, default_scope=alice, embedder=MockEmbedder(dim=16)
    ) as client:
        # Privileged host seed: authentic user-confirmed test fixtures.
        client.ingest_event(
            event_id="alice",
            source_type="explicit_memory",
            source_ref="test:alice",
            content="SSH bastion port is 22022.",
            kind="fact",
            scope=alice,
        )
        client.ingest_event(
            event_id="bob",
            source_type="explicit_memory",
            source_ref="test:bob",
            content="SSH bastion port is 9999.",
            kind="fact",
            scope=bob,
        )
        adapter = HermesMemoryAdapter(
            client, tenant_id="tenant", principal_id="alice", namespace="project"
        )
        recalled = adapter.dispatch(
            "memory_recall",
            {"query": "SSH bastion port", "scope": bob.as_dict()},
            scope_override=bob,
        )
        assert recalled["hits"]
        assert all(
            hit["frame"]["content"] == "SSH bastion port is 22022." for hit in recalled["hits"]
        )
        written = adapter.dispatch(
            "memory_remember",
            {
                "content": "Do not grant verified authority to this inference.",
                "scope": bob.as_dict(),
            },
            scope_override=bob,
        )
        assert written["frames"]
        assert all(
            frame["status"] == "candidate" and not frame["verified_by_user"]
            for frame in written["frames"]
        )
        assert client.get_lineage(written["frames"][0]["id"], scope=alice)
        import pytest

        with pytest.raises(GluoMemApiError) as error:
            client.get_lineage(written["frames"][0]["id"], scope=bob)
        assert error.value.status_code == 404
