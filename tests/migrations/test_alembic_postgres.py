"""Offline migration and CLI contracts."""

from __future__ import annotations

import pytest

from gluomem.cli import main
from gluomem.migrations import get_heads, get_revisions, normalize_sqlalchemy_url


@pytest.mark.parametrize("command", ["heads", "upgrade", "downgrade"])
def test_migration_cli(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert get_heads() == ["7e2b094eb309"]
    assert get_revisions()[0]["down_revision"] is None
    args = ["migrate", command]
    if command != "heads":
        args += [
            "head" if command == "upgrade" else "0001:base",
            "--sql",
            "--schema",
            "tenant_schema",
            "--vector-dim",
            "512",
        ]
    assert main(args) == 0
    output = capsys.readouterr().out
    if command == "heads":
        assert output.strip() == "7e2b094eb309"
    else:
        assert output.index("CREATE SCHEMA IF NOT EXISTS tenant_schema") < output.index(
            "tenant_schema.alembic_version"
        )
        assert "SET LOCAL search_path TO tenant_schema, public" in output
        if command == "upgrade":
            assert "CREATE EXTENSION IF NOT EXISTS vector SCHEMA public" in output
            assert "VECTOR(512)" in output
            assert "idx_frames_vec_hnsw" in output
            assert "CREATE TABLE source_retractions" in output
        else:
            assert "DROP TABLE memory_frames;" in output
            assert "DROP TABLE sessions;" in output


@pytest.mark.parametrize("prefix", ["postgres", "postgresql"])
def test_normalize_sqlalchemy_url(prefix: str) -> None:
    assert normalize_sqlalchemy_url(f"{prefix}://u:p@localhost:5432/db") == (
        "postgresql+psycopg://u:p@localhost:5432/db"
    )
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_sqlalchemy_url("   ")


def test_migration_config_uses_current_output_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import io
    import sys

    from gluomem.migrations import make_alembic_config

    importlib.import_module("alembic.config")
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    make_alembic_config().print_stdout("Metadata checked")
    assert stream.getvalue() == "Metadata checked\n"
