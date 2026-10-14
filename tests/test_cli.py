"""Package smoke and CLI ingest entrypoint tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from gluomem.cli import main

FIXTURE = str(Path(__file__).parent / "fixtures" / "locomo_tiny.json")


def test_package_version() -> None:
    import gluomem

    assert gluomem.__version__ == "0.1.0"


def test_cli_ingest_backend_none_e2e(tmp_path: Path, capsys) -> None:
    rc = main(
        [
            "ingest",
            FIXTURE,
            "--db",
            str(tmp_path / "cli.mind"),
            "--override",
            "write.extraction_backend=none",
            "--override",
            "embedding.model=mock",
            "--override",
            "components.allow_fake=true",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "ingested 1 conversation(s), 3 session(s), 9 turn(s)" in out
    assert "quarantined=1" in out

    conn = sqlite3.connect(tmp_path / "cli.mind")
    count = conn.execute(
        "SELECT COUNT(*) FROM memory_frames WHERE gop_id LIKE '%:session:%'"
    ).fetchone()[0]
    conn.close()
    assert count == 9


def test_cli_ingest_fake_backend_extracts(tmp_path: Path, capsys) -> None:
    rc = main(
        [
            "ingest",
            FIXTURE,
            "--db",
            str(tmp_path / "cli.mind"),
            "--override",
            "write.extraction_backend=fake",
            "--override",
            "embedding.model=mock",
            "--override",
            "components.allow_fake=true",
        ]
    )
    assert rc == 0
    assert "entities=" in capsys.readouterr().out


def test_cli_ingest_missing_file_fails(tmp_path: Path, capsys) -> None:
    rc = main(["ingest", str(tmp_path / "nope.json"), "--db", str(tmp_path / "x.mind")])
    assert rc == 1
    assert "gluomem ingest:" in capsys.readouterr().err


def test_cli_bench_requires_subcommand(capsys) -> None:
    assert main(["bench"]) == 1
    assert "pick a subcommand" in capsys.readouterr().err
