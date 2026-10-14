"""CLI ingest tests — offline e2e with backend overrides, error paths."""

from __future__ import annotations

from pathlib import Path

from gluomem.cli import main

FIXTURE = str(Path(__file__).parent.parent / "fixtures" / "locomo_tiny.json")


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
            "embedding.model=mock",  # offline gate: never load MiniLM in tests
            "--override",
            "components.allow_fake=true",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "ingested 1 conversation(s), 3 session(s), 9 turn(s)" in out
    assert "quarantined=1" in out
    # Raw turns only (backend none) — the db holds session-GOP frames.
    import sqlite3

    conn = sqlite3.connect(tmp_path / "cli.mind")
    count = conn.execute(
        "SELECT COUNT(*) FROM memory_frames WHERE gop_id LIKE '%:session:%'"
    ).fetchone()[0]
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


def test_cli_ingest_never_imports_dspy_for_offline_backends(tmp_path: Path) -> None:
    import subprocess
    import sys

    db = str(tmp_path / "probe.mind")
    code = (
        "import sys; from gluomem.cli import main; "
        f"main(['ingest', {FIXTURE!r}, '--db', {db!r}, "
        "'--override', 'write.extraction_backend=none', "
        "'--override', 'embedding.model=mock', "
        "'--override', 'components.allow_fake=true']); "
        "sys.exit(1 if 'dspy' in sys.modules else 0)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_cli_stub_commands_still_stubbed(capsys) -> None:
    # bench is live but demands a subcommand; tune/ablate stay stubs.
    assert main(["bench"]) == 1
    assert "pick a subcommand" in capsys.readouterr().err
    assert main(["tune"]) == 1
    assert "not implemented" in capsys.readouterr().err
