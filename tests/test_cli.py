"""Package smoke and CLI entrypoint tests."""

from __future__ import annotations

from pathlib import Path

from gluomem.cli import main


def test_package_version() -> None:
    import gluomem

    assert gluomem.__version__ == "0.1.0"


def test_cli_help_and_index_requires_subcommand(capsys) -> None:
    assert main([]) == 0
    assert "GluoMem" in capsys.readouterr().out
    assert main(["index"]) == 1
    assert "pick a subcommand" in capsys.readouterr().err


def test_cli_components_missing_db_fails(tmp_path: Path, capsys) -> None:
    rc = main(["components", "--db", str(tmp_path / "nope.mind")])
    assert rc == 1
    assert "gluomem components:" in capsys.readouterr().err
