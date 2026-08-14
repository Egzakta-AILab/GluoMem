"""CLI bench — offline e2e via config-flip fake backends, exit codes."""

from __future__ import annotations

import json
from pathlib import Path

from gluomem.cli import main
from tests.bench.conftest import BENCH_OVERRIDES
from tests.bench.test_runner import FAKE_BACKENDS


def _overrides(*extra: str) -> list[str]:
    merged = {**BENCH_OVERRIDES, **FAKE_BACKENDS}
    flags: list[str] = []
    for key, value in merged.items():
        flags.extend(["--override", f"{key}={value}"])
    for pair in extra:
        flags.extend(["--override", pair])
    return flags


def _find_run_dir(out: Path) -> Path:
    (run_dir,) = [p for p in out.iterdir() if p.is_dir()]
    return run_dir


def test_bench_run_offline_e2e(tmp_path: Path, capsys) -> None:
    out = tmp_path / "runs"
    rc = main(
        ["bench", "run", "--dataset", "locomo", "--retriever", "bm25",
         "--out", str(out), *_overrides()]
    )
    assert rc == 0
    stdout = capsys.readouterr().out
    assert "qa_accuracy=" in stdout  # tagged headline printed
    assert "judge=trio_strict" in stdout
    run_dir = _find_run_dir(out)
    for name in ("manifest.json", "predictions.jsonl", "judgments.jsonl",
                 "scores.json", "report.md"):
        assert (run_dir / name).is_file(), name


def test_bench_rescore_cli(tmp_path: Path, capsys) -> None:
    out = tmp_path / "runs"
    assert main(
        ["bench", "run", "--dataset", "locomo", "--retriever", "bm25",
         "--out", str(out), *_overrides()]
    ) == 0
    run_dir = _find_run_dir(out)
    capsys.readouterr()
    assert main(["bench", "rescore", str(run_dir), "--check"]) == 0
    assert "verified" in capsys.readouterr().out
    # Tamper -> check fails with rc 1.
    scores = json.loads((run_dir / "scores.json").read_text())
    scores[0]["value"] = 0.42424242
    (run_dir / "scores.json").write_text(
        json.dumps(scores, indent=2, sort_keys=True) + "\n"
    )
    assert main(["bench", "rescore", str(run_dir), "--check"]) == 1


def test_bench_gate_cli(tmp_path: Path, capsys) -> None:
    out = tmp_path / "runs"
    assert main(
        ["bench", "run", "--dataset", "locomo", "--retriever", "bm25",
         "--out", str(out), *_overrides()]
    ) == 0
    run_dir = _find_run_dir(out)
    capsys.readouterr()
    # Impossible floor -> rc 1 + GATE FAIL on stderr.
    rc = main(
        ["bench", "gate", str(run_dir),
         *_overrides("benchmark.ci_gate.min=1.0",
                     "benchmark.ci_gate.metric=precision@5")]
    )
    assert rc == 1
    assert "GATE FAIL" in capsys.readouterr().err
    # Trivial floor -> rc 0 + GATE PASS.
    rc = main(
        ["bench", "gate", str(run_dir),
         *_overrides("benchmark.ci_gate.min=0.0")]
    )
    assert rc == 0
    assert "GATE PASS" in capsys.readouterr().out


def test_bench_sample_cli(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "sample.json"
    rc = main(
        ["bench", "sample", "--dataset", "locomo",
         *_overrides(f"benchmark.sample.manifest={manifest}")]
    )
    assert rc == 0
    stdout = capsys.readouterr().out
    assert "eligible per category" in stdout
    data = json.loads(manifest.read_text())
    assert data["seed"] == 42 and len(data["ids"]) == 4


def test_bench_ingest_cli(tmp_path: Path) -> None:
    db_dir = tmp_path / "dbs"
    rc = main(
        ["bench", "ingest", "--dataset", "locomo", "--db-dir", str(db_dir),
         *_overrides("write.extraction_backend=none")]
    )
    assert rc == 0
    assert (db_dir / "conv-tiny-1.mind").is_file()


def test_bench_ingest_bad_conv_ids_manifest_rc1(tmp_path: Path, capsys) -> None:
    """A missing or malformed --conv-ids-from file exits 1 with a clean
    'gluomem bench:' error, never a raw traceback."""
    db_dir = tmp_path / "dbs"
    base = ["bench", "ingest", "--dataset", "locomo", "--db-dir", str(db_dir),
            *_overrides("write.extraction_backend=none")]

    # Missing file.
    rc = main([*base, "--conv-ids-from", str(tmp_path / "nope.json")])
    assert rc == 1
    assert "gluomem bench: cannot read conv-ids manifest" in capsys.readouterr().err

    # Malformed JSON.
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    rc = main([*base, "--conv-ids-from", str(bad)])
    assert rc == 1
    assert "gluomem bench: invalid conv-ids manifest" in capsys.readouterr().err

    # Valid JSON, wrong shape (no 'ids' key).
    wrong = tmp_path / "wrong.json"
    wrong.write_text('{"samples": []}', encoding="utf-8")
    rc = main([*base, "--conv-ids-from", str(wrong)])
    assert rc == 1
    assert "gluomem bench: invalid conv-ids manifest" in capsys.readouterr().err


def test_bench_ingest_system_mem0_cli(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import sys as _sys
    import types as _types

    from tests.bench.test_mem0 import FakeMem0Memory

    FakeMem0Memory.instances = []
    monkeypatch.setitem(
        _sys.modules, "mem0", _types.SimpleNamespace(Memory=FakeMem0Memory)
    )
    db_dir = tmp_path / "mem0_stores"
    rc = main(
        ["bench", "ingest", "--dataset", "locomo", "--system", "mem0",
         "--db-dir", str(db_dir), *_overrides()]
    )
    assert rc == 0
    assert (db_dir / "conv-tiny-1" / "ingest_meta.json").is_file()
    assert "(mem0)" in capsys.readouterr().out


def test_bench_ingest_mem0_missing_extra_rc1(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "mem0", None)
    rc = main(
        ["bench", "ingest", "--dataset", "locomo", "--system", "mem0",
         "--db-dir", str(tmp_path / "x"), *_overrides()]
    )
    assert rc == 1
    assert "uv sync --extra mem0" in capsys.readouterr().err


def test_bench_run_gluomem_without_db_dir_fails(tmp_path: Path, capsys) -> None:
    rc = main(
        ["bench", "run", "--dataset", "locomo", "--retriever", "gluomem",
         "--out", str(tmp_path), *_overrides()]
    )
    assert rc == 1
    assert "db-dir" in capsys.readouterr().err


def test_bench_run_sha_mismatch_fails(tmp_path: Path, capsys) -> None:
    # NB: an all-digit override would coerce to int; real sha256 hex always
    # carries letters.
    rc = main(
        ["bench", "run", "--dataset", "locomo", "--retriever", "bm25",
         "--out", str(tmp_path),
         *_overrides("benchmark.datasets.locomo.sha256=" + "ab" * 32)]
    )
    assert rc == 1
    assert "sha256 mismatch" in capsys.readouterr().err


def test_bench_fetch_cli_file_url(tmp_path: Path, capsys) -> None:
    import shutil

    from tests.bench.conftest import FIXTURE

    src = tmp_path / "upstream.json"
    shutil.copyfile(FIXTURE, src)
    target_dir = tmp_path / "data"
    rc = main(
        ["bench", "fetch", "--dataset", "locomo",
         *_overrides(
             f"benchmark.datasets.locomo.url={src.as_uri()}",
             f"benchmark.datasets_dir={target_dir}",
         )]
    )
    assert rc == 0
    assert "sha256" in capsys.readouterr().out
    # BENCH_OVERRIDES pins datasets.locomo.file to the fixture's name.
    assert (target_dir / FIXTURE.name).is_file()
