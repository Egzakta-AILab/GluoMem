"""CLI benchmark subcommands: run, rescore, gate, sample, ingest, fetch, replay."""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

from gluomem.bench.replay import NOT_OFFICIAL_LABEL, receipt_digest
from gluomem.cli import main
from tests.bench.conftest import BENCH_OVERRIDES, FAKE_BACKENDS, FIXTURE
from tests.bench.test_mem0 import FakeMem0Memory


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


def test_bench_run_rescore_and_gate_cli(tmp_path: Path, capsys) -> None:
    out = tmp_path / "runs"
    rc = main(
        [
            "bench",
            "run",
            "--dataset",
            "locomo",
            "--retriever",
            "bm25",
            "--out",
            str(out),
            *_overrides(),
        ]
    )
    assert rc == 0
    stdout = capsys.readouterr().out
    assert "qa_accuracy=" in stdout and "judge=trio_strict" in stdout
    run_dir = _find_run_dir(out)
    for name in (
        "manifest.json",
        "predictions.jsonl",
        "judgments.jsonl",
        "scores.json",
        "report.md",
    ):
        assert (run_dir / name).is_file(), name

    assert main(["bench", "rescore", str(run_dir), "--check"]) == 0
    assert "verified" in capsys.readouterr().out

    rc = main(
        [
            "bench",
            "gate",
            str(run_dir),
            *_overrides("benchmark.ci_gate.min=1.0", "benchmark.ci_gate.metric=precision@5"),
        ]
    )
    assert rc == 1 and "GATE FAIL" in capsys.readouterr().err

    rc = main(["bench", "gate", str(run_dir), *_overrides("benchmark.ci_gate.min=0.0")])
    assert rc == 0 and "GATE PASS" in capsys.readouterr().out

    scores = json.loads((run_dir / "scores.json").read_text())
    scores[0]["value"] = 0.42424242
    (run_dir / "scores.json").write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")
    assert main(["bench", "rescore", str(run_dir), "--check"]) == 1


def test_bench_sample_fetch_and_replay_cli(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "sample.json"
    assert (
        main(
            [
                "bench",
                "sample",
                "--dataset",
                "locomo",
                *_overrides(f"benchmark.sample.manifest={manifest}"),
            ]
        )
        == 0
    )
    assert "eligible per category" in capsys.readouterr().out
    data = json.loads(manifest.read_text())
    assert data["seed"] == 42 and len(data["ids"]) == 4

    src = tmp_path / "upstream.json"
    shutil.copyfile(FIXTURE, src)
    target_dir = tmp_path / "data"
    assert (
        main(
            [
                "bench",
                "fetch",
                "--dataset",
                "locomo",
                *_overrides(
                    f"benchmark.datasets.locomo.url={src.as_uri()}",
                    f"benchmark.datasets_dir={target_dir}",
                ),
            ]
        )
        == 0
    )
    assert "sha256" in capsys.readouterr().out
    assert (target_dir / FIXTURE.name).is_file()

    rc = main(
        [
            "bench",
            "replay",
            "--work-dir",
            str(tmp_path / "w"),
            "--out",
            str(tmp_path / "out"),
            "--override",
            "embedding.model=mock",
            "--override",
            "components.allow_fake=true",
            "--override",
            "search.rerank.enabled=false",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert NOT_OFFICIAL_LABEL in out
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text())
    assert receipt_digest(receipt) in out


def test_bench_ingest_and_error_paths_cli(tmp_path: Path, monkeypatch, capsys) -> None:
    db_dir = tmp_path / "dbs"
    base = [
        "bench",
        "ingest",
        "--dataset",
        "locomo",
        "--db-dir",
        str(db_dir),
        *_overrides("write.extraction_backend=none"),
    ]
    assert main(base) == 0
    assert (db_dir / "conv-tiny-1.mind").is_file()

    assert main([*base, "--conv-ids-from", str(tmp_path / "nope.json")]) == 1
    assert "gluomem bench: cannot read conv-ids manifest" in capsys.readouterr().err

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert main([*base, "--conv-ids-from", str(bad)]) == 1
    assert "gluomem bench: invalid conv-ids manifest" in capsys.readouterr().err

    wrong = tmp_path / "wrong.json"
    wrong.write_text('{"samples": []}', encoding="utf-8")
    assert main([*base, "--conv-ids-from", str(wrong)]) == 1
    assert "gluomem bench: invalid conv-ids manifest" in capsys.readouterr().err

    FakeMem0Memory.instances = []
    monkeypatch.setitem(sys.modules, "mem0", types.SimpleNamespace(Memory=FakeMem0Memory))
    mem0_dir = tmp_path / "mem0_stores"
    assert (
        main(
            [
                "bench",
                "ingest",
                "--dataset",
                "locomo",
                "--system",
                "mem0",
                "--db-dir",
                str(mem0_dir),
                *_overrides(),
            ]
        )
        == 0
    )
    assert (mem0_dir / "conv-tiny-1" / "ingest_meta.json").is_file()
    capsys.readouterr()

    monkeypatch.setitem(sys.modules, "mem0", None)
    assert (
        main(
            [
                "bench",
                "ingest",
                "--dataset",
                "locomo",
                "--system",
                "mem0",
                "--db-dir",
                str(tmp_path / "x"),
                *_overrides(),
            ]
        )
        == 1
    )
    assert "uv sync --extra mem0" in capsys.readouterr().err

    assert (
        main(
            [
                "bench",
                "run",
                "--dataset",
                "locomo",
                "--retriever",
                "gluomem",
                "--out",
                str(tmp_path),
                *_overrides(),
            ]
        )
        == 1
    )
    assert "db-dir" in capsys.readouterr().err

    assert (
        main(
            [
                "bench",
                "run",
                "--dataset",
                "locomo",
                "--retriever",
                "bm25",
                "--out",
                str(tmp_path),
                *_overrides("benchmark.datasets.locomo.sha256=" + "ab" * 32),
            ]
        )
        == 1
    )
    assert "sha256 mismatch" in capsys.readouterr().err
