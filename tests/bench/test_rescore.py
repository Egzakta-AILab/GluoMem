"""Rescore — golden regeneration, tamper detection, dspy-free subprocess."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gluomem.bench.rescore import RescoreMismatchError, rescore_run
from gluomem.bench.runner import run_benchmark
from gluomem.config.app_config import AppConfig
from tests.bench.test_runner import FAKE_BACKENDS


@pytest.fixture
def run_paths(bench_cfg: AppConfig, tmp_path: Path):
    cfg = bench_cfg.with_overrides(dict(FAKE_BACKENDS))
    return run_benchmark(
        cfg, dataset="locomo", retriever="bm25", out_root=tmp_path / "runs"
    )


def test_rescore_regenerates_byte_identical(run_paths) -> None:
    committed_scores = run_paths.scores.read_bytes()
    committed_report = run_paths.report.read_bytes()
    run_paths.scores.unlink()
    run_paths.report.unlink()
    entries = rescore_run(run_paths.root)
    assert run_paths.scores.read_bytes() == committed_scores
    assert run_paths.report.read_bytes() == committed_report
    assert entries  # and returns the entries it derived


def test_rescore_check_passes_on_intact_run(run_paths) -> None:
    rescore_run(run_paths.root, check=True)  # no raise


def test_rescore_check_catches_tampered_judgment(run_paths) -> None:
    lines = run_paths.judgments.read_text().splitlines()
    row = json.loads(lines[0])
    row["correct"] = not row["correct"]
    lines[0] = json.dumps(row, sort_keys=True)
    run_paths.judgments.write_text("\n".join(lines) + "\n")
    with pytest.raises(RescoreMismatchError, match="tampered|diverge"):
        rescore_run(run_paths.root, check=True)


def test_rescore_check_catches_tampered_scores(run_paths) -> None:
    entries = json.loads(run_paths.scores.read_text())
    entries[0]["value"] = 0.123456
    run_paths.scores.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    with pytest.raises(RescoreMismatchError):
        rescore_run(run_paths.root, check=True)


def test_rescore_needs_no_dataset_file(run_paths, tmp_path: Path) -> None:
    """Run dirs are self-contained: rescore works with the dataset gone."""
    import shutil

    moved = tmp_path / "moved_run"
    shutil.copytree(run_paths.root, moved)
    entries = rescore_run(moved)  # fixture path never touched
    assert entries


def test_rescore_subprocess_never_imports_dspy(run_paths) -> None:
    code = (
        "import sys\n"
        "from gluomem.bench.rescore import rescore_run\n"
        f"rescore_run({str(run_paths.root)!r}, check=True)\n"
        "sys.exit(1 if 'dspy' in sys.modules else 0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).parents[2]),
    )
    assert proc.returncode == 0, proc.stderr


def test_rescore_names_file_and_line_on_malformed_predictions(run_paths) -> None:
    """A corrupt predictions.jsonl line fails with the file path + line number,
    not a bare JSONDecodeError with no provenance."""
    lines = run_paths.predictions.read_text().splitlines()
    lines[0] = "{corrupt"
    run_paths.predictions.write_text("\n".join(lines) + "\n")
    with pytest.raises(RescoreMismatchError, match=r"predictions\.jsonl line 1"):
        rescore_run(run_paths.root, check=True)


def test_rescore_names_file_and_line_on_malformed_judgments(run_paths) -> None:
    lines = run_paths.judgments.read_text().splitlines()
    # Valid JSON, wrong shape: JudgeRecord(**row) raises TypeError.
    lines[1] = '{"unexpected_field": true}'
    run_paths.judgments.write_text("\n".join(lines) + "\n")
    with pytest.raises(RescoreMismatchError, match=r"judgments\.jsonl line 2"):
        rescore_run(run_paths.root, check=True)
