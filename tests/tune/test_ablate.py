"""Tests for deterministic signal and lane ablations (`gluomem/tune/ablate.py`)."""

from __future__ import annotations

import json
from pathlib import Path

from gluomem.cli import main
from gluomem.config.app_config import load_config
from gluomem.tune import (
    CandidateCache,
    CandidateRow,
    format_ablation_table,
    run_ablation,
)


def _ablation_cache() -> CandidateCache:
    rows = [
        # Doc A: only in FTS, critical importance + KG distance 0
        CandidateRow(
            doc_id="doc_fts_kg",
            fts_rank=0,
            vec_rank=None,
            days_since_created=10.0,
            access_count=5,
            bfs_distance=0,
            importance_level="critical",
            frame_id=1,
        ),
        # Doc B: only in Vec, normal importance
        CandidateRow(
            doc_id="doc_vec_only",
            fts_rank=None,
            vec_rank=0,
            days_since_created=2.0,
            access_count=1,
            bfs_distance=3,
            importance_level="normal",
            frame_id=2,
        ),
        # Distractor
        CandidateRow(
            doc_id="doc_noise",
            fts_rank=5,
            vec_rank=None,
            days_since_created=60.0,
            access_count=0,
            bfs_distance=4,
            importance_level="temporary",
            frame_id=3,
        ),
    ]
    return CandidateCache({"q1": rows}, qrels={"q1": {"doc_fts_kg": 1, "doc_vec_only": 1}})


def test_ablation_runner_and_cli(tmp_path: Path, capsys) -> None:
    cfg = load_config()
    cache = _ablation_cache()
    out_dir = tmp_path / "ablate_out"

    report = run_ablation(cache, cfg, k=2, out_dir=out_dir)
    by_name = {r.name: r for r in report.rows}

    assert set(by_name) == {
        "incumbent",
        "no_temporal",
        "no_popularity",
        "no_contextual",
        "no_importance",
        "hybrid_rrf",
        "fts_only",
        "vec_only",
    }
    # Hybrid retrieves both doc_fts_kg and doc_vec_only in top-2 (recall=1.0),
    # whereas fts_only and vec_only can each only retrieve 1 of the 2 (recall=0.5).
    assert by_name["incumbent"].recall == 1.0
    assert by_name["fts_only"].recall == 0.5
    assert by_name["vec_only"].recall == 0.5
    assert (out_dir / "ablation_report.json").is_file()

    table = format_ablation_table(report)
    assert "incumbent" in table
    assert "fts_only" in table

    cache_dir = tmp_path / "cache"
    cache.save(cache_dir)
    rc = main(["ablate", "--cache", str(cache_dir), "--k", "2", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["incumbent_recall"] == 1.0
