"""Tests for Optuna Tier-1 parameter optimization and `gluomem tune` CLI (`KVAD-1352`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gluomem import extras
from gluomem.cli import main
from gluomem.config.app_config import load_config
from gluomem.tune import (
    CandidateCache,
    CandidateRow,
    TuneConfig,
    run_optuna_tune,
)


def _synthetic_cache() -> CandidateCache:
    """Synthetic cache where contextual + importance signals identify the gold doc
    while temporal recency favors a fresh distractor."""
    queries: dict[str, list[CandidateRow]] = {}
    qrels: dict[str, dict[str, int]] = {}
    for i in range(8):
        qid = f"q_{i}"
        gold_id = f"gold_{i}"
        fresh_distractor = f"fresh_{i}"
        stale_distractor = f"stale_{i}"
        queries[qid] = [
            # Fresh distractor ranked slightly higher in raw FTS (rank 0), recent (1 day),
            # but temporary importance and disconnected in KG (bfs_distance=4).
            CandidateRow(
                doc_id=fresh_distractor,
                fts_rank=0,
                vec_rank=1,
                days_since_created=1.0,
                access_count=0,
                bfs_distance=4,
                importance_level="temporary",
                grade=0,
                frame_id=i * 10 + 1,
            ),
            # Gold doc ranked #1 in Vec and #1 in FTS, 60 days old (so heavy temporal
            # weight hurts it), but critical importance + KG distance 0 + high popularity.
            CandidateRow(
                doc_id=gold_id,
                fts_rank=1,
                vec_rank=0,
                days_since_created=60.0,
                access_count=25,
                bfs_distance=0,
                importance_level="critical",
                grade=1,
                frame_id=i * 10 + 2,
            ),
            CandidateRow(
                doc_id=stale_distractor,
                fts_rank=2,
                vec_rank=2,
                days_since_created=90.0,
                access_count=0,
                bfs_distance=4,
                importance_level="deprecated",
                grade=0,
                frame_id=i * 10 + 3,
            ),
        ]
        qrels[qid] = {gold_id: 1}
    return CandidateCache(queries, qrels=qrels)


def test_optuna_tune_200_trials_under_5s_and_never_regresses(tmp_path: Path) -> None:
    # Start with a temporal-heavy profile that ranks fresh_distractor above gold_id at k=1
    cfg = load_config().with_overrides(
        {
            "scoring.profiles.balanced.weights.temporal": 0.85,
            "scoring.profiles.balanced.weights.popularity": 0.05,
            "scoring.profiles.balanced.weights.contextual": 0.05,
            "scoring.profiles.balanced.weights.importance": 0.05,
            "scoring.half_life_days": 7,
            "scoring.flat_boost_days": 2,
        }
    )
    cache = _synthetic_cache()
    out_dir = tmp_path / "tune_out"

    result = run_optuna_tune(
        cache,
        cfg,
        tune_cfg=TuneConfig(
            n_trials=200,
            k=1,
            metric="ndcg",
            profile="balanced",
            seed=42,
        ),
        out_dir=out_dir,
    )

    assert result.elapsed_ms < 5000.0
    assert len(result.trials) == 200
    assert result.trials[0].is_incumbent is True
    assert result.best_score >= result.incumbent_score
    assert result.best_score == 1.0
    assert result.incumbent_score < 1.0

    # Verify artifacts written to out_dir
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_calls"] == 0
    assert manifest["n_trials"] == 200
    assert manifest["best_score"] >= manifest["incumbent_score"]

    best_params = json.loads((out_dir / "best_params.json").read_text(encoding="utf-8"))
    overrides = best_params["config_overrides"]
    tuned_cfg = cfg.with_overrides(overrides)
    assert abs(
        sum(tuned_cfg.scoring.profiles["balanced"].weights.model_dump().values()) - 1.0
    ) < 1e-6

    trials_lines = (out_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(trials_lines) == 200


def test_cli_tune_end_to_end_and_missing_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = _synthetic_cache()
    cache_dir = tmp_path / "cache"
    cache.save(cache_dir)
    out_dir = tmp_path / "cli_tune_out"

    rc = main(
        [
            "tune",
            "--cache",
            str(cache_dir),
            "--out",
            str(out_dir),
            "--trials",
            "25",
            "--k",
            "2",
            "--seed",
            "7",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "tuned 25 trial(s)" in out
    assert "model_calls=0" in out
    assert (out_dir / "best_params.json").is_file()

    # Simulate missing `tune` extra (optuna not installed)
    monkeypatch.setattr(
        extras,
        "_missing",
        lambda extra: ["optuna"] if extra == "tune" else [],
    )
    rc_missing = main(["tune", "--cache", str(cache_dir), "--trials", "5"])
    assert rc_missing == 2
    err = capsys.readouterr().err
    assert "requires the 'tune' extra" in err
    assert "gluomem[tune]" in err
