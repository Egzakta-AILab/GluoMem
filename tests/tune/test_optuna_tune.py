"""Tests for Optuna Tier-1 parameter optimization and `gluomem tune` CLI."""

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
from gluomem.tune.objective import split_queries


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
                frame_id=i * 10 + 1,
            ),
            # Gold has vector rank 0 and FTS rank 1, 60 days old (so heavy temporal
            # weight hurts it), but critical importance + KG distance 0 + high popularity.
            CandidateRow(
                doc_id=gold_id,
                fts_rank=1,
                vec_rank=0,
                days_since_created=60.0,
                access_count=25,
                bfs_distance=0,
                importance_level="critical",
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
                frame_id=i * 10 + 3,
            ),
        ]
        qrels[qid] = {gold_id: 1}
    return CandidateCache(queries, qrels=qrels)


def test_optuna_tune_never_regresses_on_held_out(tmp_path: Path) -> None:
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

    assert len(result.trials) == 200
    assert result.trials[0].is_incumbent is True
    assert result.best_score >= result.incumbent_score
    assert result.best_score == 1.0
    assert result.incumbent_score < 1.0

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_calls"] == 0
    assert manifest["n_trials"] == 200
    assert manifest["best_score"] >= manifest["incumbent_score"]

    best_params = json.loads((out_dir / "best_params.json").read_text(encoding="utf-8"))
    overrides = best_params["config_overrides"]
    tuned_cfg = cfg.with_overrides(overrides)
    assert (
        abs(sum(tuned_cfg.scoring.profiles["balanced"].weights.model_dump().values()) - 1.0) < 1e-6
    )

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


def test_same_seed_produces_identical_trials_artifacts_and_split(tmp_path: Path) -> None:
    cache = _synthetic_cache()
    kwargs = TuneConfig(n_trials=30, seed=19, k=1)
    first = run_optuna_tune(cache, load_config(), tune_cfg=kwargs, out_dir=tmp_path / "first")
    second = run_optuna_tune(cache, load_config(), tune_cfg=kwargs, out_dir=tmp_path / "second")
    assert first.best_params == second.best_params
    assert first.trials == second.trials
    assert first.to_manifest_dict() == second.to_manifest_dict()
    assert set(first.development_query_ids).isdisjoint(first.held_out_query_ids)
    for filename in ("manifest.json", "best_params.json", "trials.jsonl"):
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()


def test_dev_winner_is_rejected_when_it_regresses_on_held_out() -> None:
    source = _synthetic_cache()
    dev, held_out = split_queries(source.query_ids(), 0.3, 42)
    cache = CandidateCache(
        {qid: source.rows(qid) for qid in source.query_ids()},
        qrels={
            qid: {f"{'gold' if qid in dev else 'fresh'}_{qid.split('_')[1]}": 1}
            for qid in source.query_ids()
        },
    )
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
    result = run_optuna_tune(cache, cfg, tune_cfg=TuneConfig(n_trials=50, k=1))
    assert result.best_dev_score > result.incumbent_dev_score
    assert result.held_out_query_ids == tuple(held_out)
    assert not result.promoted
    assert result.best_params == result.incumbent_params
    assert result.best_score == result.incumbent_score == 1.0


def test_tuning_refuses_an_empty_held_out_set() -> None:
    source = _synthetic_cache()
    cache = CandidateCache({"q_0": source.rows("q_0")}, qrels={"q_0": source.qrel("q_0")})
    with pytest.raises(ValueError, match="at least two"):
        run_optuna_tune(cache, load_config())
