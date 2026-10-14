"""Benchmark runner, per-conversation ingest, report rendering, CI gate, and live smoke."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import dotenv
import pytest

from gluomem.bench.report import check_gate, render_report
from gluomem.bench.runner import (
    _make_answer_fn,
    _make_judge_fns,
    ingest_per_conversation,
    run_benchmark,
    verify_ingest_completeness,
)
from gluomem.config.app_config import AppConfig, load_config
from gluomem.embedding.reranker import StubReranker
from gluomem.llm import NullExtractors
from gluomem.llm.outputs import AnswerResult, Verdict
from gluomem.storage.db import MindDB
from tests.bench.conftest import BENCH_OVERRIDES, FAKE_BACKENDS

dotenv.load_dotenv(Path(__file__).parents[2] / ".env", override=False)

SHA = "d" * 64
FIXTURE_2CONV = Path(__file__).parent.parent / "fixtures" / "locomo_tiny_2conv.json"


@pytest.fixture
def run_cfg(bench_cfg: AppConfig) -> AppConfig:
    return bench_cfg.with_overrides(dict(FAKE_BACKENDS))


def _run(cfg: AppConfig, tmp_path: Path, retriever: str = "bm25", **kw: object):
    return run_benchmark(
        cfg,
        dataset="locomo",
        retriever=retriever,
        out_root=tmp_path / "runs",
        **kw,  # type: ignore[arg-type]
    )


def _cfg_2conv(bench_cfg: AppConfig) -> AppConfig:
    sha = hashlib.sha256(FIXTURE_2CONV.read_bytes()).hexdigest()
    return bench_cfg.with_overrides(
        {
            "benchmark.datasets.locomo.file": FIXTURE_2CONV.name,
            "benchmark.datasets.locomo.sha256": sha,
            "write.extraction_backend": "none",
        }
    )


def _frame_dump(db_path: Path, cfg: AppConfig) -> list[tuple]:
    db = MindDB(db_path, dim=cfg.storage.vector_dim)
    try:
        return db.conn.execute(
            "SELECT frame_type, gop_id, content, event_time, content_hash "
            "FROM memory_frames ORDER BY gop_id, content, content_hash"
        ).fetchall()
    finally:
        db.close()


def test_live_backends_preflight_api_key_and_retriever_errors(
    run_cfg: AppConfig, bench_cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live_cfg = bench_cfg.with_overrides(
        {"benchmark.answer.backend": "dspy", "benchmark.judge.backend": "dspy"}
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        _make_answer_fn(live_cfg)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        _make_judge_fns(live_cfg)

    with pytest.raises(ValueError, match="db-dir"):
        _run(run_cfg, tmp_path, retriever="gluomem")
    with pytest.raises(ValueError, match="full_context"):
        _run(run_cfg, tmp_path, retriever="chatgpt")


def test_run_benchmark_artifacts_scores_manifest_and_limit(
    run_cfg: AppConfig, tmp_path: Path
) -> None:
    paths = _run(run_cfg, tmp_path)
    for p in (
        paths.manifest,
        paths.predictions,
        paths.judgments,
        paths.scores,
        paths.denominators,
        paths.report,
    ):
        assert p.is_file(), p
    assert paths.root.name.endswith("_locomo_bm25")

    pred_rows = [json.loads(line) for line in paths.predictions.read_text().splitlines()]
    assert len(pred_rows) == 4
    for row in pred_rows:
        for key in (
            "sample_id", "category", "category_name", "question", "gold",
            "evidence", "answer", "abstained", "retrieved", "abstention",
        ):
            assert key in row, key

    judg_rows = [json.loads(line) for line in paths.judgments.read_text().splitlines()]
    assert len(judg_rows) == 4 * 3
    for i, pred in enumerate(pred_rows):
        chunk = judg_rows[i * 3 : (i + 1) * 3]
        assert {r["sample_id"] for r in chunk} == {pred["sample_id"]}
        assert [r["backbone"] for r in chunk] == ["gpt5-mini", "haiku", "qwen3"]

    entries = json.loads(paths.scores.read_text())
    overall = [e for e in entries if e["scope"] == "overall"]
    families = {
        (e["tag"]["metric_family"], e["tag"]["judge_family"], e["tag"]["backbone"])
        for e in overall
    }
    assert ("qa_accuracy", "trio_strict", "gpt5-mini+haiku+qwen3") in families
    assert ("qa_accuracy", "single", "haiku") in families
    assert ("precision@5", "none", "none") in families
    assert ("evidence_hit@10", "none", "none") in families
    assert len(overall) == len(families)

    manifest = json.loads(paths.manifest.read_text())
    assert manifest["dataset"] == "locomo" and manifest["retriever"] == "bm25"
    assert manifest["judge"]["backbones"] == ["gpt5-mini", "haiku", "qwen3"]
    assert manifest["judge"]["prompt_sha"] is None
    assert manifest["usage"]["judge_calls"] == 12
    assert manifest["timing"]["parallelism"] == run_cfg.benchmark.parallelism

    lim_paths = _run(run_cfg, tmp_path / "lim", limit=2)
    assert len(lim_paths.predictions.read_text().splitlines()) == 2
    assert json.loads(lim_paths.manifest.read_text())["limit"] == 2


def test_parallelism_single_family_full_context_and_gluomem_e2e(
    run_cfg: AppConfig, ingested_db_dir: Path, tmp_path: Path
) -> None:
    seq = _run(run_cfg.with_overrides({"benchmark.parallelism": 1}), tmp_path / "a")
    par = _run(run_cfg.with_overrides({"benchmark.parallelism": 8}), tmp_path / "b")
    for name in ("judgments", "scores"):
        assert getattr(seq, name).read_bytes() == getattr(par, name).read_bytes()

    single_paths = _run(
        run_cfg.with_overrides(
            {"benchmark.judge.family": "single", "benchmark.judge.backbones": ["haiku"]}
        ),
        tmp_path / "single",
    )
    single_entries = json.loads(single_paths.scores.read_text())
    assert not any(e["tag"]["judge_family"] == "trio_strict" for e in single_entries)

    fc_paths = _run(run_cfg, tmp_path / "fc", retriever="full_context")
    fc_denom = json.loads(fc_paths.denominators.read_text())
    assert fc_denom["retrieval"]["not_applicable"] == {"retriever_without_retrieval": 4}
    assert json.loads(fc_paths.manifest.read_text())["retrieves"] is False

    gm_paths = _run(
        run_cfg.with_overrides({"search.rerank.enabled": True}),
        tmp_path / "gm",
        retriever="gluomem",
        db_dir=ingested_db_dir,
        reranker=StubReranker(logit=2.0),
    )
    gm_manifest = json.loads(gm_paths.manifest.read_text())
    info = gm_manifest["retriever_info"]
    assert info["rerank"]["active"] is True
    assert gm_manifest["config"]["search"]["abstain"]["enabled"] is False


def test_answer_timeout_and_judge_failures_degrade_gracefully(
    run_cfg: AppConfig, tmp_path: Path, capsys
) -> None:
    def flaky_answer(question: str, context: str) -> AnswerResult:
        if "career" in question:
            raise RuntimeError("adapter exploded")
        return AnswerResult(text=context)

    paths = _run(run_cfg, tmp_path / "err", answer_fn=flaky_answer)
    assert "answer failed" in capsys.readouterr().err
    d = json.loads(paths.denominators.read_text())
    assert d["predictions"]["failed"] == {"error": 1}
    assert d["retrieval"]["scored"] == 4

    def slow_answer(question: str, context: str) -> AnswerResult:
        if "career" in question:
            raise TimeoutError("reader timed out")
        return AnswerResult(text=context)

    t_paths = _run(run_cfg, tmp_path / "to", answer_fn=slow_answer)
    assert json.loads(t_paths.denominators.read_text())["predictions"]["failed"] == {"timeout": 1}
    assert "timeout=1" in t_paths.report.read_text()

    calls = {"n": 0}

    def flaky_judge(question: str, gold: str, prediction: str) -> Verdict:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("judge exploded")
        return Verdict(correct=True, score=1.0, rationale="r")

    j_paths = run_benchmark(
        run_cfg.with_overrides({"benchmark.parallelism": 1}),
        dataset="locomo",
        retriever="bm25",
        out_root=tmp_path / "jerr",
        judge_fns={b: flaky_judge for b in ("haiku", "gpt5-mini", "qwen3")},
    )
    assert len(j_paths.judgments.read_text().splitlines()) == 4 * 3 - 1
    assert len(json.loads(j_paths.manifest.read_text())["failures"]["judge"]) == 1


def test_ingest_per_conversation_sentinels_and_verify(
    bench_cfg: AppConfig, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg_2conv(bench_cfg)
    seq, par = tmp_path / "seq", tmp_path / "par"
    ingest_per_conversation(
        cfg.with_overrides({"write.conv_parallelism": 1}),
        dataset="locomo",
        db_dir=seq,
        facade=NullExtractors(),
    )
    reports = ingest_per_conversation(
        cfg.with_overrides({"write.conv_parallelism": 4}),
        dataset="locomo",
        db_dir=par,
        facade=NullExtractors(),
    )
    assert [r.turns for r in reports] == [4, 3]
    for conv_id in ("conv-tiny-a", "conv-tiny-b"):
        assert _frame_dump(seq / f"{conv_id}.mind", cfg) == _frame_dump(
            par / f"{conv_id}.mind", cfg
        )

    single_cfg = bench_cfg.with_overrides({"write.extraction_backend": "none"})
    db_dir = tmp_path / "dbs"
    ingest_per_conversation(single_cfg, dataset="locomo", db_dir=db_dir, facade=NullExtractors())
    sentinel = db_dir / "conv-tiny-1.ingest.json"
    assert sentinel.is_file()

    capsys.readouterr()
    assert (
        ingest_per_conversation(
            single_cfg, dataset="locomo", db_dir=db_dir, facade=NullExtractors(), skip_existing=True
        )
        == []
    )
    assert "skipped conv-tiny-1" in capsys.readouterr().out

    rows = verify_ingest_completeness(single_cfg, dataset="locomo", db_dir=db_dir)
    assert [r["status"] for r in rows] == ["complete"]

    sentinel.unlink()
    with pytest.raises(RuntimeError, match="without"):
        ingest_per_conversation(
            single_cfg, dataset="locomo", db_dir=db_dir, facade=NullExtractors(), skip_existing=True
        )
    verify_ingest_completeness(single_cfg, dataset="locomo", db_dir=db_dir, write_sentinels=True)
    assert sentinel.is_file()

    con = sqlite3.connect(db_dir / "conv-tiny-1.mind")
    con.execute(
        "DELETE FROM memory_frames WHERE gop_id LIKE 'conv-tiny-1:session:%' "
        "AND gop_id = (SELECT gop_id FROM memory_frames "
        "WHERE gop_id LIKE 'conv-tiny-1:session:%' LIMIT 1)"
    )
    con.commit()
    con.close()
    rows = verify_ingest_completeness(single_cfg, dataset="locomo", db_dir=db_dir)
    assert rows[0]["status"] == "incomplete"
    assert [
        r["status"]
        for r in verify_ingest_completeness(single_cfg, dataset="locomo", db_dir=tmp_path / "empty")
    ] == ["absent"]


def test_run_stays_dspy_free_on_fake_backends(tmp_path: Path) -> None:
    overrides = dict(BENCH_OVERRIDES) | dict(FAKE_BACKENDS)
    code = f"""
import sys
from gluomem.config.app_config import load_config
from gluomem.bench.runner import run_benchmark
cfg = load_config(overrides={overrides!r})
run_benchmark(
    cfg, dataset="locomo", retriever="bm25", out_root={str(tmp_path / "runs")!r}
)
sys.exit(1 if "dspy" in sys.modules else 0)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).parents[2]),
    )
    assert proc.returncode == 0, proc.stderr


def _entry(
    scope: str,
    value: float,
    metric_family: str,
    judge_family: str = "none",
    backbone: str = "none",
    n: int = 4,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "value": value,
        "n": n,
        "tag": {
            "metric_family": metric_family,
            "judge_family": judge_family,
            "judge_count": 3 if judge_family == "trio_strict" else 0,
            "backbone": backbone,
            "dataset_sha": SHA,
        },
    }


def test_report_rendering_and_ci_gate() -> None:
    entries = [
        _entry("overall", 0.75, "qa_accuracy", "trio_strict", "gpt5-mini+haiku+qwen3"),
        _entry("category:temporal", 0.5, "qa_accuracy", "trio_strict", "gpt5-mini+haiku+qwen3"),
        _entry("overall", 1.0, "qa_accuracy", "single", "haiku"),
        _entry("overall", 0.4, "precision@5"),
        _entry("overall", 0.9, "evidence_hit@5"),
    ]
    manifest = {
        "run_id": "20260705T000000_locomo_bm25",
        "dataset": "locomo",
        "dataset_sha": SHA,
        "retriever": "bm25",
        "answer": {"model": "deepseek-flash", "model_slug": "x/y", "backend": "fake"},
        "judge": {
            "family": "trio_strict",
            "backbones": ["gpt5-mini", "haiku", "qwen3"],
            "backend": "fake",
        },
        "sample_ids": ["a", "b", "c", "d"],
        "sample_source": "in-memory",
        "sample_seed": 42,
        "created_at": "2026-07-05T00:00:00+00:00",
        "git_sha": "abc",
    }
    report = render_report(entries, manifest)
    sections = re.split(r"^## ", report, flags=re.MULTILINE)[1:]
    tables = [s for s in sections if "| scope |" in s]
    assert len(tables) == 4
    assert "Mem0" in report and "hive-mind" in report
    assert "| " not in report.split("Different-protocol citations")[1]

    cfg = load_config()
    ok, entry = check_gate(entries, cfg.benchmark.ci_gate)
    assert ok is True and entry["value"] == 0.9
    strict_gate = cfg.with_overrides({"benchmark.ci_gate.min": 0.95}).benchmark.ci_gate
    assert check_gate(entries, strict_gate)[0] is False
    ok_qa, qa_entry = check_gate(
        entries, cfg.with_overrides({"benchmark.ci_gate.metric": "qa_accuracy"}).benchmark.ci_gate
    )
    assert ok_qa is True and qa_entry["tag"]["judge_family"] == "trio_strict"
    missing_gate = cfg.with_overrides(
        {"benchmark.ci_gate.metric": "evidence_recall@10"}
    ).benchmark.ci_gate
    with pytest.raises(ValueError, match="never computed"):
        check_gate(entries, missing_gate)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
def test_live_answerer_and_trio_judges() -> None:
    from gluomem.bench.judge import SingleJudge
    from gluomem.bench.metrics import qa_scores_from_records
    from gluomem.bench.types import JudgeRecord, Prediction, Sample
    from gluomem.llm.answer import Answerer
    from gluomem.llm.judge import QAJudge
    from gluomem.llm.router import LMRouter

    cfg = load_config()
    answerer = Answerer(cfg)
    result = answerer(
        "Where does Bob live now?",
        "[2023-05-08] Bob: I live in Boston now.\n"
        "[2023-08-02] Bob: Update from me: I moved to Seattle last week.",
    )
    assert "seattle" in result.text.lower()

    router = LMRouter(cfg)
    records: list[JudgeRecord] = []
    judges = [
        SingleJudge(
            b,
            QAJudge(router, b).judge,
            model_slug=cfg.models[b].provider_model,
            dataset_sha="live-smoke",
            sink=records.append,
        )
        for b in cfg.benchmark.judge.backbones
    ]
    sample = Sample(
        id="live:qa:0",
        conv_id="live",
        question="What pet did Alice adopt?",
        gold="a cat",
        category=4,
        category_name="single-hop",
    )
    for j in judges:
        j.score(sample, Prediction(sample_id=sample.id, answer="a cat"))
    trio, _ = qa_scores_from_records(
        records, backbones=cfg.benchmark.judge.backbones, dataset_sha="live-smoke"
    )
    assert trio[sample.id].value == 1.0
