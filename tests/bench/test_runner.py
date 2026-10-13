"""Runner — offline e2e over the fixture with fake backends."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gluomem.bench.runner import ingest_per_conversation, run_benchmark
from gluomem.config.app_config import AppConfig

FAKE_BACKENDS: dict[str, object] = {
    "benchmark.answer.backend": "fake",
    "benchmark.judge.backend": "fake",
    # No committed manifest in tests: 1 per category over the tiny fixture.
    "benchmark.sample.per_category": 1,
    "benchmark.sample.manifest": "does-not-exist.json",
}


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


def test_live_backends_preflight_api_key(
    bench_cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-fake answer/judge factories fail loud (naming the env var) when the
    key is missing — before any LLM object is built or ingest work is spent."""
    from gluomem.bench.runner import _make_answer_fn, _make_judge_fns

    # Config first: loading config re-reads the project .env into os.environ,
    # so the delenv must come after.
    live_cfg = bench_cfg.with_overrides(
        {"benchmark.answer.backend": "dspy", "benchmark.judge.backend": "dspy"}
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        _make_answer_fn(live_cfg)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        _make_judge_fns(live_cfg)


def test_run_dir_is_complete(run_cfg: AppConfig, tmp_path: Path) -> None:
    paths = _run(run_cfg, tmp_path)
    for p in (paths.manifest, paths.predictions, paths.judgments, paths.scores):
        assert p.is_file(), p
    assert paths.root.name.endswith("_locomo_bm25")


def test_predictions_are_self_contained(run_cfg: AppConfig, tmp_path: Path) -> None:
    paths = _run(run_cfg, tmp_path)
    rows = [
        json.loads(line)
        for line in paths.predictions.read_text().splitlines()
    ]
    assert len(rows) == 4  # 1 per category x 4
    for row in rows:
        # Everything rescore needs is embedded.
        for key in (
            "sample_id", "category", "category_name", "question", "gold",
            "evidence", "answer", "abstained", "retrieved", "abstention",
        ):
            assert key in row, key


def test_judgments_cover_all_backbones(run_cfg: AppConfig, tmp_path: Path) -> None:
    paths = _run(run_cfg, tmp_path)
    rows = [json.loads(line) for line in paths.judgments.read_text().splitlines()]
    assert len(rows) == 4 * 3
    # Rows land grouped per sample, backbones in sorted order within a group,
    # samples in prediction order — deterministic regardless of parallelism.
    predictions = [
        json.loads(line)["sample_id"]
        for line in paths.predictions.read_text().splitlines()
    ]
    for i, sample_id in enumerate(predictions):
        chunk = rows[i * 3 : (i + 1) * 3]
        assert {r["sample_id"] for r in chunk} == {sample_id}
        assert [r["backbone"] for r in chunk] == ["gpt5-mini", "haiku", "qwen3"]


def test_scores_families_never_merged(run_cfg: AppConfig, tmp_path: Path) -> None:
    paths = _run(run_cfg, tmp_path)
    entries = json.loads(paths.scores.read_text())
    overall = [e for e in entries if e["scope"] == "overall"]
    families = {
        (e["tag"]["metric_family"], e["tag"]["judge_family"], e["tag"]["backbone"])
        for e in overall
    }
    # trio + 3 singles + 6 retrieval families (3 metrics x ks [5, 10]).
    assert ("qa_accuracy", "trio_strict", "gpt5-mini+haiku+qwen3") in families
    assert ("qa_accuracy", "single", "haiku") in families
    assert ("precision@5", "none", "none") in families
    assert ("evidence_hit@10", "none", "none") in families
    assert len(overall) == len(families)  # one entry per family — never merged


def test_manifest_records_protocol(run_cfg: AppConfig, tmp_path: Path) -> None:
    paths = _run(run_cfg, tmp_path)
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["dataset"] == "locomo"
    assert manifest["retriever"] == "bm25"
    assert manifest["judge"]["backbones"] == ["gpt5-mini", "haiku", "qwen3"]
    assert manifest["judge"]["backend"] == "fake"
    assert manifest["judge"]["prompt_sha"] is None  # fake path: no dspy import
    assert manifest["answer"]["model"] == "deepseek-flash"
    assert len(manifest["sample_ids"]) == 4
    assert manifest["config"]["benchmark"]["sample"]["seed"] == 42
    assert manifest["usage"]["judge_calls"] == 12


def test_limit_truncates(run_cfg: AppConfig, tmp_path: Path) -> None:
    paths = _run(run_cfg, tmp_path, limit=2)
    rows = paths.predictions.read_text().splitlines()
    assert len(rows) == 2
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["limit"] == 2


def test_parallelism_identical_artifacts(run_cfg: AppConfig, tmp_path: Path) -> None:
    """Artifacts identical at parallelism 1 vs 8 (latency is per-run timing,
    the only field allowed to differ)."""
    seq = _run(run_cfg.with_overrides({"benchmark.parallelism": 1}), tmp_path / "a")
    par = _run(run_cfg.with_overrides({"benchmark.parallelism": 8}), tmp_path / "b")
    for name in ("judgments", "scores"):
        assert getattr(seq, name).read_bytes() == getattr(par, name).read_bytes(), name

    def rows_sans_latency(path: Path) -> list[dict]:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row.pop("latency_ms")
        return rows

    assert rows_sans_latency(seq.predictions) == rows_sans_latency(par.predictions)


def test_gluomem_retriever_requires_db_dir(run_cfg: AppConfig, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="db-dir"):
        _run(run_cfg, tmp_path, retriever="gluomem")


def test_unknown_retriever_lists_known(run_cfg: AppConfig, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="full_context"):
        _run(run_cfg, tmp_path, retriever="chatgpt")


def test_end_to_end_gluomem_retriever(
    run_cfg: AppConfig, ingested_db_dir: Path, tmp_path: Path
) -> None:
    """Full pipeline over the ingested per-conversation DB: the fake answerer
    echoes context, so fake-judge accuracy == did retrieval surface gold."""
    paths = _run(run_cfg, tmp_path, retriever="gluomem", db_dir=ingested_db_dir)
    entries = json.loads(paths.scores.read_text())
    trio = [
        e
        for e in entries
        if e["scope"] == "overall" and e["tag"]["judge_family"] == "trio_strict"
    ]
    assert len(trio) == 1
    assert 0.0 <= trio[0]["value"] <= 1.0
    assert trio[0]["n"] == 4


def test_gluomem_manifest_records_effective_rerank_and_forces_abstain_off(
    run_cfg: AppConfig, ingested_db_dir: Path, tmp_path: Path
) -> None:
    """The gluomem run must (a) surface effective retrieval provenance in
    ``retriever_info`` (guards the silent-rerank-off regression) and (b) record
    abstain disabled — the recorded config matches what actually ran."""
    from gluomem.embedding.reranker import StubReranker

    cfg = run_cfg.with_overrides({"search.rerank.enabled": True})
    paths = _run(
        cfg,
        tmp_path,
        retriever="gluomem",
        db_dir=ingested_db_dir,
        reranker=StubReranker(logit=2.0),
    )
    manifest = json.loads(paths.manifest.read_text())
    info = manifest["retriever_info"]
    assert info is not None
    assert info["rerank"] == {
        "enabled": True,
        "model": "reranker",
        "pool": 100,
        "active": True,
    }
    # Runner forces abstain off for gluomem; the dumped config reflects reality.
    assert manifest["config"]["search"]["abstain"]["enabled"] is False
    assert info["abstain"] == {
        "enabled": False,
        "score": "fusion",
        "threshold": 0.0,
        "calibration": "uncalibrated",
    }
    # Every prediction row carries its abstention decision and score contract,
    # and rescore reads it back.
    rows = [json.loads(line) for line in paths.predictions.read_text().splitlines()]
    assert rows and all(r["abstention"]["score"] == "fusion" for r in rows)
    assert {r["abstention"]["reason"] for r in rows} <= {"disabled", "no_hits"}
    from gluomem.bench.rescore import _prediction_from_row

    assert _prediction_from_row(rows[0]).abstention == rows[0]["abstention"]


def test_single_family_run_has_no_trio_entry(
    run_cfg: AppConfig, tmp_path: Path
) -> None:
    cfg = run_cfg.with_overrides(
        {
            "benchmark.judge.family": "single",
            "benchmark.judge.backbones": ["haiku"],
        }
    )
    paths = _run(cfg, tmp_path)
    entries = json.loads(paths.scores.read_text())
    assert not any(e["tag"]["judge_family"] == "trio_strict" for e in entries)
    assert any(
        e["tag"] == {
            "metric_family": "qa_accuracy",
            "judge_family": "single",
            "judge_count": 1,
            "backbone": "haiku",
            "dataset_sha": e["tag"]["dataset_sha"],
            "protocol": "adapted:locomo@v1",
        }
        for e in entries
    )


def test_answer_failure_degrades_not_aborts(
    run_cfg: AppConfig, tmp_path: Path, capsys
) -> None:
    """One sample's answer crash = empty answer judged incorrect; every other
    sample keeps its result — a handful of parse errors must not kill a run."""
    from gluomem.llm.outputs import AnswerResult

    def flaky_answer(question: str, context: str) -> AnswerResult:
        if "career" in question:  # exactly one fixture question matches
            raise RuntimeError("adapter exploded")
        return AnswerResult(text=context)

    paths = run_benchmark(
        run_cfg,
        dataset="locomo",
        retriever="bm25",
        out_root=tmp_path / "runs",
        answer_fn=flaky_answer,
    )
    rows = [json.loads(x) for x in paths.predictions.read_text().splitlines()]
    assert len(rows) == 4  # full denominator preserved
    failed = [r for r in rows if r["answer"] == ""]
    assert len(failed) == 1
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["failures"]["answer"] == [failed[0]["sample_id"]]
    assert "answer failed" in capsys.readouterr().err
    assert failed[0]["failure"] == "error"
    # The failed sample stays in the retrieval denominator as a zero.
    denominators = json.loads(paths.denominators.read_text())
    assert denominators["predictions"]["failed"] == {"error": 1}
    assert denominators["retrieval"]["scored"] == 4
    assert denominators["retrieval"]["scored_with_empty_retrieval"] >= 1
    entries = json.loads(paths.scores.read_text())
    hit = next(
        e for e in entries
        if e["scope"] == "overall" and e["tag"]["metric_family"] == "evidence_hit@5"
    )
    assert hit["n"] == 4


def test_timeout_is_recorded_as_its_own_failure_kind(
    run_cfg: AppConfig, tmp_path: Path
) -> None:
    from gluomem.llm.outputs import AnswerResult

    def slow(question: str, context: str) -> AnswerResult:
        if "career" in question:
            raise TimeoutError("reader timed out")
        return AnswerResult(text=context)

    paths = _run(run_cfg, tmp_path, answer_fn=slow)
    denominators = json.loads(paths.denominators.read_text())
    assert denominators["predictions"]["failed"] == {"timeout": 1}
    assert denominators["samples"] == 4
    assert "timeout=1" in paths.report.read_text()


def test_denominators_artifact_and_report_section(run_cfg: AppConfig, tmp_path: Path) -> None:
    paths = _run(run_cfg, tmp_path)
    d = json.loads(paths.denominators.read_text())
    assert d["samples"] == 4
    p = d["predictions"]
    assert p["answered"] + sum(p["abstained"].values()) + sum(p["failed"].values()) == 4
    assert d["judge"] == {
        "expected_records": 12,
        "records": 12,
        "missing_by_backbone": {"gpt5-mini": 0, "haiku": 0, "qwen3": 0},
    }
    # LoCoMo category 5 and gold-less rows are counted, not silently dropped.
    assert set(d["excluded_from_pool"]) == {"adversarial_category", "missing_gold"}
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["excluded_from_pool"] == d["excluded_from_pool"]
    assert "## Denominators" in paths.report.read_text()


def test_full_context_reports_retrieval_as_not_applicable(
    run_cfg: AppConfig, tmp_path: Path
) -> None:
    paths = _run(run_cfg, tmp_path, retriever="full_context")
    d = json.loads(paths.denominators.read_text())
    assert d["retrieval"] == {
        "scored": 0,
        "scored_with_empty_retrieval": 0,
        "not_applicable": {"retriever_without_retrieval": 4},
    }
    entries = json.loads(paths.scores.read_text())
    assert not any(e["tag"]["judge_family"] == "none" for e in entries)
    assert json.loads(paths.manifest.read_text())["retrieves"] is False


def test_judge_failure_scores_incorrect_not_abort(
    run_cfg: AppConfig, tmp_path: Path
) -> None:
    from gluomem.llm.outputs import Verdict

    calls = {"n": 0}

    def flaky_judge(question: str, gold: str, prediction: str) -> Verdict:
        calls["n"] += 1
        if calls["n"] == 1:  # first verdict of the run explodes
            raise RuntimeError("judge exploded")
        return Verdict(correct=True, score=1.0, rationale="r")

    fns = {b: flaky_judge for b in ("haiku", "gpt5-mini", "qwen3")}
    paths = run_benchmark(
        run_cfg.with_overrides({"benchmark.parallelism": 1}),
        dataset="locomo",
        retriever="bm25",
        out_root=tmp_path / "runs",
        judge_fns=fns,
    )
    rows = [json.loads(x) for x in paths.judgments.read_text().splitlines()]
    assert len(rows) == 4 * 3 - 1  # one verdict missing, run completed
    manifest = json.loads(paths.manifest.read_text())
    assert len(manifest["failures"]["judge"]) == 1
    # The sample missing a verdict scores trio-incorrect, denominator intact.
    entries = json.loads(paths.scores.read_text())
    trio = next(
        e for e in entries
        if e["scope"] == "overall" and e["tag"]["judge_family"] == "trio_strict"
    )
    assert trio["n"] == 4


def test_ingest_per_conversation_layout(bench_cfg: AppConfig, tmp_path: Path) -> None:
    from gluomem.llm import NullExtractors

    cfg = bench_cfg.with_overrides({"write.extraction_backend": "none"})
    db_dir = tmp_path / "dbs"
    reports = ingest_per_conversation(
        cfg, dataset="locomo", db_dir=db_dir, facade=NullExtractors()
    )
    assert len(reports) == 1
    assert (db_dir / "conv-tiny-1.mind").is_file()
    assert reports[0].turns == 9


FIXTURE_2CONV = Path(__file__).parent.parent / "fixtures" / "locomo_tiny_2conv.json"


def _cfg_2conv(bench_cfg: AppConfig) -> AppConfig:
    import hashlib

    sha = hashlib.sha256(FIXTURE_2CONV.read_bytes()).hexdigest()
    return bench_cfg.with_overrides(
        {
            "benchmark.datasets.locomo.file": FIXTURE_2CONV.name,
            "benchmark.datasets.locomo.sha256": sha,
            "write.extraction_backend": "none",
        }
    )


def _frame_dump(db_path: Path, cfg: AppConfig) -> list[tuple]:
    """Semantic frame content (ignoring ids / ingest-time timestamps) for a DB."""
    from gluomem.storage.db import MindDB

    db = MindDB(db_path, dim=cfg.storage.vector_dim)
    try:
        return db.conn.execute(
            "SELECT frame_type, gop_id, content, event_time, content_hash "
            "FROM memory_frames ORDER BY gop_id, content, content_hash"
        ).fetchall()
    finally:
        db.close()


def test_ingest_cross_conv_deterministic(bench_cfg: AppConfig, tmp_path: Path) -> None:
    """Parallel batch ingest produces byte-equal per-conversation content vs
    sequential — separate .mind DBs, so scheduling can't change what's stored."""
    from gluomem.llm import NullExtractors

    cfg = _cfg_2conv(bench_cfg)
    seq, par = tmp_path / "seq", tmp_path / "par"
    ingest_per_conversation(
        cfg.with_overrides({"write.conv_parallelism": 1}),
        dataset="locomo",
        db_dir=seq,
        facade=NullExtractors(),
    )
    ingest_per_conversation(
        cfg.with_overrides({"write.conv_parallelism": 4}),
        dataset="locomo",
        db_dir=par,
        facade=NullExtractors(),
    )
    for conv_id in ("conv-tiny-a", "conv-tiny-b"):
        assert _frame_dump(seq / f"{conv_id}.mind", cfg) == _frame_dump(
            par / f"{conv_id}.mind", cfg
        ), conv_id


def test_ingest_reports_returned_in_conv_order(
    bench_cfg: AppConfig, tmp_path: Path
) -> None:
    """Reports come back in input-conversation order despite out-of-order thread
    completion (conv-tiny-a has 4 turns, conv-tiny-b has 3)."""
    from gluomem.llm import NullExtractors

    cfg = _cfg_2conv(bench_cfg).with_overrides({"write.conv_parallelism": 4})
    reports = ingest_per_conversation(
        cfg, dataset="locomo", db_dir=tmp_path / "dbs", facade=NullExtractors()
    )
    assert [r.turns for r in reports] == [4, 3]


def test_ingest_conv_failure_is_tolerated(
    bench_cfg: AppConfig, tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One conversation's crash is recorded; the rest of the batch still ingests
    (mirrors run_benchmark's per-sample failure tolerance)."""
    import gluomem.write.ingest as ingest_mod
    from gluomem.llm import NullExtractors

    real = ingest_mod.ingest_conversation

    def flaky(conv, **kw):
        if conv.conv_id == "conv-tiny-b":
            raise RuntimeError("boom")
        return real(conv, **kw)

    monkeypatch.setattr(ingest_mod, "ingest_conversation", flaky)
    cfg = _cfg_2conv(bench_cfg).with_overrides({"write.conv_parallelism": 4})
    reports = ingest_per_conversation(
        cfg, dataset="locomo", db_dir=tmp_path / "dbs", facade=NullExtractors()
    )
    assert len(reports) == 1  # conv-tiny-a survived
    assert "ingest failed for conv-tiny-b" in capsys.readouterr().err


def test_ingest_skip_existing(bench_cfg: AppConfig, tmp_path: Path, capsys) -> None:
    from gluomem.llm import NullExtractors

    cfg = bench_cfg.with_overrides({"write.extraction_backend": "none"})
    db_dir = tmp_path / "dbs"
    ingest_per_conversation(
        cfg, dataset="locomo", db_dir=db_dir, facade=NullExtractors()
    )
    capsys.readouterr()
    again = ingest_per_conversation(
        cfg,
        dataset="locomo",
        db_dir=db_dir,
        facade=NullExtractors(),
        skip_existing=True,
    )
    assert again == []  # nothing re-ingested
    assert "skipped conv-tiny-1" in capsys.readouterr().out


def test_ingest_writes_completion_sentinel(
    bench_cfg: AppConfig, tmp_path: Path
) -> None:
    """The sentinel lands only after the whole conversation ingested; it is
    the resume marker — existence-only resume would silently accept partial
    .mind files."""
    from gluomem.llm import NullExtractors

    cfg = bench_cfg.with_overrides({"write.extraction_backend": "none"})
    db_dir = tmp_path / "dbs"
    ingest_per_conversation(
        cfg, dataset="locomo", db_dir=db_dir, facade=NullExtractors()
    )
    sentinel = db_dir / "conv-tiny-1.ingest.json"
    assert sentinel.is_file()
    meta = json.loads(sentinel.read_text())
    assert meta["conv_id"] == "conv-tiny-1"
    assert meta["sessions"] >= 1


def test_partial_mind_without_sentinel_raises(
    bench_cfg: AppConfig, tmp_path: Path
) -> None:
    """A .mind without its sentinel is a partial ingest: refused loudly on
    resume (and on plain re-ingest), never silently skipped or scored."""
    from gluomem.llm import NullExtractors

    cfg = bench_cfg.with_overrides({"write.extraction_backend": "none"})
    db_dir = tmp_path / "dbs"
    db_dir.mkdir()
    (db_dir / "conv-tiny-1.mind").touch()  # simulates a mid-run interruption
    for skip in (True, False):
        with pytest.raises(RuntimeError, match="without"):
            ingest_per_conversation(
                cfg,
                dataset="locomo",
                db_dir=db_dir,
                facade=NullExtractors(),
                skip_existing=skip,
            )


def test_ingest_failure_leaves_no_sentinel(
    bench_cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gluomem.write.ingest as ingest_mod
    from gluomem.llm import NullExtractors

    def boom(conv, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(ingest_mod, "ingest_conversation", boom)
    cfg = bench_cfg.with_overrides({"write.extraction_backend": "none"})
    db_dir = tmp_path / "dbs"
    ingest_per_conversation(
        cfg, dataset="locomo", db_dir=db_dir, facade=NullExtractors()
    )
    assert not (db_dir / "conv-tiny-1.ingest.json").exists()


def test_verify_ingest_completeness(bench_cfg: AppConfig, tmp_path: Path) -> None:
    """`bench verify` audit: complete conv passes; deleting a session GOP's
    raw turns flags it incomplete; a conv with no .mind is absent (not a
    failure — subset ingests are legitimate)."""
    import sqlite3

    from gluomem.bench.runner import verify_ingest_completeness
    from gluomem.llm import NullExtractors

    cfg = bench_cfg.with_overrides({"write.extraction_backend": "none"})
    db_dir = tmp_path / "dbs"
    ingest_per_conversation(
        cfg, dataset="locomo", db_dir=db_dir, facade=NullExtractors()
    )
    rows = verify_ingest_completeness(cfg, dataset="locomo", db_dir=db_dir)
    assert [r["status"] for r in rows] == ["complete"]

    # Amputate one session's raw turns to make the conv incomplete.
    con = sqlite3.connect(db_dir / "conv-tiny-1.mind")
    con.execute(
        "DELETE FROM memory_frames WHERE gop_id LIKE 'conv-tiny-1:session:%' "
        "AND gop_id = (SELECT gop_id FROM memory_frames "
        "WHERE gop_id LIKE 'conv-tiny-1:session:%' LIMIT 1)"
    )
    con.commit()
    con.close()
    rows = verify_ingest_completeness(cfg, dataset="locomo", db_dir=db_dir)
    assert rows[0]["status"] == "incomplete"
    assert rows[0]["found_sessions"] == rows[0]["expected_sessions"] - 1

    # Absent: point at an empty dir.
    rows = verify_ingest_completeness(
        cfg, dataset="locomo", db_dir=tmp_path / "empty"
    )
    assert [r["status"] for r in rows] == ["absent"]


def test_verify_write_sentinels_blesses_legacy_dir(
    bench_cfg: AppConfig, tmp_path: Path, capsys
) -> None:
    """Pre-sentinel (legacy) complete dirs: verify --write-sentinels writes the
    sentinel so --skip-existing resume accepts them."""
    from gluomem.bench.runner import verify_ingest_completeness
    from gluomem.llm import NullExtractors

    cfg = bench_cfg.with_overrides({"write.extraction_backend": "none"})
    db_dir = tmp_path / "dbs"
    ingest_per_conversation(
        cfg, dataset="locomo", db_dir=db_dir, facade=NullExtractors()
    )
    sentinel = db_dir / "conv-tiny-1.ingest.json"
    sentinel.unlink()  # simulate a dir ingested before sentinels existed
    verify_ingest_completeness(
        cfg, dataset="locomo", db_dir=db_dir, write_sentinels=True
    )
    assert sentinel.is_file()
    assert (
        json.loads(sentinel.read_text())["verified_by"]
        == "bench verify --write-sentinels"
    )
    capsys.readouterr()
    again = ingest_per_conversation(
        cfg,
        dataset="locomo",
        db_dir=db_dir,
        facade=NullExtractors(),
        skip_existing=True,
    )
    assert again == []
    assert "skipped conv-tiny-1" in capsys.readouterr().out


def test_manifest_carries_timing_and_cost_fields(
    run_cfg: AppConfig, tmp_path: Path
) -> None:
    paths = _run(run_cfg, tmp_path)
    manifest = json.loads(paths.manifest.read_text())
    timing = manifest["timing"]
    assert timing["predict_seconds"] >= 0 and timing["judge_seconds"] >= 0
    assert timing["parallelism"] == run_cfg.benchmark.parallelism
    usage = manifest["usage"]
    for key in (
        "answer_prompt_tokens", "answer_completion_tokens",
        "answer_cost_usd_estimate", "judge_calls", "judge_usage",
        "judge_cost_usd_estimate", "total_cost_usd_estimate",
    ):
        assert key in usage, key
    assert usage["judge_usage"] == {}  # fake path: no LLM tokens


def test_run_stays_dspy_free_on_fake_backends(tmp_path: Path) -> None:
    """The fake-backend pipeline must never import dspy (offline gate)."""
    from tests.bench.conftest import BENCH_OVERRIDES

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
