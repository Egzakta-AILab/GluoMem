"""Sequential replay contract and offline run rescore."""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gluomem.bench.registry import get_entry
from gluomem.bench.replay import (
    NOT_OFFICIAL_LABEL,
    REPLAY_FIXTURES,
    ReplayFixtureError,
    ReplayNotReadyError,
    ReplayProbe,
    ReplayRunner,
    _probe_scores,
    load_registered_fixture,
    load_replay_fixture,
    receipt_digest,
    receipt_text,
    run_replay_fixture,
)
from gluomem.bench.rescore import RescoreMismatchError, rescore_run
from gluomem.bench.runner import run_benchmark
from gluomem.config.app_config import AppConfig, load_config
from gluomem.embedding.providers import MockEmbedder
from tests.bench.conftest import FAKE_BACKENDS


@pytest.fixture
def replay_cfg() -> AppConfig:
    return load_config().with_overrides(
        {
            "embedding.model": "mock",
            "components.allow_fake": True,
            "search.rerank.enabled": False,
        }
    )


@pytest.fixture
def run_paths(bench_cfg: AppConfig, tmp_path: Path):
    cfg = bench_cfg.with_overrides(dict(FAKE_BACKENDS))
    return run_benchmark(cfg, dataset="locomo", retriever="bm25", out_root=tmp_path / "runs")


def _embedder(cfg: AppConfig) -> MockEmbedder:
    return MockEmbedder(dim=cfg.storage.vector_dim)


def _fixture(tmp_path: Path, histories: list[dict[str, Any]]) -> Path:
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps({"histories": histories}), encoding="utf-8")
    return path


def _history(**probe_overrides: Any) -> dict[str, Any]:
    probe = {
        "probe_id": "h:p1",
        "after_seq": 1,
        "at": "2026-03-01T09:30:00Z",
        "competency": "accurate_retrieval",
        "question": "Which vault holds the QUESTION-SENTINEL ledger?",
        "answers": ["GOLD-SENTINEL"],
        "evidence": ["h:e1"],
    }
    probe.update(probe_overrides)
    return {
        "history_id": "h",
        "events": [
            {
                "event_id": "h:e1",
                "seq": 1,
                "at": "2026-03-01T09:00:00Z",
                "text": "The ledger is kept in the north vault.",
            },
            {
                "event_id": "h:e2",
                "seq": 2,
                "at": "2026-03-01T10:00:00Z",
                "text": "FUTURE-SENTINEL: the ledger moved to the south vault.",
            },
        ],
        "probes": [probe],
    }


def _frame_contents(db_path: Path) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return [r[0] for r in conn.execute("SELECT content FROM memory_frames")]
    finally:
        conn.close()


def _probe(**kw: Any) -> ReplayProbe:
    base: dict[str, Any] = {
        "probe_id": "p",
        "after_seq": 3,
        "at": "2026-01-01T00:00:00Z",
        "competency": "conflict_resolution",
        "question": "q",
        "answers": ("Hall C",),
        "evidence": ("e3",),
        "stale": ("e1",),
    }
    base.update(kw)
    return ReplayProbe(**base)


def test_registered_replay_fixture() -> None:
    entry, histories, sha = load_registered_fixture("mab_fixture")
    assert sha == entry.sha256
    competencies = {p.competency for h in histories for p in h.probes}
    assert competencies == {"accurate_retrieval", "conflict_resolution"}
    raw = json.loads(REPLAY_FIXTURES["mab_fixture"].read_text())
    assert raw["generated"] is True and raw["official"] is False


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda h: h["probes"][0].update(evidence=["h:e2"]), "future event"),
        (lambda h: h["probes"][0].update(stale=["h:e9"]), "unknown event"),
        (lambda h: h["probes"][0].update(at="2026-03-01T11:00:00Z"), "may not see"),
        (lambda h: h["probes"][0].update(at="2026-03-01T08:00:00Z"), "before its last"),
        (lambda h: h["probes"][0].update(answers=[]), "no accepted answers"),
        (lambda h: h["events"][1].update(seq=3), "seq must run"),
        (lambda h: h["events"][1].update(at="2026-02-01T00:00:00Z"), "backwards"),
    ],
)
def test_replay_fixture_violations_rejected(tmp_path: Path, mutate: Any, match: str) -> None:
    history = _history()
    mutate(history)
    with pytest.raises(ReplayFixtureError, match=match):
        load_replay_fixture(_fixture(tmp_path, [history]))


def test_replay_cursor_isolation_readiness_and_logical_clock(
    replay_cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gluomem.search.hybrid as hybrid

    monkeypatch.setattr(
        hybrid, "utc_now", lambda: (_ for _ in ()).throw(AssertionError("wall clock"))
    )
    work = tmp_path / "w"
    receipt, digest = run_replay_fixture(
        replay_cfg, embedder=_embedder(replay_cfg), work_dir=work, out_dir=tmp_path / "out"
    )
    assert receipt["clock"].startswith("logical")
    assert receipt["official"] is False and receipt["label"] == NOT_OFFICIAL_LABEL
    assert (tmp_path / "out" / "receipt.json").read_text() == receipt_text(receipt)
    assert digest == receipt_digest(receipt)

    _, histories, _ = load_registered_fixture("mab_fixture")
    seq = {e.event_id: e.seq for h in histories for e in h.events}
    for row in receipt["probes"]:
        allowed = [
            eid
            for eid, s in seq.items()
            if eid.startswith(row["history_id"] + ":") and s <= row["after_seq"]
        ]
        assert row["visible_events"] == allowed
        assert set(row["retrieved"]) <= set(row["visible_events"])
        assert row["ready"] is True

    with pytest.raises(FileExistsError, match="fresh store"):
        run_replay_fixture(replay_cfg, embedder=_embedder(replay_cfg), work_dir=work)

    h_list, sha = load_replay_fixture(_fixture(tmp_path, [_history()]))
    runner = ReplayRunner(replay_cfg, embedder=_embedder(replay_cfg), work_dir=tmp_path / "w2")
    runner.run(h_list, entry=get_entry("mab_fixture"), dataset_sha=sha)
    contents = _frame_contents(tmp_path / "w2" / "baseline" / "h.mind")
    assert contents == ["The ledger is kept in the north vault."]
    for sentinel in ("QUESTION-SENTINEL", "GOLD-SENTINEL", "FUTURE-SENTINEL"):
        assert sentinel not in "\n".join(contents)


def test_replay_not_ready_and_conflict_scoring(
    replay_cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gluomem.frames.store import FrameStore, IndexState

    monkeypatch.setattr(
        FrameStore,
        "index_state",
        lambda self, fid: IndexState(
            fid,
            fts_current=True,
            frame_vec=0,
            chunks=0,
            expected_chunks=0,
            chunk_vecs=0,
        ),
    )
    with pytest.raises(ReplayNotReadyError, match="not indexed"):
        run_replay_fixture(replay_cfg, embedder=_embedder(replay_cfg), work_dir=tmp_path / "w")

    current_first = [("e3", "moved to Hall C"), ("e1", "Hall A")]
    stale_first = [("e1", "Hall A"), ("e3", "moved to Hall C")]
    assert _probe_scores(_probe(), current_first, 2)["current_over_stale@2"] == 1.0
    assert _probe_scores(_probe(), stale_first, 2)["current_over_stale@2"] == 0.0
    assert _probe_scores(_probe(stale=()), current_first, 2)["current_over_stale@2"] is None


def test_rescore_regeneration_tamper_detection_and_dspy_free(run_paths, tmp_path: Path) -> None:
    committed_scores = run_paths.scores.read_bytes()
    committed_report = run_paths.report.read_bytes()
    committed_denom = run_paths.denominators.read_bytes()
    run_paths.scores.unlink()
    run_paths.report.unlink()
    entries = rescore_run(run_paths.root)
    assert run_paths.scores.read_bytes() == committed_scores
    assert run_paths.report.read_bytes() == committed_report
    assert entries

    rescore_run(run_paths.root, check=True)

    moved = tmp_path / "moved_run"
    shutil.copytree(run_paths.root, moved)
    assert rescore_run(moved)

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

    run_paths.denominators.unlink()
    with pytest.raises(RescoreMismatchError, match="denominators"):
        rescore_run(run_paths.root, check=True)
    rescore_run(run_paths.root)
    assert run_paths.denominators.read_bytes() == committed_denom

    lines = run_paths.judgments.read_text().splitlines()
    row = json.loads(lines[0])
    row["correct"] = not row["correct"]
    lines[0] = json.dumps(row, sort_keys=True)
    run_paths.judgments.write_text("\n".join(lines) + "\n")
    with pytest.raises(RescoreMismatchError, match="tampered|diverge"):
        rescore_run(run_paths.root, check=True)


def test_rescore_malformed_jsonl_diagnostics(run_paths) -> None:
    jlines = run_paths.judgments.read_text().splitlines()
    jlines[1] = '{"unexpected_field": true}'
    run_paths.judgments.write_text("\n".join(jlines) + "\n")
    with pytest.raises(RescoreMismatchError, match=r"judgments\.jsonl line 2"):
        rescore_run(run_paths.root, check=True)

    plines = run_paths.predictions.read_text().splitlines()
    plines[0] = "{corrupt"
    run_paths.predictions.write_text("\n".join(plines) + "\n")
    with pytest.raises(RescoreMismatchError, match=r"predictions\.jsonl line 1"):
        rescore_run(run_paths.root, check=True)
