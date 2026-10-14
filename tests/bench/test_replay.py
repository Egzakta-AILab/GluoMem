"""Replay contract: ordered permitted ingestion, readiness, isolation, receipts.

Runs the generated MAB-shaped fixture with the mock embedder and a logical
clock. Numbers here are contract outputs, not benchmark results.
"""

from __future__ import annotations

import json
import sqlite3
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
from gluomem.config.app_config import AppConfig, load_config
from gluomem.embedding.providers import MockEmbedder


@pytest.fixture
def cfg() -> AppConfig:
    return load_config().with_overrides(
        {"embedding.model": "mock", "components.allow_fake": True, "search.rerank.enabled": False}
    )


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
            {"event_id": "h:e1", "seq": 1, "at": "2026-03-01T09:00:00Z",
             "text": "The ledger is kept in the north vault."},
            {"event_id": "h:e2", "seq": 2, "at": "2026-03-01T10:00:00Z",
             "text": "FUTURE-SENTINEL: the ledger moved to the south vault."},
        ],
        "probes": [probe],
    }


def _frame_contents(db_path: Path) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return [r[0] for r in conn.execute("SELECT content FROM memory_frames")]
    finally:
        conn.close()


# --- fixture contract ----------------------------------------------------------


def test_registered_fixture_loads_with_both_competencies() -> None:
    entry, histories, sha = load_registered_fixture("mab_fixture")
    assert sha == entry.sha256
    competencies = {p.competency for h in histories for p in h.probes}
    assert competencies == {"accurate_retrieval", "conflict_resolution"}
    conflict = [p for h in histories for p in h.probes if p.competency == "conflict_resolution"]
    assert conflict and all(p.stale for p in conflict)
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
def test_fixture_violations_are_rejected(tmp_path: Path, mutate: Any, match: str) -> None:
    history = _history()
    mutate(history)
    with pytest.raises(ReplayFixtureError, match=match):
        load_replay_fixture(_fixture(tmp_path, [history]))


def test_tampered_registered_fixture_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = tmp_path / "mab_tiny.json"
    copy.write_text(REPLAY_FIXTURES["mab_fixture"].read_text().replace("Hall C", "Hall D"))
    monkeypatch.setitem(REPLAY_FIXTURES, "mab_fixture", copy)
    with pytest.raises(ReplayFixtureError, match="registry pin"):
        load_registered_fixture("mab_fixture")


# --- ingestion order, gold and future isolation ----------------------------------


def test_each_probe_sees_only_events_up_to_its_cursor(cfg: AppConfig, tmp_path: Path) -> None:
    receipt, _ = run_replay_fixture(cfg, embedder=_embedder(cfg), work_dir=tmp_path / "w")
    _, histories, _ = load_registered_fixture("mab_fixture")
    seq = {e.event_id: e.seq for h in histories for e in h.events}
    for row in receipt["probes"]:
        allowed = [eid for eid, s in seq.items() if eid.startswith(row["history_id"] + ":")
                   and s <= row["after_seq"]]
        assert row["visible_events"] == allowed
        assert set(row["retrieved"]) <= set(row["visible_events"])
        assert row["ready"] is True


def test_question_gold_and_future_events_never_enter_memory(
    cfg: AppConfig, tmp_path: Path
) -> None:
    histories, sha = load_replay_fixture(_fixture(tmp_path, [_history()]))
    runner = ReplayRunner(cfg, embedder=_embedder(cfg), work_dir=tmp_path / "w")
    receipt = runner.run(histories, entry=get_entry("mab_fixture"), dataset_sha=sha)
    contents = _frame_contents(tmp_path / "w" / "baseline" / "h.mind")
    assert contents == ["The ledger is kept in the north vault."]
    joined = "\n".join(contents)
    for sentinel in ("QUESTION-SENTINEL", "GOLD-SENTINEL", "FUTURE-SENTINEL"):
        assert sentinel not in joined
    assert receipt["probes"][0]["visible_events"] == ["h:e1"]


def test_query_waits_for_ready_projections(
    cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gluomem.frames.store import FrameStore, IndexState
    from gluomem.search.hybrid import HybridSearch

    def not_ready(self: FrameStore, frame_id: int) -> IndexState:
        return IndexState(frame_id, fts_current=True, frame_vec=0, chunks=0,
                          expected_chunks=0, chunk_vecs=0)

    def never(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("searched before the index was ready")

    monkeypatch.setattr(FrameStore, "index_state", not_ready)
    monkeypatch.setattr(HybridSearch, "search", never)
    with pytest.raises(ReplayNotReadyError, match="not indexed"):
        run_replay_fixture(cfg, embedder=_embedder(cfg), work_dir=tmp_path / "w")


def test_histories_and_candidates_get_isolated_stores(cfg: AppConfig, tmp_path: Path) -> None:
    work = tmp_path / "w"
    receipt, _ = run_replay_fixture(cfg, embedder=_embedder(cfg), work_dir=work)
    for row in receipt["probes"]:
        assert all(eid.startswith(row["history_id"] + ":") for eid in row["retrieved"])
    assert sorted(p.name for p in (work / "baseline").glob("*.mind")) == [
        "kestrel.mind", "orion.mind",
    ]
    with pytest.raises(FileExistsError, match="fresh store"):
        run_replay_fixture(cfg, embedder=_embedder(cfg), work_dir=work)
    other, _ = run_replay_fixture(cfg, embedder=_embedder(cfg), work_dir=work, candidate="b")
    assert other["candidate"] == "b"


def test_logical_clock_not_wall_clock(
    cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gluomem.search.hybrid as hybrid

    def boom() -> Any:
        raise AssertionError("replay read the wall clock")

    monkeypatch.setattr(hybrid, "utc_now", boom)
    receipt, _ = run_replay_fixture(cfg, embedder=_embedder(cfg), work_dir=tmp_path / "w")
    assert receipt["clock"].startswith("logical")


# --- scoring and receipts ----------------------------------------------------------


def _probe(**kw: Any) -> ReplayProbe:
    base: dict[str, Any] = {
        "probe_id": "p", "after_seq": 3, "at": "2026-01-01T00:00:00Z",
        "competency": "conflict_resolution", "question": "q",
        "answers": ("Hall C",), "evidence": ("e3",), "stale": ("e1",),
    }
    base.update(kw)
    return ReplayProbe(**base)


def test_conflict_scoring_rewards_current_over_stale() -> None:
    current_first = [("e3", "moved to Hall C"), ("e1", "Hall A")]
    stale_first = [("e1", "Hall A"), ("e3", "moved to Hall C")]
    assert _probe_scores(_probe(), current_first, 2)["current_over_stale@2"] == 1.0
    assert _probe_scores(_probe(), stale_first, 2)["current_over_stale@2"] == 0.0
    assert _probe_scores(_probe(), stale_first, 1) == {
        "evidence_hit@1": 0.0, "context_substring_match@1": 0.0, "current_over_stale@1": 0.0,
    }
    assert _probe_scores(_probe(), [], 5)["evidence_hit@5"] == 0.0
    assert _probe_scores(_probe(stale=()), current_first, 2)["current_over_stale@2"] is None


def test_receipt_is_byte_reproducible(cfg: AppConfig, tmp_path: Path) -> None:
    a, digest_a = run_replay_fixture(
        cfg, embedder=_embedder(cfg), work_dir=tmp_path / "a", out_dir=tmp_path / "out"
    )
    b, digest_b = run_replay_fixture(cfg, embedder=_embedder(cfg), work_dir=tmp_path / "b")
    assert receipt_text(a) == receipt_text(b)
    assert digest_a == digest_b == receipt_digest(a)
    assert (tmp_path / "out" / "receipt.json").read_text() == receipt_text(a)


def test_receipt_is_labeled_and_keeps_families_and_denominators(
    cfg: AppConfig, tmp_path: Path
) -> None:
    receipt, _ = run_replay_fixture(cfg, embedder=_embedder(cfg), work_dir=tmp_path / "w")
    assert receipt["official"] is False and receipt["label"] == NOT_OFFICIAL_LABEL
    assert receipt["benchmark"]["protocol"] == "adapted:mab_fixture@v1"
    assert receipt["dataset_sha"] == get_entry("mab_fixture").sha256
    assert receipt["embedder"]["implementation"] == "gluomem.mock"
    tags = {(e["tag"]["protocol"], e["tag"]["judge_family"]) for e in receipt["scores"]}
    assert tags == {("adapted:mab_fixture@v1", "none")}
    by = {(e["scope"], e["tag"]["metric_family"]): e for e in receipt["scores"]}
    assert by[("overall", "evidence_hit@5")]["n"] == 5
    assert by[("competency:conflict_resolution", "current_over_stale@5")]["n"] == 2
    assert ("competency:accurate_retrieval", "current_over_stale@5") not in by
    d = receipt["denominators"]
    assert d["probes"] == 5
    assert d["by_competency"] == {"accurate_retrieval": 3, "conflict_resolution": 2}
    assert d["not_applicable"]["current_over_stale@5"] == 3


def test_replay_cli_writes_receipt(tmp_path: Path, capsys) -> None:
    from gluomem.cli import main

    rc = main([
        "bench", "replay", "--work-dir", str(tmp_path / "w"), "--out", str(tmp_path / "out"),
        "--override", "embedding.model=mock", "--override", "components.allow_fake=true",
        "--override", "search.rerank.enabled=false",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert NOT_OFFICIAL_LABEL in out
    receipt = json.loads((tmp_path / "out" / "receipt.json").read_text())
    assert receipt_digest(receipt) in out
