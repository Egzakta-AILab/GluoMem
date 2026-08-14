"""LoCoMo ingest e2e over the committed fixture — parse, GOPs, dedup,
quarantine exclusion, reconstruction, re-ingest idempotency, report counts.

Runs with FakeExtractors (deterministic rule-based) — fully offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gluomem.llm import FakeExtractors
from gluomem.write.gate import InjectionScanner
from gluomem.write.ingest import (
    _parse_locomo_datetime,
    ingest_conversation,
    load_locomo,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "locomo_tiny.json"


def test_parse_fixture_structure() -> None:
    convs = load_locomo(FIXTURE)
    assert len(convs) == 1
    conv = convs[0]
    assert conv.conv_id == "conv-tiny-1"
    assert conv.speakers == ("Alice", "Bob")
    assert [s.index for s in conv.sessions] == [1, 2, 3]
    assert conv.sessions[0].date_time.isoformat() == "2023-05-08T13:56:00"
    assert len(conv.sessions[0].turns) == 4
    # Image turn folds the blip_caption into text.
    assert "shares photo: a golden retriever puppy" in conv.sessions[0].turns[3].text


def test_locomo_datetime_strict() -> None:
    assert _parse_locomo_datetime("1:56 pm on 8 May, 2023").hour == 13
    with pytest.raises(ValueError, match="unrecognized"):
        _parse_locomo_datetime("mai huitieme, treize heures")


def _ingest(env, cfg=None):
    cfg = cfg or env.cfg
    conv = load_locomo(FIXTURE)[0]
    facade = FakeExtractors(banner=False)
    report = ingest_conversation(
        conv,
        facade=facade,
        scanner=InjectionScanner(cfg.safety),
        store=env.store,
        kg=env.kg,
        cfg=cfg,
    )
    return report, facade


def test_raw_turn_gops_and_event_time(env) -> None:
    report, _ = _ingest(env)
    assert report.conversations == 1 and report.sessions == 3
    assert report.turns == 9
    rows = env.db.conn.execute(
        "SELECT gop_id, event_time FROM memory_frames "
        "WHERE gop_id LIKE '%:session:%' AND frame_type='I'"
    ).fetchall()
    gops = {r[0] for r in rows}
    assert gops == {
        "conv-tiny-1:session:1",
        "conv-tiny-1:session:2",
        "conv-tiny-1:session:3",
    }
    s1_times = {r[1] for r in rows if r[0].endswith(":session:1")}
    assert s1_times == {"2023-05-08T13:56:00"}


def test_injection_turn_quarantined_and_excluded_from_extraction(env) -> None:
    report, _ = _ingest(env)
    assert report.quarantined == 1
    row = env.db.conn.execute(
        "SELECT trust FROM memory_frames WHERE content LIKE '%ignore all previous%'"
    ).fetchone()
    assert row is not None and row[0] == "quarantined"
    # The flagged text never reached extraction: the rule-based fake sentence-
    # splits its input, so an included injection turn would surface as a fact/
    # profile frame carrying the marker — none may exist outside the raw GOP.
    leaked = env.db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames "
        "WHERE content LIKE '%ignore all previous%' "
        "AND gop_id NOT LIKE '%:session:%'"
    ).fetchone()[0]
    assert leaked == 0


def test_fact_lane_dedup_across_sessions(env) -> None:
    """'I adopted a cat yesterday.' said in sessions 1 AND 2 -> one fact frame."""
    report, _ = _ingest(env)
    rows = env.db.conn.execute(
        "SELECT content FROM memory_frames WHERE gop_id='conv-tiny-1:facts' "
        "AND content LIKE '%adopted a cat%'"
    ).fetchall()
    assert len(rows) == 1
    assert report.deduped > 0


def test_profile_gop_reconstruct_state(env) -> None:
    _ingest(env)
    latest_i, p_since = env.store.reconstruct_state("conv-tiny-1:profile:Alice")
    assert latest_i is not None
    assert latest_i.gop_id == "conv-tiny-1:profile:Alice"
    assert isinstance(p_since, list)


def test_reingest_is_idempotent(env) -> None:
    _ingest(env)
    count_before = env.db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames"
    ).fetchone()[0]
    report2, _ = _ingest(env)
    count_after = env.db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames"
    ).fetchone()[0]
    assert count_after == count_before
    assert report2.i_frames == 0  # everything deduped at the gate
    assert report2.deduped > 0


def test_reject_mode_drops_injection_turn(env) -> None:
    cfg = env.cfg.with_overrides({"safety.fail_mode": "reject"})
    report, _ = _ingest(env, cfg)
    assert report.rejected == 1
    row = env.db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames WHERE content LIKE '%ignore all previous%'"
    ).fetchone()
    assert row[0] == 0


def test_raw_turns_disabled_still_extracts(env) -> None:
    cfg = env.cfg.with_overrides({"write.raw_turns.enabled": False})
    report, _ = _ingest(env, cfg)
    raw_count = env.db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames WHERE gop_id LIKE '%:session:%'"
    ).fetchone()[0]
    assert raw_count == 0
    fact_count = env.db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames WHERE gop_id='conv-tiny-1:facts'"
    ).fetchone()[0]
    assert fact_count > 0
    assert report.quarantined == 1  # injection turn still counted (scan-only)


def test_malformed_file_fails_loud(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('[{"sample_id": "x"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="missing sample_id/conversation"):
        load_locomo(bad)
