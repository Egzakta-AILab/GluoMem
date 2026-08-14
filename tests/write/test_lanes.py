"""Lane extraction tests — enablement flags, importance flow, event_time mapping."""

from __future__ import annotations

from datetime import datetime

from gluomem.llm import ExtractedEvent, FakeExtractors
from gluomem.write.lanes import extract_observations, lane_importance

_REF = datetime(2023, 5, 8)


def _extract(env, facade, cfg=None):
    return extract_observations(
        "Alice got promoted yesterday. Bob likes tea.",
        facade=facade,
        reference_date=_REF,
        speakers=["Alice", "Bob"],
        speaker_texts={"Alice": "I got promoted.", "Bob": "I like tea."},
        cfg=cfg or env.cfg,
    )


def test_all_lanes_produce_ordered_observations(env) -> None:
    obs = _extract(env, FakeExtractors(banner=False))
    lanes = [o.lane for o in obs]
    # facts first, then events, then profile (Alice before Bob).
    assert lanes == sorted(lanes, key=["fact", "event", "profile"].index)
    assert any(o.lane == "fact" for o in obs)
    assert any(o.lane == "event" for o in obs)
    profile_speakers = [o.speaker for o in obs if o.lane == "profile"]
    assert profile_speakers == ["Alice", "Bob"]


def test_event_time_resolved_from_reference_date(env) -> None:
    obs = _extract(env, FakeExtractors(banner=False))
    events = [o for o in obs if o.lane == "event"]
    assert len(events) == 1
    assert events[0].event_time == "2023-05-07T00:00:00"


def test_event_without_date_keeps_none(env) -> None:
    facade = FakeExtractors(
        events=[[ExtractedEvent(text="something happened", event_date=None)]],
        banner=False,
    )
    obs = _extract(env, facade)
    events = [o for o in obs if o.lane == "event"]
    assert len(events) == 1 and events[0].event_time is None


def test_disabled_lanes_produce_nothing(env) -> None:
    cfg = env.cfg.with_overrides(
        {
            "write.lanes.fact.enabled": False,
            "write.lanes.event.enabled": False,
            "write.lanes.profile.enabled": False,
        }
    )
    facade = FakeExtractors(banner=False)
    assert _extract(env, facade, cfg) == []
    # Disabled lanes never invoke the facade at all.
    assert facade.calls["extract_facts"] == 0
    assert facade.calls["extract_events"] == 0
    assert facade.calls["extract_profile"] == 0


def test_lane_importance_reads_config(env) -> None:
    assert lane_importance("fact", env.cfg) == "normal"
    assert lane_importance("profile", env.cfg) == "important"
    cfg = env.cfg.with_overrides({"write.lanes.fact.importance": "critical"})
    assert lane_importance("fact", cfg) == "critical"
