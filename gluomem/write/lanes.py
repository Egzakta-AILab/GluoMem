"""Fact, event, and profile lane extraction over session text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from datetime import datetime

    from gluomem.config.app_config import AppConfig, LaneConfig
    from gluomem.frames.store import Importance
    from gluomem.llm.outputs import Extractors

LaneName = Literal["fact", "event", "profile"]
_T = TypeVar("_T")


@dataclass
class Observation:
    """One lane-extracted observation, ready for frame-model admission."""

    lane: LaneName
    text: str
    event_time: str | None = None
    speaker: str | None = None
    confidence: float = 1.0


def _lane_cfg(lane: LaneName, cfg: AppConfig) -> LaneConfig:
    return getattr(cfg.write.lanes, lane)


def lane_importance(lane: LaneName, cfg: AppConfig) -> Importance:
    """Return the admission importance configured for ``lane``."""
    importance: Importance = _lane_cfg(lane, cfg).importance  # type: ignore[assignment]
    return importance


def extract_observations(
    session_text: str,
    *,
    facade: Extractors,
    reference_date: datetime,
    speakers: Sequence[str],
    speaker_texts: Mapping[str, str],
    cfg: AppConfig,
) -> list[Observation]:
    """Run every enabled lane over one session in deterministic order."""
    thunks: list[tuple[str, Callable[[], list[Observation]]]] = []

    if _lane_cfg("fact", cfg).enabled:
        thunks.append(("fact", lambda: _fact_observations(facade, session_text)))

    if _lane_cfg("event", cfg).enabled:
        thunks.append(("event", lambda: _event_observations(facade, session_text, reference_date)))

    if _lane_cfg("profile", cfg).enabled:
        for speaker in speakers:
            text = speaker_texts.get(speaker, "")
            if not text.strip():
                continue
            thunks.append(
                (
                    f"profile:{speaker}",
                    lambda text=text, speaker=speaker: _profile_observations(facade, text, speaker),
                )
            )

    results = run_ordered(thunks, max_workers=cfg.write.parallelism)
    out: list[Observation] = []
    for _, observations in results:
        out.extend(observations)
    return out


def _fact_observations(facade: Extractors, session_text: str) -> list[Observation]:
    return [
        Observation(lane="fact", text=f.text, confidence=f.confidence)
        for f in facade.extract_facts(session_text)
    ]


def _event_observations(
    facade: Extractors, session_text: str, reference_date: datetime
) -> list[Observation]:
    out: list[Observation] = []
    for event in facade.extract_events(session_text, reference_date):
        event_time = (
            event.event_date.isoformat(timespec="seconds") if event.event_date is not None else None
        )
        out.append(
            Observation(
                lane="event",
                text=event.text,
                event_time=event_time,
                confidence=event.confidence,
            )
        )
    return out


def _profile_observations(facade: Extractors, text: str, speaker: str) -> list[Observation]:
    return [
        Observation(lane="profile", text=f.text, speaker=speaker, confidence=f.confidence)
        for f in facade.extract_profile(text, speaker=speaker)
    ]


def run_ordered(
    thunks: Sequence[tuple[str, Callable[[], _T]]],
    *,
    max_workers: int,
) -> list[tuple[str, _T]]:
    """Run keyed thunks, returning ``(key, result)`` in input order."""
    if max_workers <= 1 or len(thunks) <= 1:
        return [(key, thunk()) for key, thunk in thunks]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(max_workers, len(thunks))) as pool:
        futures = [(key, pool.submit(thunk)) for key, thunk in thunks]
        return [(key, future.result()) for key, future in futures]
