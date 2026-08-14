"""gluomem.write.lanes — fact/event/profile lane extraction.

Each lane extracts a different kind of knowledge from a session and carries
its own admission importance (``write.lanes.<lane>.importance``):

- **fact** — atomic facts over the whole session text;
- **event** — episodic events; the extractor resolves relative time cues
  against ``reference_date`` (the session datetime) into ``event_date``,
  mapped here to the frame ``event_time`` (ISO, seconds precision). Events
  without a resolvable date keep ``event_time=None``; ingest applies the
  session-datetime fallback (``created_at`` stays ingest time);
- **profile** — durable per-speaker attributes over that speaker's turns.

dspy-free: extraction goes through the ``Extractors`` facade.
"""

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
    """The admission importance for ``lane`` (validated against scoring keys)."""
    # The config validator guarantees membership in scoring.importance_mult,
    # whose keys are exactly the Importance literals.
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
    """Run every enabled lane over one session; deterministic ordering.

    Output order: facts, then events, then profile per speaker (in
    ``speakers`` order) — stable across runs so ingest reports and dedup
    behave deterministically. With ``write.parallelism > 1`` the independent
    lane LLM calls run concurrently; results are still assembled in the
    fixed lane order, so parallel and sequential runs are byte-identical.
    """
    # One thunk per enabled lane call, keyed for ordered assembly.
    thunks: list[tuple[str, Callable[[], list[Observation]]]] = []

    if _lane_cfg("fact", cfg).enabled:
        thunks.append(("fact", lambda: _fact_observations(facade, session_text)))

    if _lane_cfg("event", cfg).enabled:
        thunks.append(
            ("event", lambda: _event_observations(facade, session_text, reference_date))
        )

    if _lane_cfg("profile", cfg).enabled:
        for speaker in speakers:
            text = speaker_texts.get(speaker, "")
            if not text.strip():
                continue
            thunks.append(
                (
                    f"profile:{speaker}",
                    lambda text=text, speaker=speaker: _profile_observations(
                        facade, text, speaker
                    ),
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
            event.event_date.isoformat(timespec="seconds")
            if event.event_date is not None
            else None
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


def _profile_observations(
    facade: Extractors, text: str, speaker: str
) -> list[Observation]:
    return [
        Observation(lane="profile", text=f.text, speaker=speaker, confidence=f.confidence)
        for f in facade.extract_profile(text, speaker=speaker)
    ]


def run_ordered(
    thunks: Sequence[tuple[str, Callable[[], _T]]],
    *,
    max_workers: int,
) -> list[tuple[str, _T]]:
    """Run keyed thunks, returning (key, result) in INPUT order.

    ``max_workers == 1`` runs inline (no threads — strict call order for
    deterministic tests); otherwise a thread pool fans out the calls and the
    results are reassembled by input index. The first exception (by input
    order) is re-raised, matching sequential semantics.
    """
    if max_workers <= 1 or len(thunks) <= 1:
        return [(key, thunk()) for key, thunk in thunks]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(max_workers, len(thunks))) as pool:
        futures = [(key, pool.submit(thunk)) for key, thunk in thunks]
        return [(key, future.result()) for key, future in futures]
