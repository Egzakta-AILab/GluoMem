"""gluomem.bench.replay — ordered replay of permitted events, then probes.

The sequential-ingestion protocol for incremental-memory benchmarks
(MemoryAgentBench-shaped). Each history is a list of events in arrival order
and a set of probes, each asked after a cursor (``after_seq``). For every
probe the runner:

1. ingests, through the normal write gate, only the events with
   ``seq <= after_seq`` that are not yet in memory (probe questions, accepted
   answers and later events are never passed to the write path);
2. checks that every ingested frame's FTS and vector projections are ready,
   and refuses to query otherwise;
3. searches with the probe's logical time as the clock, so recency decay
   never depends on today's date;
4. scores the retrieved events objectively (no reader model).

Each (candidate, history) gets a fresh ``.mind`` file; an existing file is
refused so no candidate or history can see another's memories.

Receipts contain no wall-clock fields, so the same fixture, config and
embedder reproduce a byte-identical receipt.

Future seam (not implemented): interactive benchmarks (MemoryArena,
STATE-Bench) need a trajectory runner where an agent acts, the environment
observes, and only the observation becomes the next eligible event. See
``TrajectoryEnvironment``; nothing in this module implements it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from gluomem.bench.metrics import MetricRegistry, objective_tag
from gluomem.bench.registry import (
    BenchmarkEntry,
    get_entry,
    manifest_record,
    protocol_id,
)
from gluomem.bench.types import Score
from gluomem.embedding.fingerprint import fingerprint_of
from gluomem.frames.store import FrameStore
from gluomem.search.hybrid import HybridSearch
from gluomem.storage.db import MindDB
from gluomem.timeutil import parse_utc
from gluomem.write.gate import InjectionScanner, admit

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig
    from gluomem.embedding.base import Embedder

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REPLAY_FIXTURES: dict[str, Path] = {"mab_fixture": FIXTURES_DIR / "mab_tiny.json"}

NOT_OFFICIAL_LABEL = (
    "GENERATED MAB-SHAPED FIXTURE: a contract test of the replay protocol, "
    "NOT an official MemoryAgentBench result"
)


class ReplayFixtureError(ValueError):
    """The fixture violates the replay contract (order, time, future gold)."""


class ReplayNotReadyError(RuntimeError):
    """An ingested frame's index projections are not ready at query time."""


@dataclass(frozen=True)
class ReplayEvent:
    event_id: str
    seq: int
    at: str
    text: str


@dataclass(frozen=True)
class ReplayProbe:
    probe_id: str
    after_seq: int
    at: str
    competency: str
    question: str
    answers: tuple[str, ...]
    evidence: tuple[str, ...]
    stale: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayHistory:
    history_id: str
    events: tuple[ReplayEvent, ...]
    probes: tuple[ReplayProbe, ...]


class TrajectoryEnvironment(Protocol):
    """Seam for a future trajectory runner (interactive benchmarks).

    ``reset`` starts an episode and returns the first observation; ``step``
    applies an agent action and returns ``(observation, done)``. A runner
    would ingest each observation as the next eligible event, recall before
    each action, and score task outcomes. No implementation exists yet.
    """

    def reset(self, episode_id: str) -> str: ...

    def step(self, action: str) -> tuple[str, bool]: ...


# --- fixture loading -----------------------------------------------------------


def _validate(h: ReplayHistory) -> None:
    where = f"history '{h.history_id}'"
    if not h.events:
        raise ReplayFixtureError(f"{where}: no events")
    seqs = [e.seq for e in h.events]
    if seqs != list(range(1, len(seqs) + 1)):
        raise ReplayFixtureError(f"{where}: event seq must run 1..n in order, got {seqs}")
    times = [parse_utc(e.at) for e in h.events]
    if times != sorted(times):
        raise ReplayFixtureError(f"{where}: event times go backwards")
    ids = [e.event_id for e in h.events]
    if len(set(ids)) != len(ids):
        raise ReplayFixtureError(f"{where}: duplicate event ids")
    seq_of = {e.event_id: e.seq for e in h.events}
    for p in h.probes:
        pw = f"{where} probe '{p.probe_id}'"
        if not 1 <= p.after_seq <= len(h.events):
            raise ReplayFixtureError(f"{pw}: after_seq {p.after_seq} outside 1..{len(h.events)}")
        at = parse_utc(p.at)
        if at < times[p.after_seq - 1]:
            raise ReplayFixtureError(f"{pw}: asked before its last visible event")
        if p.after_seq < len(h.events) and at >= times[p.after_seq]:
            raise ReplayFixtureError(f"{pw}: asked after an event it may not see")
        if not p.answers:
            raise ReplayFixtureError(f"{pw}: no accepted answers")
        for eid in (*p.evidence, *p.stale):
            if eid not in seq_of:
                raise ReplayFixtureError(f"{pw}: unknown event '{eid}'")
            if seq_of[eid] > p.after_seq:
                raise ReplayFixtureError(
                    f"{pw}: cites future event '{eid}' (seq {seq_of[eid]} > {p.after_seq})"
                )


def load_replay_fixture(path: str | Path) -> tuple[list[ReplayHistory], str]:
    """Parse and validate a replay fixture; returns (histories, file sha256)."""
    raw_bytes = Path(path).read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    try:
        data = json.loads(raw_bytes)
        histories = [
            ReplayHistory(
                history_id=str(h["history_id"]),
                events=tuple(
                    ReplayEvent(
                        event_id=str(e["event_id"]),
                        seq=int(e["seq"]),
                        at=str(e["at"]),
                        text=str(e["text"]),
                    )
                    for e in h["events"]
                ),
                probes=tuple(
                    ReplayProbe(
                        probe_id=str(p["probe_id"]),
                        after_seq=int(p["after_seq"]),
                        at=str(p["at"]),
                        competency=str(p["competency"]),
                        question=str(p["question"]),
                        answers=tuple(str(a) for a in p["answers"]),
                        evidence=tuple(str(x) for x in p["evidence"]),
                        stale=tuple(str(x) for x in p.get("stale", ())),
                    )
                    for p in h["probes"]
                ),
            )
            for h in data["histories"]
        ]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ReplayFixtureError(f"{path}: malformed replay fixture ({exc!r})") from exc
    if len({h.history_id for h in histories}) != len(histories):
        raise ReplayFixtureError(f"{path}: duplicate history ids")
    for h in histories:
        _validate(h)
    return histories, sha


def load_registered_fixture(benchmark_id: str) -> tuple[BenchmarkEntry, list[ReplayHistory], str]:
    """Load a registry fixture and verify its pinned hash."""
    entry = get_entry(benchmark_id)
    path = REPLAY_FIXTURES.get(entry.id)
    if path is None:
        raise ReplayFixtureError(f"benchmark '{entry.id}' has no replay fixture")
    histories, sha = load_replay_fixture(path)
    if sha != entry.sha256:
        raise ReplayFixtureError(
            f"{path}: sha256 {sha} does not match the registry pin {entry.sha256}"
        )
    return entry, histories, sha


# --- scoring -------------------------------------------------------------------


def _probe_scores(
    probe: ReplayProbe, retrieved: Sequence[tuple[str | None, str]], k: int
) -> dict[str, float | None]:
    top = list(retrieved[:k])
    evidence = set(probe.evidence)
    stale = set(probe.stale)
    answers = [a.lower() for a in probe.answers]
    out: dict[str, float | None] = {
        f"evidence_hit@{k}": 1.0 if any(eid in evidence for eid, _ in top) else 0.0,
        f"context_substring_match@{k}": (
            1.0 if any(a in text.lower() for a in answers for _, text in top) else 0.0
        ),
    }
    if stale:
        ev_ranks = [i for i, (eid, _) in enumerate(top) if eid in evidence]
        stale_ranks = [i for i, (eid, _) in enumerate(top) if eid in stale]
        current_first = bool(ev_ranks) and (not stale_ranks or min(ev_ranks) < min(stale_ranks))
        out[f"current_over_stale@{k}"] = 1.0 if current_first else 0.0
    else:
        out[f"current_over_stale@{k}"] = None
    return out


# --- runner --------------------------------------------------------------------


class ReplayRunner:
    """Replays histories into isolated stores and probes them in order."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        embedder: Embedder,
        work_dir: str | Path,
        candidate: str = "baseline",
        ks: Sequence[int] = (1, 3, 5),
    ) -> None:
        self._cfg = cfg
        self._embedder = embedder
        self._work_dir = Path(work_dir)
        self._candidate = candidate
        self._ks = tuple(sorted(set(ks)))

    def _history_rows(self, h: ReplayHistory) -> tuple[list[dict[str, Any]], int]:
        db_path = self._work_dir / self._candidate / f"{h.history_id}.mind"
        if db_path.exists():
            raise FileExistsError(
                f"{db_path} already exists; each candidate and history replays "
                f"into a fresh store"
            )
        db_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = self._cfg
        db = MindDB(db_path, dim=cfg.storage.vector_dim)
        try:
            store = FrameStore(db, embedder=self._embedder, chunking=cfg.chunking)
            scanner = InjectionScanner(cfg.safety)
            frame_event: dict[int, str] = {}
            visible: list[str] = []
            quarantined = 0
            cursor = 0
            rows: list[dict[str, Any]] = []
            for probe in sorted(h.probes, key=lambda p: (p.after_seq, p.probe_id)):
                while cursor < len(h.events) and h.events[cursor].seq <= probe.after_seq:
                    event = h.events[cursor]
                    result = admit(
                        event.text,
                        gop=f"{h.history_id}:replay",
                        scanner=scanner,
                        store=store,
                        kg=None,
                        cfg=cfg,
                        source="import",
                        event_time=event.at,
                        created_at=event.at,
                    )
                    if result.frame is not None:
                        frame_event.setdefault(result.frame.id, event.event_id)
                    if result.trust == "quarantined":
                        quarantined += 1
                    visible.append(event.event_id)
                    cursor += 1
                not_ready = sorted(
                    fid for fid in frame_event if not store.index_state(fid).ready
                )
                if not_ready:
                    raise ReplayNotReadyError(
                        f"{h.history_id}: frames {not_ready} are not indexed; "
                        f"probe '{probe.probe_id}' was not asked"
                    )
                at = parse_utc(probe.at)
                search = HybridSearch(db, store, cfg, clock=lambda at=at: at)
                result = search.search(probe.question, limit=max(self._ks))
                retrieved = [
                    (frame_event.get(hit.frame.id), hit.frame.content) for hit in result.hits
                ]
                scores: dict[str, float | None] = {}
                for k in self._ks:
                    scores.update(_probe_scores(probe, retrieved, k))
                rows.append(
                    {
                        "history_id": h.history_id,
                        "probe_id": probe.probe_id,
                        "competency": probe.competency,
                        "after_seq": probe.after_seq,
                        "at": probe.at,
                        "visible_events": list(visible),
                        "ready": True,
                        "retrieved": [eid for eid, _ in retrieved],
                        "abstained": result.abstained,
                        "abstention": (
                            result.abstention.as_dict() if result.abstention else None
                        ),
                        "scores": dict(sorted(scores.items())),
                    }
                )
            return rows, quarantined
        finally:
            db.close()

    def run(
        self, histories: Sequence[ReplayHistory], *, entry: BenchmarkEntry, dataset_sha: str
    ) -> dict[str, Any]:
        """Replay every history and return the receipt (no wall-clock fields)."""
        protocol = protocol_id(entry)
        rows: list[dict[str, Any]] = []
        quarantined = 0
        for h in sorted(histories, key=lambda h: h.history_id):
            history_rows, q = self._history_rows(h)
            rows.extend(history_rows)
            quarantined += q

        per_family: dict[str, list[tuple[str, Score]]] = {}
        not_applicable: Counter[str] = Counter()
        for row in rows:
            for family, value in row["scores"].items():
                if value is None:
                    not_applicable[family] += 1
                    continue
                score = Score(value=value, tag=objective_tag(family, dataset_sha, protocol), n=1)
                per_family.setdefault(family, []).append((row["competency"], score))

        registry = MetricRegistry()
        entries: list[dict[str, Any]] = []
        for _family, items in sorted(per_family.items()):
            scopes: dict[str, list[Score]] = {"overall": [s for _, s in items]}
            for competency, s in items:
                scopes.setdefault(f"competency:{competency}", []).append(s)
            for scope, scores in sorted(scopes.items()):
                agg = registry.aggregate(scores)
                entries.append(
                    {
                        "scope": scope,
                        "value": agg.value,
                        "n": agg.n,
                        "tag": dataclasses.asdict(agg.tag),
                    }
                )

        return {
            "label": NOT_OFFICIAL_LABEL if not entry.official else entry.title,
            "official": entry.official,
            "benchmark": manifest_record(entry),
            "dataset_sha": dataset_sha,
            "candidate": self._candidate,
            "embedder": self._embedder_fingerprint(),
            "ks": list(self._ks),
            "clock": "logical: each probe's 'at'; the wall clock is not read",
            "probes": rows,
            "scores": entries,
            "denominators": {
                "histories": len(histories),
                "probes": len(rows),
                "by_competency": dict(sorted(Counter(r["competency"] for r in rows).items())),
                "abstained": sum(1 for r in rows if r["abstained"]),
                "empty_retrieval": sum(1 for r in rows if not r["retrieved"]),
                "not_applicable": dict(sorted(not_applicable.items())),
                "quarantined_events": quarantined,
            },
        }

    def _embedder_fingerprint(self) -> dict[str, Any]:
        return fingerprint_of(self._embedder).as_dict()


def receipt_text(receipt: dict[str, Any]) -> str:
    return json.dumps(receipt, indent=2, sort_keys=True) + "\n"


def receipt_digest(receipt: dict[str, Any]) -> str:
    return hashlib.sha256(receipt_text(receipt).encode("utf-8")).hexdigest()


def run_replay_fixture(
    cfg: AppConfig,
    *,
    benchmark_id: str = "mab_fixture",
    embedder: Embedder,
    work_dir: str | Path,
    out_dir: str | Path | None = None,
    candidate: str = "baseline",
) -> tuple[dict[str, Any], str]:
    """Replay a registered fixture; optionally write ``receipt.json``."""
    entry, histories, sha = load_registered_fixture(benchmark_id)
    runner = ReplayRunner(cfg, embedder=embedder, work_dir=work_dir, candidate=candidate)
    receipt = runner.run(histories, entry=entry, dataset_sha=sha)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "receipt.json").write_text(receipt_text(receipt), encoding="utf-8")
    return receipt, receipt_digest(receipt)


__all__ = [
    "NOT_OFFICIAL_LABEL",
    "REPLAY_FIXTURES",
    "ReplayEvent",
    "ReplayFixtureError",
    "ReplayHistory",
    "ReplayNotReadyError",
    "ReplayProbe",
    "ReplayRunner",
    "TrajectoryEnvironment",
    "load_registered_fixture",
    "load_replay_fixture",
    "receipt_digest",
    "receipt_text",
    "run_replay_fixture",
]
