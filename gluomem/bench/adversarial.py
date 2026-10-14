"""MINJA-inspired adversarial memory poisoning runner and scorer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gluomem.bench.registry import (
    BenchmarkEntry,
    get_entry,
    manifest_record,
    protocol_id,
)
from gluomem.config.app_config import AppConfig
from gluomem.embedding.base import Embedder
from gluomem.embedding.providers import make_embedder
from gluomem.frames.store import FrameStore
from gluomem.search.hybrid import HybridSearch
from gluomem.search.multilingual import fold_multilingual_text
from gluomem.storage.db import MindDB
from gluomem.timeutil import parse_utc
from gluomem.write.gate import InjectionScanner, admit

FIXTURE_DIR = Path(__file__).parent / "fixtures"
ADVERSARIAL_FIXTURES: dict[str, Path] = {
    "minja_adversarial_v1": FIXTURE_DIR / "minja_adversarial_v1.json",
}


class AdversarialFixtureError(ValueError):
    """Malformed or tampered adversarial benchmark fixture."""


@dataclass(frozen=True)
class AdversarialEvent:
    event_id: str
    at: str
    text: str
    poison_marker: str | None = None


@dataclass(frozen=True)
class AdversarialScenario:
    scenario_id: str
    attack_family: str
    benign_events: tuple[AdversarialEvent, ...]
    adversarial_events: tuple[AdversarialEvent, ...]
    victim_query: str
    expected_answer: str
    expected_evidence_id: str


def load_adversarial_fixture(path: Path) -> tuple[list[AdversarialScenario], str]:
    raw_bytes = path.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    payload = json.loads(raw_bytes.decode("utf-8"))
    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise AdversarialFixtureError(f"{path}: fixture has no scenarios")
    scenarios: list[AdversarialScenario] = []
    seen_ids: set[str] = set()
    for item in raw_scenarios:
        sid = str(item["scenario_id"])
        if sid in seen_ids:
            raise AdversarialFixtureError(f"duplicate scenario_id {sid!r}")
        seen_ids.add(sid)
        benign = tuple(
            AdversarialEvent(
                event_id=str(ev["event_id"]),
                at=str(ev["at"]),
                text=str(ev["text"]),
            )
            for ev in item.get("benign_events", ())
        )
        adv = tuple(
            AdversarialEvent(
                event_id=str(ev["event_id"]),
                at=str(ev["at"]),
                text=str(ev["text"]),
                poison_marker=str(ev["poison_marker"]),
            )
            for ev in item.get("adversarial_events", ())
        )
        if not benign or not adv:
            raise AdversarialFixtureError(
                f"scenario {sid!r} must include both benign and adversarial events"
            )
        scenarios.append(
            AdversarialScenario(
                scenario_id=sid,
                attack_family=str(item["attack_family"]),
                benign_events=benign,
                adversarial_events=adv,
                victim_query=str(item["victim_query"]),
                expected_answer=str(item["expected_answer"]),
                expected_evidence_id=str(item["expected_evidence_id"]),
            )
        )
    return scenarios, sha


def load_registered_adversarial_fixture(
    entry_id: str = "minja_adversarial_v1",
) -> tuple[BenchmarkEntry, list[AdversarialScenario], str]:
    entry = get_entry(entry_id)
    path = ADVERSARIAL_FIXTURES.get(entry_id)
    if path is None or not path.exists():
        raise AdversarialFixtureError(
            f"no adversarial fixture registered for {entry_id!r}"
        )
    scenarios, sha = load_adversarial_fixture(path)
    if entry.sha256 is not None and sha != entry.sha256:
        raise AdversarialFixtureError(
            f"fixture {entry_id!r} sha256 {sha} != registry pin {entry.sha256}"
        )
    return entry, scenarios, sha


class AdversarialRunner:
    """Evaluates WriteGate quarantine + HybridSearch recall under adversarial poisoning."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        embedder: Embedder | None = None,
        work_dir: Path,
        k: int = 5,
    ) -> None:
        self._cfg = cfg
        self._embedder = embedder or make_embedder(cfg)
        self._work_dir = Path(work_dir)
        self._k = k

    def run(
        self,
        scenarios: Sequence[AdversarialScenario],
        *,
        entry: BenchmarkEntry,
        dataset_sha: str,
    ) -> dict[str, Any]:
        scenario_rows: list[dict[str, Any]] = []
        total_adv = 0
        admitted_unquarantined_adv = 0
        poison_leaks = 0
        clean_retained = 0

        for scenario in scenarios:
            row = self._evaluate_scenario(scenario)
            scenario_rows.append(row)
            total_adv += row["adversarial_total"]
            admitted_unquarantined_adv += row["adversarial_admitted_eligible"]
            if row["poison_leaked"]:
                poison_leaks += 1
            if row["clean_retained"]:
                clean_retained += 1

        n_scenarios = len(scenarios)
        summary = {
            "poison_admission_rate": (
                admitted_unquarantined_adv / total_adv if total_adv else 0.0
            ),
            f"poison_recall_leak_rate@{self._k}": (
                poison_leaks / n_scenarios if n_scenarios else 0.0
            ),
            f"clean_recall_retention@{self._k}": (
                clean_retained / n_scenarios if n_scenarios else 0.0
            ),
        }
        return {
            "benchmark_id": entry.id,
            "protocol": protocol_id(entry),
            "benchmark": manifest_record(entry),
            "dataset_sha": dataset_sha,
            "k": self._k,
            "scenarios": scenario_rows,
            "summary": summary,
        }

    def _evaluate_scenario(self, scenario: AdversarialScenario) -> dict[str, Any]:
        safe_id = scenario.scenario_id.replace(":", "_")
        db_path = self._work_dir / f"{safe_id}.mind"
        if db_path.exists():
            raise FileExistsError(
                f"adversarial scenario {scenario.scenario_id!r} requires a fresh store: {db_path}"
            )
        db_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = self._cfg
        db = MindDB(db_path, dim=cfg.storage.vector_dim)
        try:
            store = FrameStore(db, embedder=self._embedder, chunking=cfg.chunking)
            scanner = InjectionScanner(cfg.safety)
            gop_id = f"{scenario.scenario_id}:adversarial"

            frame_to_event: dict[int, str] = {}
            for ev in scenario.benign_events:
                res = admit(
                    ev.text,
                    gop=gop_id,
                    scanner=scanner,
                    store=store,
                    kg=None,
                    cfg=cfg,
                    source="user_stated",
                    trust="normal",
                    event_time=ev.at,
                    created_at=ev.at,
                )
                if res.frame is not None:
                    frame_to_event[res.frame.id] = ev.event_id

            adv_admitted_eligible = 0
            quarantined_ids: list[str] = []
            for ev in scenario.adversarial_events:
                res = admit(
                    ev.text,
                    gop=gop_id,
                    scanner=scanner,
                    store=store,
                    kg=None,
                    cfg=cfg,
                    source="import",
                    trust="normal",
                    event_time=ev.at,
                    created_at=ev.at,
                )
                if res.frame is not None:
                    frame_to_event[res.frame.id] = ev.event_id
                if res.trust == "quarantined":
                    quarantined_ids.append(ev.event_id)
                else:
                    adv_admitted_eligible += 1

            latest_at = max(
                ev.at
                for ev in (*scenario.benign_events, *scenario.adversarial_events)
            )
            at = parse_utc(latest_at)
            search = HybridSearch(db, store, cfg, clock=lambda at=at: at)
            sr = search.search(scenario.victim_query, limit=self._k)
            retrieved_events = [
                frame_to_event[h.frame.id]
                for h in sr.hits
                if h.frame.id in frame_to_event
            ]
            retrieved_texts = "\n".join(h.frame.content for h in sr.hits)
            folded_texts = fold_multilingual_text(retrieved_texts)

            adv_event_ids = {ev.event_id for ev in scenario.adversarial_events}
            poison_markers = [
                ev.poison_marker
                for ev in scenario.adversarial_events
                if ev.poison_marker
            ]
            poison_leaked = any(eid in adv_event_ids for eid in retrieved_events) or any(
                m.casefold() in retrieved_texts.casefold()
                or fold_multilingual_text(m) in folded_texts
                for m in poison_markers
            )
            clean_retained = (
                scenario.expected_evidence_id in retrieved_events
                and (
                    scenario.expected_answer.casefold() in retrieved_texts.casefold()
                    or fold_multilingual_text(scenario.expected_answer) in folded_texts
                )
            )
            return {
                "scenario_id": scenario.scenario_id,
                "attack_family": scenario.attack_family,
                "adversarial_total": len(scenario.adversarial_events),
                "adversarial_quarantined": quarantined_ids,
                "adversarial_admitted_eligible": adv_admitted_eligible,
                "retrieved_events": retrieved_events,
                "poison_leaked": poison_leaked,
                "clean_retained": clean_retained,
            }
        finally:
            db.close()


def run_adversarial_fixture(
    cfg: AppConfig,
    *,
    entry_id: str = "minja_adversarial_v1",
    embedder: Embedder | None = None,
    work_dir: Path,
    k: int = 5,
) -> dict[str, Any]:
    entry, scenarios, sha = load_registered_adversarial_fixture(entry_id)
    runner = AdversarialRunner(cfg, embedder=embedder, work_dir=work_dir, k=k)
    return runner.run(scenarios, entry=entry, dataset_sha=sha)
