"""LoCoMo benchmark conversation ingestion into sessions, frames, and KG."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gluomem.embedding.providers import make_embedder
from gluomem.frames.eligibility import is_recall_eligible
from gluomem.frames.store import FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.kg.normalizer import predicate_family
from gluomem.llm.outputs import ExtractionError
from gluomem.storage.db import MindDB
from gluomem.write.cognify import (
    ResolvedGraph,
    cognify_texts,
    retire_conflicting_relations,
)
from gluomem.write.dedup import SemanticDeduper
from gluomem.write.gate import InjectionScanner, admit
from gluomem.write.lanes import Observation, extract_observations, lane_importance

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig
    from gluomem.embedding.base import Embedder
    from gluomem.frames.store import Frame, Importance
    from gluomem.kg.graph import Relation
    from gluomem.llm.outputs import Extractors


@dataclass
class Turn:
    speaker: str
    text: str
    dia_id: str | None = None


@dataclass
class Session:
    index: int
    date_time: datetime
    turns: list[Turn]


@dataclass
class Conversation:
    conv_id: str
    speakers: tuple[str, str]
    sessions: list[Session]


_LOCOMO_DT_FORMATS = (
    "%I:%M %p on %d %B, %Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


def _parse_locomo_datetime(raw: str) -> datetime:
    """Parse a LoCoMo session datetime string against known formats."""
    for fmt in _LOCOMO_DT_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(
        f"unrecognized LoCoMo session datetime {raw!r}; known formats: "
        f"{', '.join(_LOCOMO_DT_FORMATS)}"
    )


def load_locomo(path: str | Path) -> list[Conversation]:
    """Parse a LoCoMo-format JSON file into typed conversations."""
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of LoCoMo samples")

    conversations: list[Conversation] = []
    for i, sample in enumerate(data):
        conv_id = sample.get("sample_id")
        conv = sample.get("conversation")
        if not conv_id or not isinstance(conv, dict):
            raise ValueError(f"{path}: sample {i} missing sample_id/conversation")
        speaker_a = conv.get("speaker_a")
        speaker_b = conv.get("speaker_b")
        if not speaker_a or not speaker_b:
            raise ValueError(f"{path}: sample '{conv_id}' missing speaker_a/b")

        sessions: list[Session] = []
        n = 1
        while f"session_{n}" in conv:
            raw_turns = conv[f"session_{n}"]
            raw_dt = conv.get(f"session_{n}_date_time")
            if raw_dt is None:
                raise ValueError(
                    f"{path}: sample '{conv_id}' session_{n} has no date_time"
                )
            turns: list[Turn] = []
            for t in raw_turns:
                text = t.get("text", "")
                caption = t.get("blip_caption")
                if caption:
                    text = f"{text} [shares photo: {caption}]".strip()
                turns.append(
                    Turn(
                        speaker=t.get("speaker", "unknown"),
                        text=text,
                        dia_id=t.get("dia_id"),
                    )
                )
            sessions.append(
                Session(index=n, date_time=_parse_locomo_datetime(raw_dt), turns=turns)
            )
            n += 1
        if not sessions:
            raise ValueError(f"{path}: sample '{conv_id}' has no sessions")
        conversations.append(
            Conversation(
                conv_id=str(conv_id),
                speakers=(str(speaker_a), str(speaker_b)),
                sessions=sessions,
            )
        )
    return conversations


@dataclass
class IngestReport:
    conversations: int = 0
    sessions: int = 0
    turns: int = 0
    i_frames: int = 0
    p_frames: int = 0
    b_frames: int = 0
    deduped: int = 0
    semantic_deduped: int = 0
    quarantined: int = 0
    rejected: int = 0
    entities: int = 0
    relations: int = 0
    retired_edges: int = 0
    demoted: int = 0
    extraction_errors: list[str] = field(default_factory=list)


def _lane_gop(conv_id: str, obs: Observation) -> str:
    if obs.lane == "fact":
        return f"{conv_id}:facts"
    if obs.lane == "event":
        return f"{conv_id}:events"
    return f"{conv_id}:profile:{obs.speaker}"


def find_subject_frames(
    entity_ids: list[int],
    gop: str,
    *,
    kg: KnowledgeGraph,
    store: FrameStore,
    cfg: AppConfig,
) -> list[Frame]:
    """Return non-deprecated I-frames in ``gop`` sharing >= threshold entities."""
    if not entity_ids:
        return []
    overlap: dict[int, int] = {}
    for eid in entity_ids:
        for fid in kg.frame_ids_for_entity(eid):
            overlap[fid] = overlap.get(fid, 0) + 1

    candidate_ids = [
        fid
        for fid, count in overlap.items()
        if count >= cfg.write.subject.min_entity_overlap
    ]
    if not candidate_ids:
        return []
    frames_by_id = store.get_frames(candidate_ids)

    scored: list[tuple[int, int, Frame]] = []
    for fid in candidate_ids:
        frame = frames_by_id.get(fid)
        if frame is None or frame.gop_id != gop or frame.frame_type != "I":
            continue
        if frame.importance == "deprecated":
            continue
        scored.append((overlap[fid], frame.id, frame))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [frame for _, _, frame in scored]


def find_base_frame(
    entity_ids: list[int],
    gop: str,
    *,
    kg: KnowledgeGraph,
    store: FrameStore,
    cfg: AppConfig,
) -> Frame | None:
    """Return the highest-overlap same-subject base I-frame in ``gop``."""
    frames = find_subject_frames(entity_ids, gop, kg=kg, store=store, cfg=cfg)
    return frames[0] if frames else None


def find_edge_conflict_frames(
    relations: list[Relation],
    gop: str,
    *,
    kg: KnowledgeGraph,
    store: FrameStore,
    cfg: AppConfig,
) -> list[Frame]:
    """Return contradiction candidate I-frames in ``gop`` from KG edge conflicts."""
    frames: list[Frame] = []
    seen: set[int] = set()
    for rel in relations:
        for rtype in predicate_family(rel.relation_type, cfg.kg):
            for existing in kg.find_valid_relations(rel.source_id, rtype):
                if existing.id == rel.id or existing.target_id == rel.target_id:
                    continue
                source_fids = set(kg.frame_ids_for_entity(rel.source_id))
                target_fids = kg.frame_ids_for_entity(existing.target_id)
                fids = [f for f in target_fids if f in source_fids] or target_fids
                ordered = [f for f in sorted(fids, reverse=True) if f not in seen]
                if not ordered:
                    continue
                frames_by_id = store.get_frames(ordered)
                for fid in ordered:
                    seen.add(fid)
                    frame = frames_by_id.get(fid)
                    if frame is None or frame.gop_id != gop or frame.frame_type != "I":
                        continue
                    if frame.importance == "deprecated":
                        continue
                    frames.append(frame)
    return frames


def _count_frame(result, report: IngestReport) -> None:
    if not result.admitted:
        report.rejected += 1
        return
    if result.trust == "quarantined":
        report.quarantined += 1
    if result.deduped:
        report.deduped += 1
        return
    if result.semantic_deduped:
        report.semantic_deduped += 1
        return
    if result.superseded_id is not None:
        report.demoted += 1
    frame = result.frame
    if frame.frame_type == "I":
        report.i_frames += 1
    elif frame.frame_type == "P":
        report.p_frames += 1
    else:
        report.b_frames += 1


def _event_time_for(obs: Observation, session_iso: str) -> str | None:
    if obs.lane != "event":
        return None
    return obs.event_time if obs.event_time else session_iso


def _admit_observation(
    obs: Observation,
    *,
    conv_id: str,
    session_iso: str,
    graph: ResolvedGraph,
    facade: Extractors,
    scanner: InjectionScanner,
    store: FrameStore,
    kg: KnowledgeGraph,
    cfg: AppConfig,
    report: IngestReport,
    deduper: SemanticDeduper | None = None,
) -> None:
    gop = _lane_gop(conv_id, obs)
    event_time = _event_time_for(obs, session_iso)
    created_at = session_iso if cfg.write.backdate_created_at else None

    report.entities += len(graph.entity_ids)
    report.relations += len(graph.relation_ids)

    obs_deduper = (
        deduper if (deduper is not None and obs.lane in cfg.write.dedup.lanes) else None
    )

    subject_frames = find_subject_frames(
        graph.entity_ids, gop, kg=kg, store=store, cfg=cfg
    )
    base = subject_frames[0] if subject_frames else None
    importance = lane_importance(obs.lane, cfg)
    admit_kwargs: dict[str, Any] = dict(
        gop=gop,
        scanner=scanner,
        store=store,
        kg=kg,
        cfg=cfg,
        source="agent_inferred",
        entity_ids=graph.entity_ids,
        created_at=created_at,
    )

    mode = cfg.write.frame_model
    candidates: list[Frame] = []
    if mode != "i_only" and cfg.write.contradiction.enabled:
        trigger = cfg.write.subject.trigger
        if trigger in ("entity_overlap", "both"):
            candidates.extend(subject_frames)
        if trigger in ("kg_edge_conflict", "both"):
            for frame in find_edge_conflict_frames(
                graph.relations, gop, kg=kg, store=store, cfg=cfg
            ):
                if all(frame.id != c.id for c in candidates):
                    candidates.append(frame)

    verdict = None
    hit: Frame | None = None
    for cand in candidates[: cfg.write.contradiction.top_k]:
        v = facade.check_contradiction(obs.text, cand.content)
        if v is not None and v.is_contradiction:
            verdict, hit = v, cand
            break

    if verdict is None or hit is None:
        if mode == "ipb" and base is not None:
            result = admit(
                obs.text,
                frame_type="P",
                base_id=base.id,
                importance=importance,
                event_time=event_time,
                deduper=obs_deduper,
                **admit_kwargs,
            )
        else:
            result = admit(
                obs.text,
                importance=importance,
                event_time=event_time,
                deduper=obs_deduper,
                **admit_kwargs,
            )
        _count_frame(result, report)
        return

    new_i = admit(
        obs.text,
        importance=importance,
        event_time=event_time,
        **admit_kwargs,
    )
    _count_frame(new_i, report)
    if new_i.frame is None or not is_recall_eligible(new_i.trust):
        return

    if mode == "ipb":
        b = admit(
            verdict.description or f"contradicts frame {hit.id}",
            frame_type="B",
            base_id=hit.id,
            references=[hit.id, new_i.frame.id],
            event_time=event_time,
            **admit_kwargs,
        )
        _count_frame(b, report)
    else:
        stale_frames = list(subject_frames)
        if all(hit.id != f.id for f in stale_frames):
            stale_frames.append(hit)
        for stale in stale_frames:
            store.set_importance(stale.id, "deprecated")
            report.demoted += 1
            for child in store.frames_based_on(stale.id):
                if child.frame_type == "P":
                    store.set_importance(child.id, "deprecated")
                    report.demoted += 1

    if cfg.write.cognify.enabled and cfg.write.cognify.retire_conflicting_edges:
        report.retired_edges += len(
            retire_conflicting_relations(graph.relations, kg=kg, kg_cfg=cfg.kg)
        )


def ingest_conversation(
    conv: Conversation,
    *,
    facade: Extractors,
    scanner: InjectionScanner,
    store: FrameStore,
    kg: KnowledgeGraph,
    cfg: AppConfig,
) -> IngestReport:
    """Ingest one conversation through the admission gate."""
    report = IngestReport(conversations=1, sessions=len(conv.sessions))

    deduper = (
        SemanticDeduper(
            store.embedder, threshold=cfg.write.dedup.similarity_threshold
        )
        if cfg.write.dedup.mode == "semantic"
        else None
    )

    for session in conv.sessions:
        session_iso = session.date_time.isoformat(timespec="seconds")
        gop_raw = f"{conv.conv_id}:session:{session.index}"
        clean_turns: list[Turn] = []

        for turn in session.turns:
            report.turns += 1
            content = f"{turn.speaker}: {turn.text}"
            if cfg.write.raw_turns.enabled:
                raw_importance: Importance = cfg.write.raw_turns.importance  # type: ignore[assignment]
                result = admit(
                    content,
                    gop=gop_raw,
                    scanner=scanner,
                    store=store,
                    kg=kg,
                    cfg=cfg,
                    importance=raw_importance,
                    source="import",
                    event_time=session_iso,
                    created_at=(
                        session_iso if cfg.write.backdate_created_at else None
                    ),
                )
                _count_frame(result, report)
                excluded = (not result.admitted) or result.trust == "quarantined"
            else:
                verdict = scanner.scan(content, context="import")
                excluded = verdict.score >= cfg.safety.block_threshold
                if excluded:
                    report.quarantined += 1
            if not excluded:
                clean_turns.append(turn)

        session_text = "\n".join(f"{t.speaker}: {t.text}" for t in clean_turns)
        speaker_texts = {
            speaker: "\n".join(t.text for t in clean_turns if t.speaker == speaker)
            for speaker in conv.speakers
        }
        try:
            observations = extract_observations(
                session_text,
                facade=facade,
                reference_date=session.date_time,
                speakers=conv.speakers,
                speaker_texts=speaker_texts,
                cfg=cfg,
            )
        except ExtractionError as exc:
            report.extraction_errors.append(
                f"{conv.conv_id}:session:{session.index}: {exc}"
            )
            continue

        flagged = [
            scanner.scan(obs.text, context="agent_inferred").score
            >= cfg.safety.block_threshold
            for obs in observations
        ]
        clean_idx = [i for i, f in enumerate(flagged) if not f]
        clean_graphs = cognify_texts(
            [observations[i].text for i in clean_idx],
            facade=facade,
            kg=kg,
            cfg=cfg,
            valid_from_list=[
                _event_time_for(observations[i], session_iso) for i in clean_idx
            ],
            max_workers=cfg.write.parallelism,
        )
        graphs: list[ResolvedGraph | Exception] = [
            ResolvedGraph() for _ in observations
        ]
        for i, g in zip(clean_idx, clean_graphs, strict=True):
            graphs[i] = g

        touched: set[str] = set()
        for obs, graph in zip(observations, graphs, strict=True):
            if isinstance(graph, Exception):
                report.extraction_errors.append(
                    f"{conv.conv_id}:{obs.lane}: cognify failed: {graph}"
                )
                graph = ResolvedGraph()
            try:
                _admit_observation(
                    obs,
                    conv_id=conv.conv_id,
                    session_iso=session_iso,
                    graph=graph,
                    facade=facade,
                    scanner=scanner,
                    store=store,
                    kg=kg,
                    cfg=cfg,
                    report=report,
                    deduper=deduper,
                )
            except ExtractionError as exc:
                report.extraction_errors.append(
                    f"{conv.conv_id}:{obs.lane}: {exc}"
                )
                continue
            touched.add(_lane_gop(conv.conv_id, obs))

        for gop in sorted(touched):
            store.compact(gop, cfg)

    return report


def ingest_path(
    path: str | Path,
    cfg: AppConfig,
    *,
    facade: Extractors,
    db_path: str | Path | None = None,
    embedder: Embedder | None = None,
) -> IngestReport:
    """Wire storage and gate, then ingest every conversation in a LoCoMo file."""
    conversations = load_locomo(path)
    db = MindDB(db_path or cfg.storage.path, dim=cfg.storage.vector_dim)
    if embedder is None:
        embedder = make_embedder(cfg)
    store = FrameStore(db, embedder=embedder, chunking=cfg.chunking)
    kg = KnowledgeGraph(db)
    scanner = InjectionScanner(cfg.safety)

    total = IngestReport()
    for conv in conversations:
        report = ingest_conversation(
            conv, facade=facade, scanner=scanner, store=store, kg=kg, cfg=cfg
        )
        for attr in (
            "conversations",
            "sessions",
            "turns",
            "i_frames",
            "p_frames",
            "b_frames",
            "deduped",
            "semantic_deduped",
            "quarantined",
            "rejected",
            "entities",
            "relations",
            "retired_edges",
            "demoted",
        ):
            setattr(total, attr, getattr(total, attr) + getattr(report, attr))
        total.extraction_errors.extend(report.extraction_errors)
    return total
