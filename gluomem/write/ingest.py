"""gluomem.write.ingest — LoCoMo benchmark conversations → sessions → frames.

Flow per session: admit raw turns → lane extraction over the
non-quarantined text → per-observation frame-model dispatch → compaction of
touched GOPs. Every store write goes through ``gate.admit()``.

GOP layout (dedup/reconstruction semantics drive the split):

- raw dialog turns → ``{conv_id}:session:{n}`` (``event_time`` = session dt);
- fact lane        → ``{conv_id}:facts``   (cross-session dedup);
- event lane       → ``{conv_id}:events``;
- profile lane     → ``{conv_id}:profile:{speaker}`` (``reconstruct_state``
  = "current state of this speaker"; the natural P-frame habitat).

Frame-model dispatch (``write.frame_model``):

- ``ipb``       — same-subject delta → P on the base; contradiction → new I
  (the new truth) + B correction on the base + conflicting KG edges retired;
- ``i_only``    — every observation a fresh I; the contradiction detector is
  NEVER invoked;
- ``overwrite`` — new I; on contradiction the superseded base I and its
  P-refinements are demoted to ``importance='deprecated'`` (demote-and-add:
  append-only/dedup/index invariants hold, retrieval down-weights the old
  truth) and conflicting KG edges retired. No P/B rows ever.

Same-subject detection: cognify first, then the base is the non-deprecated
I-frame in the target lane GOP sharing ≥ ``write.subject.min_entity_overlap``
KG entities with the observation (max overlap, then most recent).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gluomem.embedding.providers import MockEmbedder, make_embedder
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


# --- LoCoMo parsing ----------------------------------------------------------


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


# Formats observed in LoCoMo dumps; strict list, fail-loud on anything else.
_LOCOMO_DT_FORMATS = (
    "%I:%M %p on %d %B, %Y",  # "1:56 pm on 8 May, 2023"
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


def _parse_locomo_datetime(raw: str) -> datetime:
    """Strict small-format-list parser; ValueError on miss, never a guess."""
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
    """Parse a LoCoMo-format JSON file into typed conversations.

    Expected shape per sample: ``sample_id`` + ``conversation`` holding
    ``speaker_a``/``speaker_b``, ``session_{n}`` (list of
    ``{"speaker", "dia_id", "text"[, "blip_caption"]}``) and
    ``session_{n}_date_time``. Image turns fold their caption into the text.
    Malformed input raises ``ValueError`` (fail loud).
    """
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


# --- ingest orchestration ------------------------------------------------------


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
    # Extraction failures are recorded, never silently swallowed; the ingest
    # degrades to raw-turns for the affected session instead of aborting the
    # whole run.
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
    """Non-deprecated I-frames in ``gop`` sharing ≥ threshold entities.

    Ordered by (overlap, id) DESCENDING — the head is the same-subject base
    (most-shared, then most recent). The full list is the subject CLUSTER the
    ``overwrite`` mode demotes wholesale on contradiction.
    """
    if not entity_ids:
        return []
    overlap: dict[int, int] = {}
    for eid in entity_ids:
        for fid in kg.frame_ids_for_entity(eid):
            overlap[fid] = overlap.get(fid, 0) + 1

    scored: list[tuple[int, int, Frame]] = []
    for fid, count in overlap.items():
        if count < cfg.write.subject.min_entity_overlap:
            continue
        frame = store.get_frame(fid)
        if frame.gop_id != gop or frame.frame_type != "I":
            continue
        if frame.importance == "deprecated":
            continue
        scored.append((count, frame.id, frame))
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
    """Same-subject base: max entity overlap in ``gop``, threshold from config."""
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
    """Contradiction candidates from KG edge conflicts (``kg_edge_conflict``).

    A freshly-resolved relation ``(s, T, t_new)`` whose source already holds a
    valid same-family edge ``(s, T', t_old != t_new)`` flags the OLD edge's
    frames: non-deprecated I-frames in ``gop`` linked to BOTH ``s`` and
    ``t_old`` (falling back to ``t_old`` alone when the intersection is
    empty), newest first. This catches update-type facts entity overlap
    structurally misses — ``{Bob, Seattle}`` vs stored ``{Bob, Boston}``
    share only ``Bob``, but the ``LIVES_IN``-family edge conflict is exact.
    """
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
                for fid in sorted(fids, reverse=True):
                    if fid in seen:
                        continue
                    seen.add(fid)
                    frame = store.get_frame(fid)
                    if frame.gop_id != gop or frame.frame_type != "I":
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
        report.demoted += 1  # keep_longest demoted the shorter stored near-dup
    frame = result.frame
    if frame.frame_type == "I":
        report.i_frames += 1
    elif frame.frame_type == "P":
        report.p_frames += 1
    else:
        report.b_frames += 1


def _event_time_for(obs: Observation, session_iso: str) -> str | None:
    """Event-lane frames carry the resolved (or session-fallback) event time."""
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

    # Semantic dedup applies only to eligible lanes (events excluded — same-text
    # events at different times are distinct; raw turns / B-frames never probe).
    # The contradiction path's new-truth I and its B-frame keep deduper=None so a
    # correction is never dropped as a near-duplicate.
    obs_deduper = deduper if (deduper is not None and obs.lane in cfg.write.dedup.lanes) else None

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

    # Contradiction candidates, per write.subject.trigger. P-frame attachment
    # stays entity-overlap-only regardless: the trigger nominates frames for
    # the CONTRADICTION check, never for delta attachment, so kg_edge_conflict
    # cannot inflate P-frame attachment.
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
        else:  # fresh I: i_only, no same-subject base, or overwrite delta
            result = admit(
                obs.text,
                importance=importance,
                event_time=event_time,
                deduper=obs_deduper,
                **admit_kwargs,
            )
        _count_frame(result, report)
        return

    # Contradiction path: the new truth is admitted as I in BOTH modes.
    new_i = admit(
        obs.text,
        importance=importance,
        event_time=event_time,
        **admit_kwargs,
    )
    _count_frame(new_i, report)
    if new_i.frame is None or not is_recall_eligible(new_i.trust):
        # Ineligible "new truth" must not correct, demote or retire anything.
        return

    if mode == "ipb":
        if new_i.frame is not None:
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
        # overwrite: demote the WHOLE superseded subject cluster (in this
        # mode earlier deltas were admitted as sibling I-frames, so demoting
        # only the tie-break base would leave stale truth at full weight)
        # plus any P-refinements hanging off the demoted frames. An edge-
        # conflict hit outside the overlap cluster is part of the stale truth.
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
    """Ingest one conversation; every write goes through the gate."""
    report = IngestReport(conversations=1, sessions=len(conv.sessions))

    # Semantic dedup is per-conversation (fresh cache; same-GOP scope). Built
    # only under mode='semantic' — mode='exact' passes deduper=None everywhere.
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
            continue  # raw turns are already in; degrade, don't abort

        # Cognify: LLM calls fan out (write.parallelism); KG resolution is
        # applied sequentially in observation order (identical ids to a
        # sequential run). Per-observation failures are recorded, not fatal.
        # Observations the gate will quarantine or reject never reach cognify:
        # their entities/relations would otherwise wire flagged text into the
        # KG and nominate contradiction targets before admission.
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
    """Wire storage + gate and ingest every conversation in a LoCoMo file."""
    conversations = load_locomo(path)
    db = MindDB(db_path or cfg.storage.path, dim=cfg.storage.vector_dim)
    if embedder is None:
        embedder = (
            MockEmbedder(dim=db.dim)
            if cfg.embedding.model == "mock"
            else make_embedder(cfg)
        )
    store = FrameStore(db, embedder=embedder, chunking=cfg.chunking)
    kg = KnowledgeGraph(db)
    scanner = InjectionScanner(cfg.safety)

    total = IngestReport()
    for conv in conversations:
        report = ingest_conversation(
            conv, facade=facade, scanner=scanner, store=store, kg=kg, cfg=cfg
        )
        for attr in (
            "conversations", "sessions", "turns", "i_frames", "p_frames",
            "b_frames", "deduped", "semantic_deduped", "quarantined", "rejected",
            "entities", "relations", "retired_edges", "demoted",
        ):
            setattr(total, attr, getattr(total, attr) + getattr(report, attr))
        total.extraction_errors.extend(report.extraction_errors)
    return total
