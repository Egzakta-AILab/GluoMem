"""Current usable evidence, distinct from raw ranked search.

A superseded row can establish identity, never payload eligibility. No history
retention or as-of policy is implied by a seed's event-time match window.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from gluomem.client.models import (
    EvidenceGroupModel,
    EvidenceLinkModel,
    EvidenceRefusalModel,
    LaneDiagnosticsModel,
    MemoryFrameModel,
    ProvenanceModel,
    RecallHitModel,
    RecallResponse,
    RetrievalDiagnosticsModel,
    TemporalEvidenceModel,
)
from gluomem.frames.eligibility import is_recall_eligible
from gluomem.frames.generation import current_scope_generation
from gluomem.frames.source_support import retained_support
from gluomem.frames.temporal import TemporalPolicy, TemporalQuery, TemporalStore
from gluomem.storage.scope import eligible_sql, scope_sql
from gluomem.timeutil import now_iso as format_now
from gluomem.timeutil import parse_utc

if TYPE_CHECKING:
    from gluomem.frames.store import Frame, FrameStore
    from gluomem.search.hybrid import HybridSearch
    from gluomem.storage.scope import MemoryScope


def _observed_before(value: str | None, limit: str) -> bool:
    if value is None:
        return False
    try:
        return parse_utc(value) <= parse_utc(limit)
    except ValueError:
        return False


def _known_payload(store: FrameStore, frame_id: int, scope: MemoryScope, known_at: str) -> bool:
    rows = store.db.fetchall(
        f"SELECT observed_at FROM memory_provenance WHERE frame_id = ? AND {scope_sql()}",
        (frame_id, *scope.as_tuple()),
    )
    return any(_observed_before(row["observed_at"], known_at) for row in rows)


def evidence_frames(
    store: FrameStore,
    frames: list[Frame],
    scope: MemoryScope,
    known_at: str | None = None,
) -> list[MemoryFrameModel]:
    """Bind attribution to persisted scoped envelopes, never generated role text."""
    models = [MemoryFrameModel.model_validate(frame) for frame in frames]
    if not models:
        return models
    ids = [frame.id for frame in models]
    rows = store.db.fetchall(
        f"SELECT p.*, se.payload_json, se.created_at AS source_observed_at, "
        "se.created_at AS source_created_at, se.payload_hash, se.event_id, se.revision, "
        "m.status AS frame_status, m.content_hash AS frame_content_hash, "
        "r.source_ref AS withdrawn "
        f"FROM memory_provenance p LEFT JOIN source_events se "
        f"ON se.id = p.source_event_id AND {scope_sql('se')} "
        "AND se.source_ref = p.source_ref AND se.source_type = p.source_type "
        f"JOIN memory_frames m ON m.id = p.frame_id AND {scope_sql('m')} "
        f"LEFT JOIN source_retractions r ON {scope_sql('r')} AND r.source_ref = p.source_ref "
        f"WHERE {scope_sql('p')} AND p.frame_id IN ({','.join('?' for _ in ids)}) "
        "ORDER BY p.id",
        (*scope.as_tuple(), *scope.as_tuple(), *scope.as_tuple(), *scope.as_tuple(), *ids),
    )
    witnesses = {
        int(w["provenance_id"]): dict(w)
        for w in store.db.fetchall(
            f"SELECT s.* FROM memory_source_support s JOIN memory_provenance p "
            f"ON p.id = s.provenance_id WHERE {scope_sql('s')} AND {scope_sql('p')} "
            f"AND p.frame_id IN ({','.join('?' for _ in ids)})",
            (*scope.as_tuple(), *scope.as_tuple(), *ids),
        )
    }
    by_id = {model.id: model for model in models}
    for row in rows:
        data = dict(row)
        record = ProvenanceModel.model_validate(data)
        if known_at is not None and not _observed_before(record.observed_at, known_at):
            continue
        if record.observed_at is None:
            record.observed_at = data["source_observed_at"]
        try:
            payload = json.loads(data["payload_json"] or "{}")
        except (TypeError, ValueError):
            payload = {}
        messages = payload.get("messages", []) if isinstance(payload, dict) else []
        source_text = None
        if record.message_id is not None:
            matches = [
                message
                for message in messages
                if isinstance(message, dict) and message.get("message_id") == record.message_id
            ]
            if len(matches) == 1:
                message = matches[0]
                record.message_role = message.get("role")
                record.message_time = message.get("timestamp")
                source_text = message.get("content")
        elif isinstance(payload, dict) and not messages:
            source_text = payload.get("content")
        record.span_verified = bool(
            record.verbatim_span
            and isinstance(source_text, str)
            and record.verbatim_span in source_text
        )
        if record.span_verified:
            record.verification_basis = "raw-source"
        if not record.span_verified:
            support = retained_support(data, witnesses.get(record.id))
            if support is not None:
                record.message_role, record.message_time = support
                record.verification_basis = "retained-fragment"
                record.span_verified = True
        if not record.span_verified:
            record.verbatim_span = None
        by_id[int(data["frame_id"])].provenance.append(record)
    return models


def render_group(group: EvidenceGroupModel, frames: dict[int, MemoryFrameModel]) -> str:
    """One JSON line per atomic group; whole-line packers cannot cut dependencies."""
    fields = {
        "id",
        "frame_type",
        "gop_id",
        "t",
        "base_frame_id",
        "content",
        "source",
        "trust",
        "kind",
        "status",
        "event_time",
        "created_at",
        "expires_at",
        "pinned",
        "verified_by_user",
        "confidence",
        "supersedes",
        "superseded_by",
        "provenance",
    }
    payload = {
        "view": "current",
        **group.model_dump(),
        "frames": [frames[fid].model_dump(include=fields) for fid in group.frame_ids],
    }
    return "[evidence] " + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _resolve_seed(
    store: FrameStore,
    seed: Frame,
    scope: MemoryScope,
    now: str,
    budget: int,
    query: TemporalQuery,
    policy: TemporalPolicy,
) -> tuple[list[Frame], list[EvidenceLinkModel], list[TemporalEvidenceModel]]:
    frames: dict[int, Frame] = {}
    links: list[EvidenceLinkModel] = []
    visiting: set[int] = set()
    temporal = TemporalStore(store)
    selections: list[TemporalEvidenceModel] = []

    def select(frame: Frame, *, root: bool = False) -> Frame:
        if any(item.selected_frame_id == frame.id for item in selections):
            return frame
        if frame.frame_type != "I":
            return frame
        resolution = temporal.resolve(frame.id, scope=scope, policy=policy, query=query)
        if resolution.status == "unmanaged":
            if query.as_of is not None or query.known_at is not None:
                raise ValueError("temporal_unmanaged")
            return frame
        if resolution.status != "resolved" or resolution.frame_id is None:
            raise ValueError(f"temporal_{resolution.status}")
        if len(resolution.path) > budget:
            raise ValueError("evidence_budget_exhausted")
        erased = store.db.scalar(
            f"SELECT 1 FROM memory_tombstones WHERE {scope_sql()} AND reason = 'forgotten' "
            f"AND frame_id IN ({','.join('?' for _ in resolution.path)}) LIMIT 1",
            (*scope.as_tuple(), *resolution.path),
        )
        if erased:
            raise ValueError("temporal_erased")
        if not root and resolution.frame_id != frame.id:
            raise ValueError("temporal_dependency_changed")

        metadata = resolution.metadata
        assert metadata is not None
        selections.append(
            TemporalEvidenceModel(
                requested_frame_id=frame.id,
                selected_frame_id=resolution.frame_id,
                selection_now=query.now,
                path=list(resolution.path),
                operation=metadata.operation,
                previous_frame_id=metadata.previous_frame_id,
                effective_at=metadata.observation.effective_at,
                observed_at=metadata.observation.observed_at,
                recorded_at=metadata.recorded_at,
            )
        )
        return store.get_frame(resolution.frame_id, scope=scope)

    def predecessor(frame: Frame) -> Frame:
        if frame.supersedes is None:
            raise ValueError("invalid_version_lineage")
        try:
            old = store.get_frame(frame.supersedes, scope=scope)
        except KeyError:
            raise ValueError("invalid_version_lineage") from None
        typed = any(
            entry.source_frame_id == old.id and entry.operation in ("supersede", "correct")
            for entry in store.lineage(frame.id, scope=scope)
        )
        if old.status != "superseded" or old.superseded_by != frame.id or not typed:
            raise ValueError("invalid_version_lineage")
        link = EvidenceLinkModel(frame_id=frame.id, related_frame_id=old.id, relation="supersedes")
        if link not in links:
            links.append(link)
        return old

    def visit(frame: Frame) -> None:
        if (
            query.known_at is not None
            and frame.frame_type != "I"
            and not _known_payload(store, frame.id, scope, query.known_at)
        ):
            raise ValueError("temporal_unknown")
        frame = select(frame)
        if frame.id in visiting:
            raise ValueError("dependency_cycle")
        if frame.id in frames:
            return
        if not is_recall_eligible(frame, now_iso=now):
            raise ValueError("ineligible_dependency")
        if len(frames) + len(visiting) >= budget:
            raise ValueError("evidence_budget_exhausted")
        visiting.add(frame.id)
        old = predecessor(frame) if frame.supersedes is not None else None
        if frame.frame_type == "P":
            if frame.base_frame_id is None:
                raise ValueError("missing_dependency")
            try:
                base = store.get_frame(frame.base_frame_id, scope=scope)
            except KeyError:
                raise ValueError("missing_dependency") from None
            if base.gop_id != frame.gop_id:
                raise ValueError("invalid_base")
            visit(base)
            links.append(
                EvidenceLinkModel(frame_id=frame.id, related_frame_id=base.id, relation="refines")
            )
        elif frame.frame_type == "B":
            if old is None:
                raise ValueError("unversioned_correction")
            if frame.base_frame_id != old.id:
                raise ValueError("invalid_correction_base")
            try:
                payload = json.loads(frame.content)
                references = payload["references"]
                if (
                    not isinstance(payload["description"], str)
                    or not payload["description"].strip()
                    or not isinstance(references, list)
                    or any(type(ref) is not int or ref < 1 for ref in references)
                ):
                    raise ValueError
            except (ValueError, TypeError, KeyError):
                raise ValueError("invalid_correction_payload") from None
            # A replacement for a P still needs that P's original (eligible) context.
            if old.frame_type == "P":
                if old.base_frame_id is None:
                    raise ValueError("missing_dependency")
                visit(store.get_frame(old.base_frame_id, scope=scope))
                links.append(
                    EvidenceLinkModel(
                        frame_id=old.id, related_frame_id=old.base_frame_id, relation="refines"
                    )
                )
            for reference in dict.fromkeys(references):
                if reference == old.id:
                    continue  # typed identity, never superseded plaintext
                visit(store.get_frame(reference, scope=scope))
                links.append(
                    EvidenceLinkModel(
                        frame_id=frame.id, related_frame_id=reference, relation="references"
                    )
                )
        visiting.remove(frame.id)
        frames[frame.id] = frame

    seed = select(seed, root=True)
    visit(seed)
    queue = list(frames.values())
    while queue:
        parent = queue.pop(0)
        rows = store.db.fetchall(
            f"SELECT * FROM memory_frames WHERE {scope_sql()} AND base_frame_id = ? "
            f"AND frame_type IN ('P', 'B') AND ({eligible_sql()} OR status = 'superseded') "
            "ORDER BY id LIMIT ?",
            (*scope.as_tuple(), parent.id, now, budget + 1),
        )
        if len(rows) > budget:
            raise ValueError("evidence_budget_exhausted")
        for row in rows:
            child = store._row_to_frame(row)
            identity_path: set[int] = set()
            while child.status == "superseded":
                if child.id in identity_path:
                    raise ValueError("dependency_cycle")
                if len(identity_path) >= budget:
                    raise ValueError("evidence_budget_exhausted")
                identity_path.add(child.id)
                if child.superseded_by is None:
                    raise ValueError("invalid_version_lineage")
                successor = store.get_frame(child.superseded_by, scope=scope)
                if successor.supersedes != child.id:
                    raise ValueError("invalid_version_lineage")
                predecessor(successor)
                child = successor
            if (
                query.known_at is not None
                and child.frame_type == "P"
                and not _known_payload(store, child.id, scope, query.known_at)
            ):
                continue
            if child.id not in frames:
                visit(child)
                queue.append(child)
    if (query.as_of is not None or query.known_at is not None) and not selections:
        raise ValueError("temporal_unmanaged")
    return list(frames.values()), links, selections


def recall_evidence(
    search: HybridSearch,
    query: str,
    *,
    limit: int = 10,
    profile: str | None = None,
    since: str | None = None,
    until: str | None = None,
    scope: MemoryScope | None = None,
    now_iso: str | None = None,
    candidate_budget: int = 1024,
    evidence_frame_budget: int = 256,
    as_of: str | None = None,
    known_at: str | None = None,
) -> RecallResponse:
    """Resolve current evidence; a seed quota never truncates its dependency closure."""
    if type(evidence_frame_budget) is not int or not 1 <= evidence_frame_budget <= 1024:
        raise ValueError("evidence_frame_budget must be between 1 and 1024")
    started = time.perf_counter()
    store = search._store
    scope = scope or search.scope
    now = now_iso or format_now(search._clock())
    generation = current_scope_generation(store.db.backend, scope)
    temporal_query = TemporalQuery(as_of=as_of, known_at=known_at, now=format_now(search._clock()))
    policy = search.temporal_policy
    historical = as_of is not None or known_at is not None
    if historical and not policy.allow_historical:
        raise PermissionError("Historical temporal views require host policy authorization")
    groups: dict[int, EvidenceGroupModel] = {}
    refusals = []
    expanded: dict[int, Frame] = {}
    scores = {}
    inspected: set[int] = set()
    raw_limit = min(candidate_budget, max(1, limit))
    while True:
        result = search.search(
            query,
            limit=raw_limit,
            profile=profile,
            since=since,
            until=until,
            scope=scope,
            now_iso=now,
            candidate_budget=candidate_budget,
        )
        for hit in result.hits:
            if hit.frame.id in inspected:
                continue
            inspected.add(hit.frame.id)
            try:
                frame = store.get_frame(hit.frame.id, scope=scope)
                frames, links, selections = _resolve_seed(
                    store, frame, scope, now, evidence_frame_budget, temporal_query, policy
                )
                if len(set(expanded) | {member.id for member in frames}) > evidence_frame_budget:
                    raise ValueError("evidence_budget_exhausted")
            except (KeyError, ValueError) as exc:
                reason = str(exc) if isinstance(exc, ValueError) else "missing_dependency"
                refusals.append(EvidenceRefusalModel(seed_id=hit.frame.id, reason=reason))
                continue
            root_id = frames[0].id
            group = groups.setdefault(
                root_id,
                EvidenceGroupModel(
                    view="historical" if historical else "current",
                    as_of=as_of,
                    known_at=known_at,
                    root_frame_id=root_id,
                    seed_ids=[],
                    frame_ids=[],
                    links=[],
                ),
            )
            group.seed_ids.append(frame.id)
            for member in frames:
                if member.id not in group.frame_ids:
                    group.frame_ids.append(member.id)
                expanded[member.id] = member
            for link in links:
                if link not in group.links:
                    group.links.append(link)
            for selection in selections:
                if selection not in group.temporal:
                    group.temporal.append(selection)
            selected_id = next(
                (s.selected_frame_id for s in selections if s.requested_frame_id == frame.id),
                frame.id,
            )
            scores[selected_id] = hit
            if len(scores) >= limit:
                break
        if (
            len(scores) >= limit
            or result.abstained
            or len(result.hits) < raw_limit
            or raw_limit == candidate_budget
        ):
            break
        raw_limit = min(candidate_budget, raw_limit * 2)
    frames = evidence_frames(store, list(expanded.values()), scope, known_at=known_at)
    current_generation = current_scope_generation(store.db.backend, scope)
    if current_generation != generation:
        refusals.extend(EvidenceRefusalModel(seed_id=fid, reason="scope_changed") for fid in scores)
        groups.clear()
        scores.clear()
        frames = []
    hits = []
    for frame in frames:
        hit = scores.get(frame.id)
        hits.append(
            RecallHitModel(
                frame=frame,
                rrf=hit.rrf if hit else 0.0,
                relevance=hit.relevance if hit else 0.0,
                fusion=hit.fusion if hit else 0.0,
                rerank=hit.rerank if hit else None,
                final=hit.final if hit else 0.0,
            )
        )
    abstained = result.abstained or not hits
    abstention = result.abstention.as_dict() if result.abstention else {}
    abstention["raw_reason"] = abstention.get("reason")
    abstention["raw_abstained"] = abstention.get("abstained")
    abstention["abstained"] = abstained
    if not hits and not result.abstained:
        abstention["reason"] = "no_usable_evidence"
    return RecallResponse(
        hits=hits,
        groups=list(groups.values()),
        refusals=refusals,
        scope_generation=current_generation,
        view="historical" if historical else "current",
        as_of=as_of,
        known_at=known_at,
        retrieval=RetrievalDiagnosticsModel(
            candidate_budget=candidate_budget,
            evidence_frame_budget=evidence_frame_budget,
            stop_reason=(
                "abstained"
                if result.abstained
                else "quota_satisfied"
                if len(scores) >= limit
                else "evidence_refused"
                if refusals
                else "budget_exhausted"
                if any(d.budget_exhausted for d in result.lanes.values())
                else "lane_exhausted"
            ),
            seed_limit=limit,
            matched_seeds=len(scores),
            expanded_frames=len(hits),
            quota_satisfied=len(scores) >= limit,
            lanes={
                name: LaneDiagnosticsModel.model_validate(diagnostic.__dict__)
                for name, diagnostic in result.lanes.items()
            },
        ),
        context="\n".join(
            render_group(group, {frame.id: frame for frame in frames}) for group in groups.values()
        ),
        abstained=abstained,
        abstention=abstention,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )
