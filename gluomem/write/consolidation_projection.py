"""Conditional per-claim effect projection, never a planner or a source of stored truth."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from gluomem.frames.content_hash import content_hash, suppression_key
from gluomem.frames.eligibility import is_recall_eligible
from gluomem.frames.expiry import episode_expiry
from gluomem.frames.provenance import is_suppressed_by_tombstone
from gluomem.frames.store import Trust
from gluomem.llm.outputs import (
    ConsolidationProjection,
    ProjectedCandidate,
    ProjectedConsolidationDecision,
    StoredCandidate,
)
from gluomem.storage.scope import FrameStatus, scope_sql
from gluomem.timeutil import now_utc_iso, parse_utc, to_storage
from gluomem.write.gate import _exact_reuse_policy_updates

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gluomem.frames.store import FrameStore


@dataclass(frozen=True)
class _StoredReusePolicy:
    trust: Trust
    status: FrameStatus
    expires_at: str | None
    pinned: bool
    verified_by_user: bool


@dataclass(frozen=True)
class _ProjectedAdmission:
    content_hash: str
    gop_id: str
    claim_index: int
    trust: Trust
    status: FrameStatus
    expires_at: str | None
    pinned: bool
    verified_by_user: bool


class ClaimProjectionState:
    """Keep explicit prior choices visible to the next independent claim selection."""

    def __init__(self):
        self.publications: list[ProjectedCandidate] = []
        self.admissions: list[_ProjectedAdmission] = []
        self.stored_reuse: dict[int, _StoredReusePolicy] = {}
        self.consumed: set[tuple[int | None, int | None]] = set()
        self.transitioned: set[tuple[int | None, int | None]] = set()

    def project(self, claim_index: int, candidates: Sequence[StoredCandidate], claim, store, scope):
        targets = [
            ProjectedCandidate(
                **{k: v for k, v in c.model_dump().items() if k != "event_time"},
            )
            for c in candidates
        ] + self.publications
        projected = []
        for target in targets:
            key = (target.frame_id, target.claim_index)
            actions = []
            if key not in self.consumed:
                actions = ["NOOP"]
                # A prior projected transition keeps this exact predecessor frozen
                # through publication end. Historical DB successors alone do not.
                if key not in self.transitioned:
                    actions.append("SUPERSEDE")
                if target.frame_type == "I":
                    actions.append("REFINE")
                    successor = (
                        target.frame_id is not None
                        and store._backend.fetchone(
                            "SELECT frame_id FROM memory_frame_temporal "
                            f"WHERE previous_frame_id = ? AND {scope_sql()}",
                            (target.frame_id, *scope.as_tuple()),
                        )
                        is not None
                    )
                    if (
                        key not in self.transitioned
                        and not successor
                        and target.temporal_operation is not None
                        and target.observed_at is not None
                        and claim.observed_at is not None
                        and claim.effective_at is not None
                        and (
                            target.effective_at is None
                            or parse_utc(claim.effective_at) > parse_utc(target.effective_at)
                        )
                    ):
                        actions.append("TRANSITION")
            projected.append(target.model_copy(update={"allowed_actions": tuple(actions)}))
        return ConsolidationProjection(claim_index=claim_index, targets=tuple(projected))

    def apply(
        self,
        claim_index,
        claim,
        decision: ProjectedConsolidationDecision,
        source,
        *,
        event,
        cfg,
        verdict,
        store: FrameStore,
        scope,
    ):
        from gluomem.write.events import _gop_for_source_ref

        if decision.action == "NOOP" or is_suppressed_by_tombstone(
            store._backend,
            scope=scope,
            content_hash_val=content_hash(claim.content),
            suppression_key_val=suppression_key(claim.content),
            event_id=event.event_id,
            force_reassert=event.force_reassert,
        ):
            return
        target_key = (decision.target_frame_id, decision.target_claim_index)
        stored_target = (
            store.get_frame(decision.target_frame_id, scope=scope)
            if decision.target_frame_id is not None
            else None
        )
        target = (
            stored_target
            if decision.target_frame_id is not None
            else next(
                (p for p in self.publications if p.claim_index == decision.target_claim_index), None
            )
        )
        gop = (
            stored_target.gop_id
            if decision.action == "REFINE" and stored_target is not None
            else _gop_for_source_ref(event.source_ref)
        )
        needs_review = (
            event.require_review
            or decision.action == "CANDIDATE"
            or (
                event.source_type != "explicit_memory"
                and claim.confidence < cfg.lifecycle.review_threshold
            )
            or (
                decision.action == "SUPERSEDE"
                and target is not None
                and source in cfg.safety.untrusted_sources
                and target.source not in cfg.safety.untrusted_sources
                and not (event.pinned or event.source_type == "explicit_memory")
            )
        )
        trust = "trusted" if event.source_type == "explicit_memory" else "normal"
        if verdict.score >= cfg.safety.block_threshold:
            if cfg.safety.fail_mode == "reject":
                return
            trust = "quarantined"
        h = content_hash(claim.content)
        # gate.admit's first lookup is ANY-type, active/candidate, scope-wide
        # except for REFINE's GOP. Exclude only explicitly proposed retirements.
        rows = store._backend.fetchall(
            f"SELECT id FROM memory_frames WHERE {scope_sql()} AND content_hash = ? "
            "AND status IN ('active', 'candidate') "
            + ("AND gop_id = ? " if decision.action == "REFINE" else "")
            + "ORDER BY id",
            (*scope.as_tuple(), h, *((gop,) if decision.action == "REFINE" else ())),
        )
        match = next(
            (
                store.get_frame(int(r["id"]), scope=scope)
                for r in rows
                if (int(r["id"]), None) not in self.consumed
            ),
            None,
        )
        prior = next(
            (
                p
                for p in self.admissions
                if p.content_hash == h
                and (decision.action != "REFINE" or p.gop_id == gop)
                and (None, p.claim_index) not in self.consumed
            ),
            None,
        )
        if match is not None or prior is not None:
            if match is not None:
                # Freeze only policy at the first reuse; later claims carry
                # admission-owned updates, not a refreshed stored observation.
                reused = self.stored_reuse.get(match.id)
                if reused is None:
                    reused = _StoredReusePolicy(
                        trust=match.trust,
                        status=match.status,
                        expires_at=match.expires_at,
                        pinned=match.pinned,
                        verified_by_user=match.verified_by_user,
                    )
            else:
                assert prior is not None  # The enclosing branch found one reuse identity.
                reused = _StoredReusePolicy(
                    trust=prior.trust,
                    status=prior.status,
                    expires_at=prior.expires_at,
                    pinned=prior.pinned,
                    verified_by_user=prior.verified_by_user,
                )
            updates = _exact_reuse_policy_updates(
                trust=trust,
                existing_trust=reused.trust,
                pinned=event.pinned,
                existing_pinned=reused.pinned,
                verified_by_user=event.source_type == "explicit_memory",
                existing_verified_by_user=reused.verified_by_user,
                existing_expires_at=reused.expires_at,
            )
            # Apply only gate-owned flags to this conditional copy, never refresh
            # frozen target revisions or create a handle for a reused identity.
            projected_reuse = replace(reused, **updates)
            eligible = is_recall_eligible(
                projected_reuse.trust,
                status=projected_reuse.status,
                expires_at=projected_reuse.expires_at,
                pinned=projected_reuse.pinned,
            )
            if match is not None:
                self.stored_reuse[match.id] = projected_reuse
                match_key = (match.id, None)
            else:
                assert prior is not None
                self.admissions[self.admissions.index(prior)] = replace(prior, **updates)
                match_key = (None, prior.claim_index)
            if decision.action == "SUPERSEDE" and eligible and match_key != target_key:
                self.consumed.add(target_key)
            return  # Reuse never creates a new handle or remaps an old one.
        clf = cfg.safety.scanner.classifier
        if (
            clf.enabled
            and source in clf.reviewed_sources
            and not (event.pinned or event.source_type == "explicit_memory")
            and verdict.classifier_score >= clf.review_threshold
        ):
            needs_review = True
        expires_at = (
            episode_expiry(to_storage(event.created_at) or now_utc_iso(), cfg.lifecycle)
            if claim.kind == "episode" and not event.pinned
            else None
        )
        eligible = is_recall_eligible(
            trust,
            status="candidate" if needs_review else "active",
            expires_at=expires_at,
            pinned=event.pinned,
        )
        self.admissions.append(
            _ProjectedAdmission(
                content_hash=h,
                gop_id=gop,
                claim_index=claim_index,
                trust=trust,
                status="candidate" if needs_review else "active",
                expires_at=expires_at,
                pinned=event.pinned,
                verified_by_user=event.source_type == "explicit_memory",
            )
        )
        if not eligible:
            return
        if decision.action == "SUPERSEDE":
            self.consumed.add(target_key)
        if decision.action == "TRANSITION":
            self.transitioned.add(target_key)
        if decision.action in ("ADD", "SUPERSEDE", "TRANSITION"):
            self.publications.append(
                ProjectedCandidate(
                    claim_index=claim_index,
                    conditional=True,
                    content=claim.content,
                    kind=claim.kind,
                    source=source,
                    observed_at=claim.observed_at,
                    effective_at=claim.effective_at,
                    temporal_operation=(
                        "transition" if decision.action == "TRANSITION" else "observation"
                    )
                    if claim.observed_at
                    else None,
                    previous_frame_id=decision.target_frame_id
                    if decision.action == "TRANSITION"
                    else None,
                    previous_claim_index=(
                        decision.target_claim_index if decision.action == "TRANSITION" else None
                    ),
                )
            )
