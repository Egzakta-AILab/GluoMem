"""Selected current-source corroboration, never resurrection of revoked authority.

Only an eligible ADD that will reuse a frozen, known, equal I binding may carry a
transaction-local receipt prelude. No publication reference or new frame results.
"""

from dataclasses import dataclass
from typing import Literal

from gluomem.frames.content_hash import content_hash, suppression_key
from gluomem.frames.eligibility import is_recall_eligible
from gluomem.frames.provenance import is_suppressed_by_tombstone
from gluomem.llm.assertion_scope import AssertionScopedMeaning, AssertionScopedProjection
from gluomem.llm.semantic import source_witness

POLICY = "same-event-corroboration-v1"


class ReceiptCoherentProjection(AssertionScopedProjection):
    # Private host authority; wire_view and all native output leaves are unchanged.
    feasible_untargeted: tuple[Literal["ADD", "CANDIDATE"], ...]

    @property
    def untargeted_actions(self):
        return self.feasible_untargeted


@dataclass(frozen=True)
class CurrentSourceSupport:
    claim_index: int
    frame_id: int
    frame_revision: str


def enabled(cfg):
    return getattr(cfg.write, "semantic_reuse_policy", None) == POLICY


def survives(snapshot, target_key, revoked_sources, event_receipts, event_revoked):
    return bool(
        snapshot and any(r.source_event_row_id not in revoked_sources for r in snapshot.receipts)
    ) or (not event_revoked and (target_key[0] is None or target_key in event_receipts))


def suppressed(claim, event, store, scope):
    return is_suppressed_by_tombstone(
        store._backend,
        scope=scope,
        content_hash_val=content_hash(claim.content),
        suppression_key_val=suppression_key(claim.content),
        event_id=event.event_id,
        force_reassert=event.force_reassert,
    )


def current_support(
    claim_index, claim, decision, frame, snapshot, *, event, cfg, verdict, store, scope
):
    """Pure authority derivation from current source, selected action and frozen meaning."""
    from gluomem.llm.assertion_scope import AssertionScopedClaimSpec
    from gluomem.write.events import _frame_revision

    if not enabled(cfg) or decision.action != "ADD" or frame is None or snapshot is None:
        return None
    source = claim.frame_source
    clf = cfg.safety.scanner.classifier
    if (
        type(claim.meaning) is not AssertionScopedMeaning
        or claim.meaning.assertion_act != "direct_assertion"
        or not claim.meaning.independently_versioned
        or snapshot.meaning != claim.meaning
        or not snapshot.receipts
        or frame.frame_type != "I"
        or frame.base_frame_id is not None
        or not is_recall_eligible(frame)
        or content_hash(claim.content) != frame.content_hash
        or claim.source_role != claim.meaning.key.perspective
        or claim.semantic_witness is None
        or event.require_review
        or (
            event.source_type != "explicit_memory"
            and claim.confidence < cfg.lifecycle.review_threshold
        )
        or verdict.score >= cfg.safety.block_threshold
        or (
            clf.enabled
            and source in clf.reviewed_sources
            and not (event.pinned or event.source_type == "explicit_memory")
            and verdict.classifier_score >= clf.review_threshold
        )
        or suppressed(claim, event, store, scope)
    ):
        return None
    witness = source_witness(
        event,
        AssertionScopedClaimSpec(
            text=claim.content,
            meaning=claim.meaning,
            message_id=claim.message_id,
            verbatim_span=claim.verbatim_span,
        ),
    )
    if (
        witness != claim.semantic_witness
        or witness.source_role != claim.source_role
        or witness.observed_at != claim.observed_at
    ):
        raise ValueError("Current-source corroboration witness changed")
    return CurrentSourceSupport(claim_index, frame.id, _frame_revision(frame))


def feasible_action(state, claim_index, claim, decision, *, event, cfg, verdict, store, scope):
    """Use the same receipt lifetime for explicit menus and implicit hash reuse."""
    from gluomem.frames.semantics import SemanticStore

    if decision.action == "NOOP":
        return True  # Its explicit target's surviving authority is independently intersected.
    gop = None
    if decision.action == "REFINE" and decision.target_frame_id is not None:
        gop = store.get_frame(decision.target_frame_id, scope=scope).gop_id
    frame = store.find_frame_by_hash(gop, claim.content, scope=scope)
    if (
        frame is None
        or (frame.id, None) in state.consumed
        or suppressed(claim, event, store, scope)
    ):
        return True
    snapshot = SemanticStore(store).snapshot(
        frame.id, scope=scope, producer_contract=cfg.write.semantic_producer_contract
    )
    state.observe(frame.id, snapshot)
    if snapshot is None or snapshot.meaning != claim.meaning:
        return False
    return (
        survives(
            snapshot,
            (frame.id, None),
            state.revoked_sources,
            state.event_receipts,
            state.event_revoked,
        )
        or current_support(
            claim_index,
            claim,
            decision,
            frame,
            snapshot,
            event=event,
            cfg=cfg,
            verdict=verdict,
            store=store,
            scope=scope,
        )
        is not None
    )


def finish_support(state, claims, *, event, cfg, verdicts, store, scope):
    commands = tuple(state.receipt_support.values())
    if commands and (
        state.event_revoked
        or any((c.frame_id, None) in state.consumed for c in commands)
        or any(suppressed(c, event, store, scope) for c in claims)
        or any(
            v.score >= cfg.safety.block_threshold and cfg.safety.fail_mode == "reject"
            for v in verdicts
        )
    ):
        raise ValueError("Current-source corroboration must survive the complete selected batch")
    return commands


def insert_provenance(backend, *, scope, frame_id, event_row_id, event, claim, now_s):
    """Ordinary admitted-source row, shared by the prelude and normal publication."""
    row = backend.fetchone(
        "INSERT INTO memory_provenance (tenant_id, principal_id, namespace, "
        "frame_id, source_event_id, source_type, source_ref, message_id, message_role, "
        "observed_at, verbatim_span, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
        (
            *scope.as_tuple(),
            frame_id,
            event_row_id,
            event.source_type,
            event.source_ref,
            claim.message_id,
            claim.source_role,
            claim.observed_at,
            claim.verbatim_span or claim.content,
            now_s,
        ),
    )
    return int(row["id"])


def attach_prelude(plan, store, guard, event_row_id, now_s):
    """Reconstruct commands before cleanup, then mint real receipts inside the transaction."""
    from gluomem.frames.semantics import SemanticStore
    from gluomem.frames.source_fences import assert_source_not_retracted
    from gluomem.write.events import _frame_revision

    staged = {}
    guard.check()
    for command in plan.receipt_support:
        assert_source_not_retracted(store._backend, plan.scope, plan.event.source_ref)
        index = command.claim_index
        claim = plan.claims[index]
        frame = store.get_frame(command.frame_id, scope=plan.scope)
        snapshot = SemanticStore(store).require_reuse(frame.id, claim.meaning, scope=plan.scope)
        if (
            snapshot != plan.semantic_revisions.get(frame.id)
            or _frame_revision(frame) != command.frame_revision
        ):
            raise ValueError("Current-source corroboration observation changed")
        expected = current_support(
            index,
            claim,
            plan.decisions[index],
            frame,
            snapshot,
            event=plan.event,
            cfg=plan.eff_cfg,
            verdict=plan.verdicts[index],
            store=store,
            scope=plan.scope,
        )
        if expected != command:
            raise ValueError("Current-source corroboration command changed")
        provenance_id = insert_provenance(
            store._backend,
            scope=plan.scope,
            frame_id=frame.id,
            event_row_id=event_row_id,
            event=plan.event,
            claim=claim,
            now_s=now_s,
        )
        guard.attach(provenance_id, frame.id, claim)
        staged[index] = frame.id
    return staged


def require_staged(plan, staged, guard):
    """A staged receipt is conditional on both actual reuse and final live source support."""
    guard.check()
    for command in plan.receipt_support:
        if staged.get(command.claim_index) != command.frame_id:
            raise ValueError("Current-source corroboration was not consumed by its selected claim")
        snapshot = guard.expected.get(command.frame_id)
        claim = plan.claims[command.claim_index]
        if (
            not snapshot
            or snapshot.meaning != claim.meaning
            or not any(
                r.witness == claim.semantic_witness and r.source_context == plan.payload_json
                for r in snapshot.receipts
            )
        ):
            raise ValueError("Current-source corroboration did not survive publication")
