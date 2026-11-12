"""Opt-in intersection and commit binding over the unchanged structural planner."""

import json
from copy import copy
from dataclasses import asdict

from gluomem.llm.observed_identity import (
    PRODUCER_CONTRACT,
    PRODUCER_CONTRACTS,
    PROPERTY_IDENTITY_CONTRACT,
    SAFE_PREPARATION_CONTRACTS,
    IdentityContext,
    IdentityProposal,
    IdentityTarget,
    ObservedIdentityAlignment,
    ObservedSemanticProjection,
    PropertyIdentityProposal,
)
from gluomem.llm.outputs import TARGET_REQUIRED_ACTIONS, StoredCandidate
from gluomem.llm.projection_contract import semantic_projection_type
from gluomem.llm.semantic import SemanticClaimContext, SemanticProjection, TargetMeaning
from gluomem.write.consolidation_projection import ClaimProjectionState
from gluomem.write.semantic_commit import SemanticCommitGuard
from gluomem.write.semantic_policy import SemanticProjectionState


def producer_binding(plan):
    return json.dumps(
        {
            "policy": plan.eff_cfg.write.semantic_producer_contract,
            "claims": [asdict(c) for c in plan.claims],
            "decisions": [d.model_dump(mode="json") for d in plan.decisions],
            "projections": [p.model_dump(mode="json") for p in plan.projections],
            "semantic_revisions": plan.semantic_revisions,
            "candidate_revisions": plan.candidate_revisions,
            "temporal_revisions": plan.temporal_revisions,
            "generation": plan.projection_generation,
            **(
                {
                    "receipt_policy": plan.receipt_policy,
                    "receipt_support": plan.receipt_support,
                    "receipt_config": plan.eff_cfg.model_dump(mode="json"),
                    "receipt_verdicts": plan.verdicts,
                    "receipt_payload": (plan.payload_json, plan.payload_hash, plan.scope),
                }
                if plan.receipt_policy is not None
                else {}
            ),
        },
        default=lambda x: x.model_dump(mode="json") if hasattr(x, "model_dump") else asdict(x),
        sort_keys=True,
    )


class ObservedProjectionState(SemanticProjectionState):
    def __init__(
        self,
        extractor,
        *,
        producer_contract=PRODUCER_CONTRACT,
        event=None,
        cfg=None,
        verdicts=(),
        frame_source=None,
    ):
        super().__init__()
        if (
            producer_contract not in PRODUCER_CONTRACTS
            or not isinstance(extractor, ObservedIdentityAlignment)
            or extractor.semantic_producer_capability != producer_contract
        ):
            raise ValueError("Observed producer requires its explicit identity capability")
        if producer_contract in SAFE_PREPARATION_CONTRACTS and (event is None or cfg is None):
            raise ValueError("Observed preparation requires its source admission policy")
        self.context_type = SemanticClaimContext
        self.target_type = TargetMeaning
        self.identity_target_type = IdentityTarget
        self.identity_context_type = IdentityContext
        self.projection_type = semantic_projection_type(
            producer_contract,
            getattr(cfg.write, "semantic_reuse_policy", None) if cfg is not None else None,
        )
        if producer_contract == "observed-assertion-scope-v4":
            from gluomem.llm.assertion_scope import (
                AssertionScopedContext,
                AssertionScopedIdentityContext,
                AssertionScopedIdentityTarget,
                AssertionScopedTargetMeaning,
            )

            self.context_type = AssertionScopedContext
            self.target_type = AssertionScopedTargetMeaning
            self.identity_target_type = AssertionScopedIdentityTarget
            self.identity_context_type = AssertionScopedIdentityContext

        self.extractor = extractor
        self.producer_contract = producer_contract
        self.event, self.cfg, self.verdicts, self.frame_source = event, cfg, verdicts, frame_source

    def _preserves_transition_authority(self, target, claim_index, claim, store, scope):
        """Predict named semantic scrub effects, not an updated stored observation."""
        if self.producer_contract not in SAFE_PREPARATION_CONTRACTS or not self.transitioned:
            return True
        from gluomem.llm.outputs import ProjectedConsolidationDecision

        shadow = copy(self)
        for name, value in vars(self).items():
            if isinstance(value, (dict, list, set)):
                setattr(shadow, name, value.copy())
        # Reuse the real conditional admission/revocation projection, including
        # suppression, reuse, review, quarantine and current-event receipts.
        SemanticProjectionState.apply(
            shadow,
            claim_index,
            claim,
            ProjectedConsolidationDecision(
                action="SUPERSEDE",
                target_frame_id=target.frame_id,
                target_claim_index=target.claim_index,
            ),
            claim.frame_source or self.frame_source,
            event=self.event,
            cfg=self.cfg,
            verdict=self.verdicts[claim_index],
            store=store,
            scope=scope,
        )
        for key in self.transitioned:
            snapshot = self.revisions.get(key[0])
            stored = snapshot is not None and any(
                r.source_event_row_id not in shadow.revoked_sources for r in snapshot.receipts
            )
            current = not shadow.event_revoked and (key[0] is None or key in shadow.event_receipts)
            if key in shadow.consumed or not (stored or current):
                return False
        return True

    def project(self, claim_index, candidates, claim, store, scope):
        structural = ClaimProjectionState.project(
            self,
            claim_index,
            [StoredCandidate.model_validate(c.model_dump()) for c in candidates],
            claim,
            store,
            scope,
        )
        snapshots = {c.frame_id: c.semantic for c in candidates}
        for fid, snapshot in snapshots.items():
            self.observe(fid, snapshot)
        context = self.context_type(
            kind=claim.kind,
            source_role=claim.source_role,
            observed_at=claim.observed_at,
            effective_at=claim.effective_at,
            meaning=claim.meaning,
            witness=claim.semantic_witness,
        )
        bindings, visible, surviving_refs = [], [], set()
        for target in structural.targets:
            snapshot = snapshots.get(target.frame_id)
            meaning = (
                (snapshot.meaning if snapshot else None)
                if target.frame_id is not None
                else self.meanings[target.claim_index]
            )
            binding = self.target_type(
                frame_id=target.frame_id,
                claim_index=target.claim_index,
                meaning=meaning,
                semantic=snapshot,
            )
            bindings.append(binding)
            tokens = (
                tuple(
                    r.token
                    for r in snapshot.receipts
                    if r.source_event_row_id not in self.revoked_sources
                )
                if snapshot
                else ()
            )
            key = (target.frame_id, target.claim_index)
            if (
                not tokens
                and snapshot
                and self.cfg is not None
                and getattr(self.cfg.write, "semantic_reuse_policy", None) is not None
                and not self.event_revoked
                and any(c.frame_id == target.frame_id for c in self.receipt_support.values())
            ):
                # These are genuine frozen observation tokens, NOT minted fresh
                # receipts. A preceding selected, typed current-source command
                # conditionally preserves their equal meaning. Commit requires
                # that command's actual new receipt, never these revoked tokens.
                tokens = tuple(r.token for r in snapshot.receipts)
            survives = bool(tokens) or (
                not self.event_revoked and (target.frame_id is None or key in self.event_receipts)
            )
            if (
                self.cfg is not None
                and getattr(self.cfg.write, "semantic_reuse_policy", None) is not None
            ):
                from gluomem.write.receipt_coherence import survives as receipt_survives

                survives = receipt_survives(
                    snapshot,
                    key,
                    self.revoked_sources,
                    self.event_receipts,
                    self.event_revoked,
                )
            if survives and meaning is not None:
                surviving_refs.add(key)
                if target.allowed_actions and meaning.key.perspective == claim.source_role:
                    visible.append(
                        self.identity_target_type(
                            target=target, meaning=meaning, receipt_tokens=tokens
                        )
                    )
        observed = self.identity_context_type(claim_index=claim_index, targets=tuple(visible))
        proposal = (
            self.extractor.align_claim_identity(
                claim.content, candidates, context=context, projection=observed
            )
            if visible
            else PropertyIdentityProposal(
                relation="independent" if claim.meaning.independently_versioned else "unresolved",
                assertion_relation="unrelated"
                if claim.meaning.independently_versioned
                else "unresolved",
                subject_alignment="not_aligned",
                property_alignment="not_aligned",
            )
            if self.producer_contract in (PROPERTY_IDENTITY_CONTRACT, "observed-assertion-scope-v4")
            else IdentityProposal(
                relation="independent" if claim.meaning.independently_versioned else "unresolved"
            )
        )
        proposal = observed.validate_proposal(proposal, context)
        # Construct authority before intersecting; never validate the wider structural menu.
        draft = self.projection_type.model_construct(
            producer_contract=self.producer_contract,
            claim_index=claim_index,
            targets=(),
            meaning=claim.meaning,
            context=context,
            structural=structural,
            target_meanings=tuple(bindings),
            identity_context=observed,
            proposal=proposal,
            identity_call="native" if visible else "empty-context",
        )
        targets = tuple(
            t.model_copy(
                update={
                    "allowed_actions": tuple(
                        a
                        for a in t.allowed_actions
                        if (t.frame_id, t.claim_index) in surviving_refs
                        and draft.permits_relation(a, t, b)
                        and (
                            a != "SUPERSEDE"
                            or self._preserves_transition_authority(
                                t, claim_index, claim, store, scope
                            )
                        )
                    )
                }
            )
            for t, b in zip(structural.targets, bindings, strict=True)
        )
        extra = {}
        if (
            self.cfg is not None
            and getattr(self.cfg.write, "semantic_reuse_policy", None) is not None
        ):
            from gluomem.llm.assertion_scope import AssertionScopedProjection
            from gluomem.llm.outputs import ProjectedConsolidationDecision
            from gluomem.write.receipt_coherence import feasible_action

            def feasible(action, target=None):
                return feasible_action(
                    self,
                    claim_index,
                    claim,
                    ProjectedConsolidationDecision(
                        action=action,
                        target_frame_id=target.frame_id if target else None,
                        target_claim_index=target.claim_index if target else None,
                    ),
                    event=self.event,
                    cfg=self.cfg,
                    verdict=self.verdicts[claim_index],
                    store=store,
                    scope=scope,
                )

            targets = tuple(
                t.model_copy(
                    update={
                        "allowed_actions": tuple(a for a in t.allowed_actions if feasible(a, t))
                    }
                )
                for t in targets
            )
            base_actions = AssertionScopedProjection.untargeted_actions.fget
            assert base_actions is not None
            extra["feasible_untargeted"] = tuple(a for a in base_actions(draft) if feasible(a))
            if not extra["feasible_untargeted"] and not any(t.allowed_actions for t in targets):
                raise ValueError("No receipt-feasible action for the current claim")
        return self.projection_type.model_validate(
            {**draft.model_dump(), "targets": targets, **extra}
        )


class ObservedCommitGuard(SemanticCommitGuard):
    projection_type: type[SemanticProjection | ObservedSemanticProjection[str]] = (
        ObservedSemanticProjection
    )

    def __init__(self, plan, store):
        self.projection_type = semantic_projection_type(
            plan.eff_cfg.write.semantic_producer_contract, plan.receipt_policy
        )
        super().__init__(plan, store)

    def check_inputs(self):
        if self.plan.receipt_policy != getattr(
            self.plan.eff_cfg.write, "semantic_reuse_policy", None
        ) or (self.plan.receipt_support and self.plan.receipt_policy is None):
            raise ValueError("Current-source corroboration policy changed after preparation")
        if (
            self.plan.eff_cfg.write.semantic_producer_contract not in PRODUCER_CONTRACTS
            or any(
                p.producer_contract != self.plan.eff_cfg.write.semantic_producer_contract
                for p in self.plan.projections
            )
            or self.plan.producer_inputs != producer_binding(self.plan)
        ):
            raise ValueError("Observed identity input binding changed after preparation")
        super().check_inputs()
        for claim, projection in zip(self.plan.claims, self.plan.projections, strict=True):
            if (
                projection.context.meaning != claim.meaning
                or projection.context.witness != claim.semantic_witness
                or projection.context.kind != claim.kind
                or projection.context.observed_at != claim.observed_at
                or projection.context.effective_at != claim.effective_at
                or projection.context.source_role != claim.source_role
            ):
                raise ValueError("Observed identity claim context changed")

    def require_target(self, decision, claim):
        if decision.action not in TARGET_REQUIRED_ACTIONS:
            return
        self.check()
        index = next(i for i, c in enumerate(self.plan.claims) if c is claim)
        projection = self.plan.projections[index]
        selected = self.plan.decisions[index]
        projection.validate_decision(selected)
        target = next(
            t
            for t in projection.targets
            if (t.frame_id, t.claim_index)
            == (selected.target_frame_id, selected.target_claim_index)
        )
        binding = next(
            b
            for b in projection.target_meanings
            if (b.frame_id, b.claim_index) == (target.frame_id, target.claim_index)
        )
        actual = self.expected.get(decision.target_frame_id)
        if (
            actual is None
            or not actual.receipts
            or actual.meaning != binding.meaning
            or not projection.permits_relation(decision.action, target, binding)
        ):
            raise ValueError(
                "Target requires surviving observed identity authority; prepare ingestion again"
            )
