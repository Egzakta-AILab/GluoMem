"""Invocation-owned revision checks for V4; never refresh an unowned binding."""

from gluomem.frames.semantics import SemanticStore
from gluomem.llm.observed_identity import ObservedSemanticProjection
from gluomem.llm.outputs import TARGET_REQUIRED_ACTIONS
from gluomem.llm.semantic import (
    SemanticProjection,
    permits,
    source_witness,
)


class SemanticCommitGuard:
    projection_type: type[SemanticProjection | ObservedSemanticProjection[str]] = SemanticProjection

    def __init__(self, plan, store):
        self.plan = plan
        self.store = SemanticStore(store)
        self.scope = plan.scope
        self.producer_contract = plan.eff_cfg.write.semantic_producer_contract
        from gluomem.llm.assertion_scope import snapshot_type, spec_type

        self.spec_type = spec_type(self.producer_contract)
        self.snapshot_type = snapshot_type(self.producer_contract)
        self.expected = dict(plan.semantic_revisions)
        self.check_inputs()

    def check_inputs(self):
        plan = self.plan
        if (
            plan.eff_cfg.write.writer_contract != "semantic-state-writer-v4"
            or tuple(plan.claims) != plan.semantic_inputs
        ):
            raise ValueError("Semantic input binding changed after preparation")
        from gluomem.write.events import _canonical_event_payload

        if _canonical_event_payload(plan.event)[0] != plan.payload_json:
            raise ValueError("Semantic source context changed after preparation")
        for claim, projection in zip(plan.claims, plan.projections, strict=True):
            if type(projection) is not self.projection_type or projection.meaning != claim.meaning:
                raise ValueError("Semantic projection binding changed after preparation")
            try:
                self.projection_type.model_validate(projection.model_dump())
            except ValueError:
                raise ValueError("Semantic projection binding changed after preparation") from None
            for target in projection.target_meanings:
                if target.frame_id is not None:
                    if (
                        target.frame_id not in plan.semantic_revisions
                        or target.semantic != plan.semantic_revisions[target.frame_id]
                    ):
                        raise ValueError("Semantic target binding changed after preparation")
                elif (
                    target.meaning is None
                    or target.meaning != plan.claims[target.claim_index].meaning
                    or not target.meaning.independently_versioned
                ):
                    raise ValueError("Semantic publication binding changed after preparation")
            current = source_witness(
                plan.event,
                self.spec_type(
                    text=claim.content,
                    meaning=claim.meaning,
                    message_id=claim.message_id,
                    verbatim_span=claim.verbatim_span,
                ),
            )
            if (
                current != claim.semantic_witness
                or current.source_role != claim.source_role
                or current.observed_at != claim.observed_at
            ):
                raise ValueError("Semantic source witness changed after preparation")

    def check(self):
        self.check_inputs()
        for frame_id, expected in self.expected.items():
            try:
                actual = self.store.snapshot(
                    frame_id, scope=self.scope, producer_contract=self.producer_contract
                )
            except (ValueError, KeyError):
                raise ValueError(
                    "Semantic source binding changed; prepare ingestion again"
                ) from None
            if actual != expected:
                raise ValueError("Semantic source binding changed; prepare ingestion again")

    def require_target(self, decision, claim):
        if decision.action not in TARGET_REQUIRED_ACTIONS:
            return
        self.check()
        surviving = self.expected.get(decision.target_frame_id)
        if (
            surviving is None
            or not surviving.receipts
            or not permits(decision.action, claim.meaning, surviving.meaning)
        ):
            raise ValueError(
                "Target requires surviving semantic authority; prepare ingestion again"
            )

    def attach(self, provenance_id, frame_id, claim):
        if frame_id not in self.expected:
            # Only an actual new publication may enter here; reuse must have been observed.
            self.expected[frame_id] = None
        self.check()
        prior = self.expected[frame_id]
        if prior is not None and prior.meaning != claim.meaning:
            raise ValueError("Semantic reused assertion changed during admission")
        receipt = self.store.attach(
            provenance_id, frame_id, claim.meaning, claim.semantic_witness, scope=self.scope
        )
        self.expected[frame_id] = self.snapshot_type(
            meaning=claim.meaning, receipts=(*prior.receipts, receipt) if prior else (receipt,)
        )
        self.check()

    def scrubbed(self, frame_ids=(), event_ids=()):
        # Named lifecycle effects only. Never accept a revision from an unrestricted reread.
        for fid, snapshot in self.expected.items():
            receipts = (
                tuple(r for r in snapshot.receipts if r.source_event_row_id not in event_ids)
                if snapshot
                else ()
            )
            self.expected[fid] = (
                self.snapshot_type(meaning=snapshot.meaning, receipts=receipts)
                if fid not in frame_ids and receipts
                else None
            )
