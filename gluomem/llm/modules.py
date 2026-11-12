"""gluomem.llm.modules — config→module mapping + the dspy-backed ``Extractors``.

THE ONLY package whose import pulls dspy transitively is this module tree
(router/signatures/modules/judge/bootstrap/optimize). Never import it at
module top level outside ``gluomem/llm/``; go through
``gluomem.llm.make_extractors`` instead.

Module mapping (config ``dspy.tasks.<task>.module``): ``predict`` →
``dspy.Predict``, ``cot`` → ``dspy.ChainOfThought``; a task with
``refine: {enabled: true, N: k}`` is wrapped in ``dspy.Refine`` for
output-validity retries.

Refine notes (dspy 3.2.1 behaviour):

- the reward fn runs only AFTER the adapter successfully parsed the typed
  output — returning 1.0 with ``threshold=1.0`` therefore means "first valid
  parse wins" and the OfferFeedback LLM path never triggers;
- a parse failure raises inside the attempt and Refine retries with a new
  rollout id at ``temperature=1.0`` (the task temperature is bypassed by
  design during retries);
- ``Refine`` MUTATES ``self.fail_count`` across calls and returns ``None``
  when every attempt failed — so the wrapper is constructed per invocation
  and a ``None`` prediction becomes ``ExtractionError``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dspy

from gluomem.llm import make_extractors
from gluomem.llm.bootstrap import load_or_build
from gluomem.llm.outputs import (
    CONSOLIDATION_PROJECTION_PROTOCOL,
    ClaimConsolidationContext,
    CognifyResult,
    ConsolidationDecision,
    ConsolidationProjection,
    ContradictionVerdict,
    ExtractedClaimSpec,
    ExtractedEvent,
    ExtractedFact,
    ExtractionError,
    ProjectedConsolidationDecision,
    StoredCandidate,
)
from gluomem.llm.retry import run_with_retry
from gluomem.llm.router import LMRouter
from gluomem.llm.signatures import TASK_SIGNATURES

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from gluomem.config.app_config import AppConfig, DspyTaskConfig


def _parse_validity_reward(args: dict, pred: dspy.Prediction) -> float:  # noqa: ARG001
    """Reaching this fn means the adapter parsed the typed output — valid."""
    return 1.0


def build_module(task: str, cfg: AppConfig) -> dspy.Module:
    """Build the (unwrapped) dspy module for ``task`` from config."""
    if task == "identity_alignment" and cfg.write.semantic_producer_contract is not None:
        from gluomem.llm.observed_identity import PROPERTY_IDENTITY_CONTRACT
        from gluomem.llm.observed_signatures import IdentityAlignment, PropertyIdentityAlignment

        signature = (
            PropertyIdentityAlignment
            if cfg.write.semantic_producer_contract
            in (PROPERTY_IDENTITY_CONTRACT, "observed-assertion-scope-v4")
            else IdentityAlignment
        )
        if cfg.write.semantic_producer_contract == "observed-assertion-scope-v4":
            from gluomem.llm.observed_signatures import assertion_scope_identity_signature

            signature = assertion_scope_identity_signature()
    else:
        try:
            signature = TASK_SIGNATURES[task]
        except KeyError as exc:
            known = ", ".join(sorted(TASK_SIGNATURES))
            raise KeyError(f"unknown dspy task '{task}'; known: {known}") from exc
    if task == "cognify" and cfg.kg.entity_type_prompt_enum:
        from gluomem.llm.signatures import make_cognify_signature

        signature = make_cognify_signature(cfg.kg.entity_type_allowlist)
    tc = cfg.dspy.tasks[task]
    if tc.module == "predict":
        return dspy.Predict(signature)
    if tc.module == "cot":
        return dspy.ChainOfThought(signature)
    raise ValueError(f"dspy.tasks.{task}.module '{tc.module}' invalid; expected 'predict' or 'cot'")


def _refine_wrap(module: dspy.Module, tc: DspyTaskConfig) -> dspy.Module:
    """Per-invocation Refine wrapper (Refine mutates fail_count across calls)."""
    refine = tc.refine or {}
    if not refine.get("enabled", False):
        return module
    n = int(refine.get("N", 1))
    return dspy.Refine(
        module=module,
        N=n,
        reward_fn=_parse_validity_reward,
        threshold=1.0,
        fail_count=n,
    )


class DspyExtractors:
    """dspy-backed ``Extractors`` (fact/event/profile/cognify/contradiction).

    Profile extraction deliberately reuses the ``fact_extraction`` task over
    per-speaker text (the canonical config surface has four tasks; profile is
    a lane, not a separate signature).
    """

    def __new__(cls, cfg, **kwargs):
        if cls is DspyExtractors and cfg.write.semantic_producer_contract is not None:
            return object.__new__(ObservedDspyExtractors)
        return object.__new__(cls)

    consolidation_projection_protocol = CONSOLIDATION_PROJECTION_PROTOCOL

    @property
    def writer_capability(self):
        return self._cfg.write.writer_contract

    def __init__(self, cfg: AppConfig, *, router: LMRouter | None = None) -> None:
        from gluomem.llm.concurrency import ConcurrencyGate
        from gluomem.llm.usage import UsageAccumulator

        self._cfg = cfg.model_copy(deep=True)
        if (
            self._cfg.write.writer_contract == "semantic-state-writer-v4"
            and self._cfg.dspy.cache.disk
        ):
            raise ValueError("Semantic writer requires native disk cache disabled")
        self._router = router if router is not None else LMRouter(self._cfg)
        self._modules: dict[str, dspy.Module] = {}
        for task in self._cfg.dspy.output_field_limits:
            if (
                task not in TASK_SIGNATURES
                and not (
                    task == "identity_alignment" and self._cfg.write.semantic_producer_contract
                )
            ) or task not in self._cfg.dspy.tasks:
                raise ValueError("unknown output_field_limits task")
            self._modules[task] = load_or_build(task, self._cfg)
        # Token accounting across every extraction call (thread-safe; the
        # write path fans out). Consumers read .usage.snapshot().
        self.usage = UsageAccumulator()
        # Process-wide throttle: however many threads the batch/session pools
        # spawn, at most max_concurrent_llm extraction calls hit the API at once.
        self._llm_gate = ConcurrencyGate(cfg.write.max_concurrent_llm)

    def _module(self, task: str) -> dspy.Module:
        if task not in self._modules:
            self._modules[task] = load_or_build(task, self._cfg)
        return _refine_wrap(self._modules[task], self._cfg.dspy.tasks[task])

    def _run(
        self,
        task: str,
        *,
        _module_override: dspy.Module | None = None,
        **inputs: Any,
    ) -> dspy.Prediction:
        if self._cfg.dspy.retry.max_attempts > 1 and dspy.settings.send_stream is not None:
            raise ExtractionError("transient retries require non-streaming extraction")
        module = _module_override if _module_override is not None else self._module(task)

        def execute() -> dspy.Prediction:
            with self._llm_gate, self._router.bind(task):
                pred = module(**inputs)
            if pred is None:
                raise ExtractionError("extraction produced no valid output")
            return pred

        try:
            pred = run_with_retry(execute, task=task, policy=self._cfg.dspy.retry)
        except Exception as exc:
            raise ExtractionError("structured extraction failed") from exc
        if self._cfg.dspy.track_usage:
            self.usage.add(pred.get_lm_usage())
        return pred

    # --- Extractors protocol -------------------------------------------------

    def extract_facts(self, text: str) -> list[ExtractedFact]:
        pred = self._run("fact_extraction", text=text)
        return list(pred.facts)

    def extract_events(self, source_text: str, reference_date: datetime) -> list[ExtractedEvent]:
        pred = self._run(
            "event_extraction",
            source_text=source_text,
            reference_date=reference_date,
        )
        return list(pred.events)

    def extract_profile(self, text: str, *, speaker: str) -> list[ExtractedFact]:
        pred = self._run(
            "fact_extraction",
            text=f"Speaker {speaker} said:\n{text}",
        )
        return list(pred.facts)

    def cognify(self, text: str) -> CognifyResult:
        pred = self._run("cognify", text=text)
        return CognifyResult(entities=list(pred.entities), relations=list(pred.relations))

    def check_contradiction(
        self, new_statement: str, stored_statement: str
    ) -> ContradictionVerdict:
        pred = self._run(
            "contradiction",
            new_statement=new_statement,
            stored_statement=stored_statement,
        )
        return pred.verdict

    def extract_event_claims(
        self, source_text: str, *, reference_date: datetime | None = None
    ) -> list[ExtractedClaimSpec]:
        if self._cfg.write.writer_contract == "semantic-state-writer-v4":
            from gluomem.llm.semantic import SemanticClaimSpec
            from gluomem.write.source_binding import ACTIVE_SOURCE_ENVELOPE

            source = ACTIVE_SOURCE_ENVELOPE.get()
            if source is None or source.get("writer_contract") != self.writer_capability:
                raise ExtractionError("Semantic extraction requires its source envelope")
            tc = self._cfg.dspy.tasks["event_claims"]
            if (
                self._cfg.dspy.retry.max_attempts != 1
                or self._cfg.dspy.validation_retries
                or (tc.refine or {}).get("enabled")
            ):
                raise ExtractionError("Semantic extraction requires a single send policy")
            base = self._module("event_claims")
            if (
                not isinstance(base, (dspy.Predict, dspy.ChainOfThought))
                or len(base.predictors()) != 1
                or isinstance(base, dspy.Refine)
                or any(base.named_sub_modules(type_=dspy.Refine, skip_compiled=False))
            ):
                raise ExtractionError("Semantic extraction requires a single native send")
            module = base.deepcopy()
            for predictor in module.predictors():
                predictor.lm = None
                predictor.config = {
                    k: v
                    for k, v in predictor.config.items()
                    if k not in {"n", "max_tokens", "max_completion_tokens"}
                }
                predictor.config["n"] = 1
                if tc.max_tokens is not None:
                    predictor.config["max_tokens"] = tc.max_tokens
            pred = self._run(
                "event_claims",
                _module_override=module,
                source_text=source_text,
                source_context=source["source_event"],
                reference_date=reference_date.isoformat() if reference_date is not None else "",
            )
            if self._cfg.write.semantic_producer_contract == "observed-assertion-scope-v4":
                from gluomem.llm.assertion_scope import scoped_claim_type

                return [scoped_claim_type().model_validate(c.model_dump()) for c in pred.claims]
            return [SemanticClaimSpec.model_validate(c.model_dump()) for c in pred.claims]
        pred = self._run(
            "event_claims",
            source_text=source_text,
            reference_date=reference_date.isoformat() if reference_date is not None else "",
        )
        return list(pred.claims)

    def consolidate_claim(
        self, new_claim: str, candidates: Sequence[StoredCandidate]
    ) -> ConsolidationDecision:
        if self.writer_capability == "semantic-state-writer-v4":
            raise ExtractionError("Semantic writer requires its bound context and legal projection")
        if not candidates:
            return ConsolidationDecision(action="ADD")
        pred = self._run(
            "consolidate",
            new_claim=new_claim,
            candidates=list(candidates),
        )
        return self._validate_consolidation_decision(pred.decision, candidates)

    def consolidate_claim_with_context(
        self,
        new_claim: str,
        candidates: Sequence[StoredCandidate],
        *,
        context: ClaimConsolidationContext,
    ) -> ConsolidationDecision:
        """Use source-bound metadata without changing the legacy compiled task."""
        from gluomem.llm.signatures import contextual_consolidation_module

        if self.writer_capability == "semantic-state-writer-v4":
            raise ExtractionError("Semantic writer requires its bound context and legal projection")
        if not candidates:
            return ConsolidationDecision(action="ADD")
        module = contextual_consolidation_module(self._module("consolidate"))
        pred = self._run(
            "consolidate",
            _module_override=module,
            new_claim=new_claim,
            candidates=list(candidates),
            new_claim_context=context,
        )
        return self._validate_consolidation_decision(pred.decision, candidates)

    def consolidate_claim_with_projection(
        self,
        new_claim: str,
        candidates: Sequence[StoredCandidate],
        *,
        context: ClaimConsolidationContext,
        projection: ConsolidationProjection,
    ) -> ProjectedConsolidationDecision:
        from gluomem.llm.signatures import projected_consolidation_module

        if self.consolidation_projection_protocol != CONSOLIDATION_PROJECTION_PROTOCOL:
            raise ExtractionError("Unsupported consolidation projection protocol")
        projection.validate_candidates(candidates)
        observed_projection = None
        can_add = True
        if self.writer_capability == "semantic-state-writer-v4":
            from gluomem.llm.observed_identity import ObservedSemanticProjection
            from gluomem.llm.projection_contract import semantic_projection_type
            from gluomem.llm.semantic import SemanticClaimContext, SemanticProjection

            expected = semantic_projection_type(
                self._cfg.write.semantic_producer_contract,
                self._cfg.write.semantic_reuse_policy,
            )
            if self._cfg.write.semantic_producer_contract == "observed-assertion-scope-v4":
                from gluomem.llm.assertion_scope import AssertionScopedContext

                if type(context) is not AssertionScopedContext:
                    raise ExtractionError("Assertion scope requires its exact claim context")
            if (
                not isinstance(context, SemanticClaimContext)
                or not isinstance(projection, (SemanticProjection, ObservedSemanticProjection))
                or type(projection) is not expected
                or context.meaning != projection.meaning
                or (
                    self._cfg.write.semantic_producer_contract is not None
                    and (
                        not isinstance(projection, ObservedSemanticProjection)
                        or projection.producer_contract
                        != self._cfg.write.semantic_producer_contract
                    )
                )
            ):
                raise ExtractionError("Semantic consolidation requires matching bound meanings")
            can_add = (
                "ADD" in projection.untargeted_actions
                if self._cfg.write.semantic_producer_contract == "observed-assertion-scope-v4"
                else context.meaning.independently_versioned
            )
            if isinstance(projection, ObservedSemanticProjection):
                observed_projection = projection
        if not projection.targets and can_add:
            return ProjectedConsolidationDecision(action="ADD")
        tc = self._cfg.dspy.tasks["consolidate"]
        if (
            self._cfg.dspy.retry.max_attempts != 1
            or self._cfg.dspy.validation_retries
            or (tc.refine or {}).get("enabled")
        ):
            raise ExtractionError("Projected consolidation requires a single send/attempt policy")
        base = self._module("consolidate")
        if (
            not isinstance(base, (dspy.Predict, dspy.ChainOfThought))
            or len(base.predictors()) != 1
            or isinstance(base, dspy.Refine)
            or any(base.named_sub_modules(type_=dspy.Refine, skip_compiled=False))
        ):
            raise ExtractionError("Projected consolidation requires a single native send")
        module = projected_consolidation_module(
            base,
            projection,
            writer_contract=self._cfg.write.writer_contract,
            semantic_producer_contract=self._cfg.write.semantic_producer_contract,
        )
        for predictor in module.predictors():
            predictor.lm = None  # Only the invocation's task router may dispatch.
            predictor.config = {
                k: v
                for k, v in predictor.config.items()
                if k not in {"max_tokens", "max_completion_tokens", "n"}
            }
            predictor.config["n"] = 1
            if tc.max_tokens is not None:
                predictor.config["max_tokens"] = tc.max_tokens
        pred = self._run(
            "consolidate",
            _module_override=module,
            new_claim=new_claim,
            candidates=(
                [StoredCandidate.model_validate(c.model_dump()) for c in candidates]
                if self._cfg.write.semantic_producer_contract
                else list(candidates)
            ),
            new_claim_context=context,
            projection=(
                observed_projection.wire_view() if observed_projection is not None else projection
            ),
        )
        return projection.validate_decision(pred.decision)

    @staticmethod
    def _validate_consolidation_decision(
        decision: ConsolidationDecision,
        candidates: Sequence[StoredCandidate],
    ) -> ConsolidationDecision:
        if decision.target_frame_id is not None and decision.target_frame_id not in {
            candidate.frame_id for candidate in candidates
        }:
            raise ExtractionError("Consolidation target is outside the observed candidate set")
        if decision.action == "REFINE" and not any(
            candidate.frame_id == decision.target_frame_id and candidate.frame_type == "I"
            for candidate in candidates
        ):
            raise ExtractionError("REFINE requires an observed eligible I-frame target")
        if decision.action == "TRANSITION" and not any(
            candidate.frame_id == decision.target_frame_id
            and candidate.frame_type == "I"
            and candidate.temporal_operation is not None
            and candidate.observed_at is not None
            for candidate in candidates
        ):
            raise ExtractionError("TRANSITION requires an observed I-frame with temporal metadata")
        return decision


class ObservedDspyExtractors(DspyExtractors):
    @property
    def semantic_producer_capability(self):
        return self._cfg.write.semantic_producer_contract

    def align_claim_identity(self, new_claim, candidates, *, context, projection):
        from gluomem.llm.observed_identity import identity_proposal_model
        from gluomem.llm.observed_signatures import identity_alignment_module

        if not projection.targets:
            raise ExtractionError("Empty identity context is a preparation-owned no-call path")
        projection.validate_candidates(candidates, context)
        tc = self._cfg.dspy.tasks["identity_alignment"]
        base = self._module("identity_alignment")
        if (
            self._cfg.dspy.retry.max_attempts != 1
            or self._cfg.dspy.validation_retries
            or (tc.refine or {}).get("enabled")
            or type(base) not in {dspy.Predict, dspy.ChainOfThought}
            or len(base.predictors()) != 1
            or any(base.named_sub_modules(type_=dspy.Refine, skip_compiled=False))
        ):
            raise ExtractionError("Identity alignment requires one native attempt")
        module = identity_alignment_module(
            base, projection, context, producer_contract=self.semantic_producer_capability
        )
        for predictor in module.predictors():
            predictor.lm = None
            predictor.config = {
                k: v
                for k, v in predictor.config.items()
                if k not in {"n", "max_tokens", "max_completion_tokens"}
            }
            predictor.config["n"] = 1
            if tc.max_tokens is not None:
                predictor.config["max_tokens"] = tc.max_tokens
        from gluomem.llm.observed_identity import PROPERTY_IDENTITY_CONTRACT

        inputs = {}
        if self.semantic_producer_capability in (
            PROPERTY_IDENTITY_CONTRACT,
            "observed-assertion-scope-v4",
        ):
            from gluomem.llm.property_identity import property_slot_context

            inputs["property_slot_context"] = property_slot_context(projection, context, candidates)
        pred = self._run(
            "identity_alignment",
            _module_override=module,
            new_claim=new_claim,
            new_claim_context=context,
            identity_context=projection,
            **inputs,
        )
        return identity_proposal_model(
            projection, context, producer_contract=self.semantic_producer_capability
        ).model_validate(pred.proposal.model_dump())


__all__ = ["DspyExtractors", "ObservedDspyExtractors", "build_module", "make_extractors"]
