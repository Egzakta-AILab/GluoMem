"""Versioned semantic intersection over the unchanged canonical frame projection."""

from gluomem.frames.content_hash import content_hash, suppression_key
from gluomem.frames.provenance import is_suppressed_by_tombstone
from gluomem.frames.semantics import SemanticStore
from gluomem.llm.outputs import ConsolidationProjection, StoredCandidate
from gluomem.llm.semantic import SemanticProjection, TargetMeaning, permits
from gluomem.write.consolidation_projection import ClaimProjectionState


class SemanticProjectionState(ClaimProjectionState):
    def __init__(self):
        super().__init__()
        self.meanings = {}
        self.revisions = {}
        self.revoked_sources = set()
        self.event_receipts = set()
        self.event_revoked = False
        self.receipt_support = {}

    def observe(self, frame_id, snapshot):
        if frame_id in self.revisions and self.revisions[frame_id] != snapshot:
            raise ValueError("Semantic binding changed during preparation")
        self.revisions[frame_id] = snapshot

    def project(self, claim_index, candidates, claim, store, scope) -> ConsolidationProjection:
        legacy = [StoredCandidate.model_validate(c.model_dump()) for c in candidates]
        projection = super().project(claim_index, legacy, claim, store, scope)
        semantics = {c.frame_id: c.semantic for c in candidates}
        for fid, snapshot in semantics.items():
            self.observe(fid, snapshot)
        bindings = []
        targets = []
        for target in projection.targets:
            snapshot = semantics.get(target.frame_id)
            meaning = (
                (snapshot.meaning if snapshot else None)
                if target.frame_id is not None
                else self.meanings[target.claim_index]
            )
            key = (target.frame_id, target.claim_index)
            surviving = (
                not self.event_revoked and (target.frame_id is None or key in self.event_receipts)
            ) or (
                snapshot is not None
                and any(
                    r.source_event_row_id not in self.revoked_sources for r in snapshot.receipts
                )
            )
            bindings.append(
                TargetMeaning(
                    frame_id=target.frame_id,
                    claim_index=target.claim_index,
                    meaning=meaning,
                    semantic=snapshot,
                )
            )
            targets.append(
                target.model_copy(
                    update={
                        "allowed_actions": tuple(
                            a
                            for a in target.allowed_actions
                            if surviving and permits(a, claim.meaning, meaning)
                        )
                    }
                )
            )
        return SemanticProjection(
            claim_index=claim_index,
            targets=tuple(targets),
            meaning=claim.meaning,
            target_meanings=tuple(bindings),
        )

    def apply(self, claim_index, claim, decision, source, *, event, cfg, verdict, store, scope):
        suppressed = is_suppressed_by_tombstone(
            store._backend,
            scope=scope,
            content_hash_val=content_hash(claim.content),
            suppression_key_val=suppression_key(claim.content),
            event_id=event.event_id,
            force_reassert=event.force_reassert,
        )
        match = None
        if decision.action != "NOOP" and not suppressed:
            from gluomem.write.events import _gop_for_source_ref

            gop = (
                store.get_frame(decision.target_frame_id, scope=scope).gop_id
                if decision.action == "REFINE" and decision.target_frame_id is not None
                else _gop_for_source_ref(event.source_ref)
            )
            match = store.find_frame_by_hash(
                gop if decision.action == "REFINE" else None, claim.content, scope=scope
            )
            if match is not None and (match.id, None) not in self.consumed:
                binding = SemanticStore(store).require_reuse(match.id, claim.meaning, scope=scope)
                self.observe(match.id, binding)
                if not (
                    any(r.source_event_row_id not in self.revoked_sources for r in binding.receipts)
                    or ((match.id, None) in self.event_receipts and not self.event_revoked)
                ):
                    if getattr(cfg.write, "semantic_reuse_policy", None) is None:
                        raise ValueError(
                            "Semantic reuse requires review: revoked assertion binding"
                        )
                    from gluomem.write.receipt_coherence import current_support

                    command = current_support(
                        claim_index,
                        claim,
                        decision,
                        match,
                        binding,
                        event=event,
                        cfg=cfg,
                        verdict=verdict,
                        store=store,
                        scope=scope,
                    )
                    if command is None or self.event_revoked:
                        raise ValueError(
                            "Semantic reuse requires review: revoked assertion binding"
                        )
                    self.receipt_support[claim_index] = command
            for prior in self.admissions:
                if (
                    prior.content_hash == content_hash(claim.content)
                    and (None, prior.claim_index) not in self.consumed
                    and (decision.action != "REFINE" or prior.gop_id == gop)
                    and self.meanings.get(prior.claim_index) != claim.meaning
                ):
                    raise ValueError(
                        "Semantic reuse requires review: incompatible event-local assertion binding"
                    )
        super().apply(
            claim_index,
            claim,
            decision,
            source,
            event=event,
            cfg=cfg,
            verdict=verdict,
            store=store,
            scope=scope,
        )
        target_key = (decision.target_frame_id, decision.target_claim_index)
        if not suppressed and (
            decision.action == "NOOP"
            or (
                match is not None
                and (match.id, None) not in self.consumed
                and not (
                    verdict.score >= cfg.safety.block_threshold and cfg.safety.fail_mode == "reject"
                )
            )
        ):
            if decision.action == "NOOP":
                self.event_receipts.add(target_key)
            else:
                assert match is not None  # Required by the reuse condition above.
                self.event_receipts.add((match.id, None))
        if (
            decision.action == "SUPERSEDE"
            and (decision.target_frame_id, decision.target_claim_index) in self.consumed
        ):
            # Project only named context removals; frozen observations stay intact.
            if decision.target_frame_id is None or target_key in self.event_receipts:
                self.event_revoked = True
            if decision.target_frame_id is not None:
                from gluomem.storage.scope import scope_sql

                self.revoked_sources.update(
                    int(r["source_event_id"])
                    for r in store._backend.fetchall(
                        f"SELECT source_event_id FROM memory_provenance WHERE {scope_sql()} "
                        "AND source_event_id IS NOT NULL "
                        "AND frame_id=?",
                        (*scope.as_tuple(), decision.target_frame_id),
                    )
                )
        if not claim.meaning.independently_versioned:
            self.publications = [p for p in self.publications if p.claim_index != claim_index]
        self.meanings[claim_index] = claim.meaning
