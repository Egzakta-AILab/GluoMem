"""SDK-free real SQLite publication and erasure controls for receipt coherence."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import UTC, datetime

import pytest

from gluomem.client.sdk import EmbeddedMemoryClient
from gluomem.config.app_config import AppConfig
from gluomem.embedding.providers import MockEmbedder
from gluomem.frames.lifecycle import forget_memory, retract_source
from gluomem.frames.semantics import SemanticStore
from gluomem.frames.store import Frame, FrameStore, _PreparedIndex
from gluomem.frames.temporal import TemporalPolicy
from gluomem.llm.assertion_scope import scoped_claim_type
from gluomem.llm.observed_identity import identity_proposal_model
from gluomem.llm.outputs import ExtractedClaimSpec, StoredCandidate, projected_decision_model
from gluomem.search.hybrid import HybridSearch
from gluomem.storage.db import MindDB
from gluomem.storage.scope import MemoryScope
from gluomem.write.events import (
    SourceEvent,
    SourceEventMessage,
    commit_source_event,
    prepare_source_event,
)
from gluomem.write.gate import ScanVerdict
from gluomem.write.receipt_coherence import ReceiptCoherentProjection
from tests._source_helpers import UnusedExtractorLanes

PRODUCER = "observed-assertion-scope-v4"
POLICY = "same-event-corroboration-v1"
CONTROLS = (
    "missing-quote",
    "wrong-role",
    "wrong-scope",
    "wrong-act",
    "unknown-legacy",
    "review-only",
    "quarantined",
    "mutated-source",
    "mutated-policy",
    "mutated-command",
    "late-mutation",
    "reentrant-retraction",
    "whole-batch-revocation",
    "independent-withdrawal",
    "hard-forget",
    "explicit-target-after-corroboration",
    "implicit-outside-bucket",
)


def assertion(text, predicate, value, act="direct_assertion"):
    return {
        "text": text,
        "message_id": "m",
        "verbatim_span": text,
        "event_date": None,
        "meaning": {
            "key": {"subject": "proof-cabinet", "predicate": predicate, "perspective": "user"},
            "value_key": value,
            "independently_versioned": True,
            "refines": None,
            "assertion_act": act,
        },
    }


def choice(action, frame_id=None):
    return {
        "action": action,
        "target_frame_id": frame_id,
        "target_claim_index": None,
        "rationale": "Authored SDK-free fixture.",
    }


def relation(kind, frame_id=None):
    linked = kind not in {"unrelated", "unresolved"}
    return {
        "relation": "same_assertion"
        if linked
        else "independent"
        if kind == "unrelated"
        else "unresolved",
        "assertion_relation": kind,
        "target_frame_id": frame_id,
        "target_claim_index": None,
        "subject_alignment": "same_label" if linked else "not_aligned",
        "property_alignment": "same_label" if linked else "not_aligned",
        "rationale": "Authored SDK-free fixture.",
    }


class Scanner:
    def __init__(self, blocked=None):
        self.blocked = blocked

    def scan(self, content, *, context=""):
        return ScanVerdict(score=1.0 if content == self.blocked else 0.0)


class Computation(UnusedExtractorLanes):
    writer_capability = "semantic-state-writer-v4"
    semantic_producer_capability = PRODUCER
    consolidation_projection_protocol = "evolving-per-claim-v1"

    def __init__(self, specs, decisions, proposals):
        self.specs, self.decisions, self.proposals = specs, decisions, proposals
        self.menus = []
        self.identities = []
        self.calls = []

    def extract_event_claims(
        self, source_text: str, *, reference_date: datetime | None = None
    ) -> list[ExtractedClaimSpec]:
        self.calls.append("extract")
        return [scoped_claim_type().model_validate(s) for s in self.specs]

    def align_claim_identity(self, text, candidates, *, context, projection):
        self.calls.append("identity")
        self.identities.append(projection)
        return identity_proposal_model(
            projection, context, producer_contract=PRODUCER
        ).model_validate(self.proposals[projection.claim_index])

    def consolidate_claim_with_projection(self, text, candidates, *, context, projection):
        self.calls.append("consolidate")
        self.menus.append(projection)
        return projected_decision_model(
            projection, writer_contract=self.writer_capability
        ).model_validate(self.decisions[projection.claim_index])


def snapshot(db):
    names = [
        r["name"]
        for r in db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    assert all(n.replace("_", "").isalnum() for n in names)
    return {
        n: sorted([dict(r) for r in db.fetchall('SELECT * FROM "' + n + '"')], key=repr)
        for n in names
    }


def digest(value):
    import hashlib

    def encode(v):
        if isinstance(v, bytes):
            return {"bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()}
        if hasattr(v, "model_dump"):
            return v.model_dump(mode="json")
        return asdict(v)

    return hashlib.sha256(json.dumps(value, default=encode, sort_keys=True).encode()).hexdigest()


class Fixture:
    def __init__(self, path):
        self.path = path
        self.cfg = AppConfig(
            dspy={
                "model": "mock",
                "adapter": "json",
                "schema_mode": "required",
                "cache": {"disk": False},
            },
            write={
                "writer_contract": "semantic-state-writer-v4",
                "semantic_producer_contract": PRODUCER,
                "semantic_reuse_policy": POLICY,
                "source_capture_policy": "attributed-interaction",
                "cognify": {"enabled": False},
            },
            _env_file=None,
            _secrets_dir=None,
        )
        self.cfg.search.abstain.enabled = False
        self.scope = MemoryScope(
            tenant_id="receipt", principal_id="receipt-controls", namespace="owned"
        )
        self.scanner = Scanner()
        self.db = MindDB(path, dim=384)
        self.store = FrameStore(
            self.db, MockEmbedder(dim=384), scope=self.scope, chunking=self.cfg.chunking
        )
        self.old = assertion("The proof cabinet access code is BRASS.", "access-code", "BRASS")
        self.sibling = assertion(
            "The proof cabinet stands beside the west wall.", "position", "west-wall"
        )
        self.fixed = assertion(
            "Correction: the proof cabinet access code is SILVER.", "access-code", "SILVER"
        )
        self.note = assertion(
            "The earlier proof cabinet BRASS code report was mistaken.",
            "access-code",
            "BRASS",
            "prior_mistake_report",
        )
        self.seed_plan, _ = self.prepare(
            "seed", [self.old, self.sibling], [choice("ADD"), choice("ADD")], when="2026-01-01"
        )
        self.seed = commit_source_event(self.seed_plan, store=self.store)
        self.old_id, self.sibling_id = [f.id for f in self.seed.frames]

    def prepare(self, eid, specs, decisions, proposals=None, *, when="2026-01-03", event=None):
        event = event or SourceEvent(
            event_id=eid,
            source_ref="authored:receipt:" + eid,
            scope=self.scope,
            capture_policy="attributed-interaction",
            created_at=when,
            messages=[
                SourceEventMessage(
                    message_id="m",
                    role="user",
                    timestamp=when,
                    content=" ".join(s["text"] for s in specs),
                )
            ],
        )
        ext = Computation(specs, decisions, proposals or [relation("unrelated") for _ in specs])
        plan = prepare_source_event(
            event, store=self.store, cfg=self.cfg, scanner=self.scanner, extractors=ext
        )
        return plan, ext

    def correction(self, specs=None, decisions=None, proposals=None, **kwargs):
        return self.prepare(
            "correction",
            specs or [self.fixed, self.note, self.sibling],
            decisions or [choice("SUPERSEDE", self.old_id), choice("ADD"), choice("ADD")],
            proposals
            or [
                relation("mistaken_assertion", self.old_id),
                relation("unresolved"),
                relation("unrelated"),
            ],
            **kwargs,
        )

    def reader(self):
        # Only ranking is fixture-disabled. Actual FTS/index/provenance/recall remain production.
        self.db.chunk_vec.search = lambda *a, **k: []
        self.db.frame_vec.search = lambda *a, **k: []
        policy = TemporalPolicy(allow_historical=True)
        search = HybridSearch(
            self.db,
            self.store,
            self.cfg,
            embedder=self.store.embedder,
            temporal_policy=policy,
            clock=lambda: datetime(2026, 11, 11, tzinfo=UTC),
        )
        return EmbeddedMemoryClient(
            self.db, self.store, cfg=self.cfg, search=search, temporal_policy=policy
        )

    def assert_publication(self, plan, result):
        assert [d.action for d in plan.decisions] == ["SUPERSEDE", "ADD", "ADD"]
        new = next(f for f in result.frames if f.content == self.fixed["text"])
        retired = self.store.get_frame(self.old_id)
        assert retired.content == "[forgotten]" and retired.status == "superseded"
        assert retired.superseded_by == new.id and new.supersedes == self.old_id
        assert SemanticStore(self.store).snapshot(self.old_id, producer_contract=PRODUCER) is None
        live = SemanticStore(self.store).snapshot(self.sibling_id, producer_contract=PRODUCER)
        assert live and len(live.receipts) == 1
        assert live.receipts[0].source_event_row_id == result.event_row_id
        observed_at = live.receipts[0].witness.observed_at
        assert observed_at is not None and observed_at.startswith("2026-01-03")
        assert (
            self.db.scalar(
                "SELECT COUNT(*) FROM memory_provenance WHERE source_event_id=?",
                (result.event_row_id,),
            )
            == 3
        )
        assert (
            self.db.scalar(
                "SELECT payload_json FROM source_events WHERE id=?", (self.seed.event_row_id,)
            )
            == "{}"
        )
        assert (
            self.db.scalar(
                "SELECT COUNT(*) FROM memory_frame_lineage WHERE source_frame_id=? "
                "AND derived_frame_id=? AND operation='supersede'",
                (self.old_id, new.id),
            )
            == 1
        )
        evidence = self.reader().recall("proof cabinet", k=10)
        assert {f.id for f in result.frames} <= {h.frame.id for h in evidence.hits}
        assert self.old_id not in {h.frame.id for h in evidence.hits}
        assert self.fixed["text"] in evidence.context and self.note["text"] in evidence.context
        assert self.sibling["text"] in evidence.context and self.old["text"] not in evidence.context
        assert all(p.span_verified for h in evidence.hits for p in h.frame.provenance)
        return live, evidence

    def readback(self):
        before = snapshot(self.db)
        with MindDB(self.path, dim=384) as observer:
            assert snapshot(observer) == before
        return {
            "tables": sorted(before),
            "snapshot_sha256": digest(before),
            "table_counts": {n: len(rows) for n, rows in before.items()},
        }


def exercise(case, path):
    f = Fixture(path)
    proof = {"case_id": case, "passed": False}
    try:
        before = snapshot(f.db)
        if case in {"independent-withdrawal", "hard-forget", "useful-complete"}:
            plan, ext = f.correction()
            result = commit_source_event(plan, store=f.store)
            live, evidence = f.assert_publication(plan, result)
            assert len(plan.receipt_support) == 1 and plan.receipt_support[0].claim_index == 2
            sibling_projection = plan.projections[2]
            assert isinstance(sibling_projection, ReceiptCoherentProjection)
            assert sibling_projection.untargeted_actions == ("ADD",)
            for projection in plan.projections:
                assert projection is not None
                assert not any(t.claim_index == 2 for t in projection.targets)
            committed = f.readback()
            duplicate_before = snapshot(f.db)
            duplicate = prepare_source_event(
                plan.event,
                store=f.store,
                cfg=f.cfg,
                scanner=f.scanner,
                extractors=Computation([], [], []),
            )
            replay = commit_source_event(duplicate, store=f.store)
            assert replay.deduplicated and replay.event_row_id == result.event_row_id
            assert snapshot(f.db) == duplicate_before
            if case == "independent-withdrawal":
                retract_source(f.store, "authored:receipt:seed")
                after = SemanticStore(f.store).snapshot(f.sibling_id, producer_contract=PRODUCER)
                assert after and len(after.receipts) == 1 and after.receipts[0] == live.receipts[0]
                assert f.sibling["text"] in f.reader().recall("proof cabinet", k=10).context
                early = f.reader().recall("proof cabinet", k=10, known_at="2026-01-02")
                assert f.sibling_id not in {h.frame.id for h in early.hits}
                retract_source(f.store, "authored:receipt:correction")
                assert (
                    SemanticStore(f.store).snapshot(f.sibling_id, producer_contract=PRODUCER)
                    is None
                )
                assert f.store.get_frame(f.sibling_id).content == "[forgotten]"
                assert not f.reader().recall("proof cabinet", k=10).context
                with pytest.raises(ValueError, match="removed|retract|eras"):
                    prepare_source_event(
                        plan.event, store=f.store, cfg=f.cfg, scanner=f.scanner, extractors=ext
                    )
            elif case == "hard-forget":
                forget_memory(f.store, f.sibling_id)
                assert (
                    SemanticStore(f.store).snapshot(f.sibling_id, producer_contract=PRODUCER)
                    is None
                )
                assert f.store.get_frame(f.sibling_id).content == "[forgotten]"
                assert f.sibling["text"] not in f.reader().recall("proof cabinet", k=10).context
                assert (
                    f.db.scalar(
                        "SELECT COUNT(*) FROM memory_claim_semantics WHERE frame_id=?",
                        (f.sibling_id,),
                    )
                    == 0
                )
            proof.update(
                committed=committed,
                live=live.model_dump(mode="json"),
                recall=evidence.model_dump(mode="json"),
            )
        elif case in {
            "missing-quote",
            "wrong-role",
            "wrong-scope",
            "wrong-act",
            "unknown-legacy",
            "review-only",
            "quarantined",
        }:
            sibling = deepcopy(f.sibling)
            event = None
            if case == "missing-quote":
                sibling["verbatim_span"] = "A fresh quote absent from this envelope."
                expected = "quote|span|source"
            elif case == "wrong-role":
                sibling["meaning"]["key"]["perspective"] = "assistant"
                expected = "role|perspective"
            elif case == "wrong-act":
                sibling["meaning"]["assertion_act"] = "prior_mistake_report"
                expected = "No receipt-feasible|incompatible"
            elif case == "unknown-legacy":
                # Legacy family is valid but grants no assertion-scope authority.
                legacy = deepcopy(f.sibling["meaning"])
                legacy.pop("assertion_act")
                f.db.execute(
                    "UPDATE memory_claim_semantics SET meaning_json=? WHERE frame_id=?",
                    (json.dumps(legacy), f.sibling_id),
                )
                before = snapshot(f.db)
                expected = "No receipt-feasible|incompatible"
            elif case == "review-only":
                sibling["confidence"] = 0.1
                expected = "No receipt-feasible|outside|review"
            elif case == "quarantined":
                f.scanner.blocked = sibling["text"]
                f.cfg.safety.fail_mode = "quarantine"
                expected = "No receipt-feasible|outside|review"
            else:
                plan, _ = f.correction()
                plan.event.scope = replace(f.scope, namespace="foreign")
                with pytest.raises(ValueError, match="input binding|context changed"):
                    commit_source_event(plan, store=f.store)
                assert snapshot(f.db) == before
                proof.update(
                    passed=True, readback=f.readback(), refusal="source-scope-input-binding"
                )
                return proof
            with pytest.raises(ValueError, match=expected) as refused:
                f.correction(specs=[f.fixed, f.note, sibling], event=event)
            assert snapshot(f.db) == before
            proof["refusal"] = str(refused.value)
        elif case in {
            "mutated-source",
            "mutated-policy",
            "mutated-command",
            "late-mutation",
            "reentrant-retraction",
        }:
            plan, _ = f.correction()
            if case == "mutated-source":
                plan.event.messages[0].content += " Unowned late change."
            elif case == "mutated-policy":
                plan.eff_cfg.write.semantic_reuse_policy = None
            elif case == "mutated-command":
                plan.receipt_support = ()
            callback = None
            original_ensure_indexed = f.store.ensure_indexed
            if case in {"late-mutation", "reentrant-retraction"}:
                seen = []

                def mutate(frame: Frame, *, prepared: _PreparedIndex | None = None) -> bool:
                    if not seen:
                        seen.append(frame.id)
                        if case == "late-mutation":
                            f.db.execute(
                                "UPDATE memory_claim_semantics SET binding_token='unowned' "
                                "WHERE frame_id=?",
                                (f.sibling_id,),
                            )
                        else:
                            retract_source(f.store, "authored:receipt:correction")
                    return original_ensure_indexed(frame, prepared=prepared)

                f.store.ensure_indexed = mutate
            with pytest.raises(
                ValueError, match="changed|retract|removed|surviv|incompatible|unknown"
            ) as refused:
                commit_source_event(plan, store=f.store, write_guard=callback)
            if case == "reentrant-retraction":
                assert (
                    f.db.scalar(
                        "SELECT COUNT(*) FROM source_retractions "
                        "WHERE source_ref='authored:receipt:correction'"
                    )
                    == 1
                )
                assert not f.db.scalar(
                    "SELECT COUNT(*) FROM source_events WHERE event_id='correction'"
                )
                assert not f.db.scalar(
                    "SELECT COUNT(*) FROM memory_frames WHERE content=?", (f.fixed["text"],)
                )
                assert not f.db.scalar(
                    "SELECT COUNT(*) FROM memory_claim_semantics "
                    "WHERE provenance_id NOT IN (SELECT id FROM memory_provenance)"
                )
            else:
                assert snapshot(f.db) == before
            proof["refusal"] = str(refused.value)
        elif case == "whole-batch-revocation":
            later = assertion(
                "Correction: the proof cabinet stands beside the east wall.",
                "position",
                "east-wall",
            )
            with pytest.raises(
                ValueError,
                match="Current-source corroboration must survive the complete selected batch",
            ) as refused:
                f.correction(
                    specs=[f.fixed, f.note, f.sibling, later],
                    decisions=[
                        choice("SUPERSEDE", f.old_id),
                        choice("ADD"),
                        choice("ADD"),
                        choice("SUPERSEDE", f.sibling_id),
                    ],
                    proposals=[
                        relation("mistaken_assertion", f.old_id),
                        relation("unresolved"),
                        relation("unrelated"),
                        relation("mistaken_assertion", f.sibling_id),
                    ],
                )
            assert snapshot(f.db) == before
            proof["refusal"] = str(refused.value)
        elif case == "implicit-outside-bucket":
            from gluomem.write import events

            original_candidates = events._retrieve_consolidation_candidates

            def omit_exact(
                store: FrameStore,
                prepared: _PreparedIndex,
                *,
                scope: MemoryScope,
                limit: int = 5,
            ) -> list[StoredCandidate]:
                return [
                    c
                    for c in original_candidates(store, prepared, scope=scope, limit=limit)
                    if c.frame_id != f.sibling_id
                ]

            events._retrieve_consolidation_candidates = omit_exact
            try:
                plan, _ = f.correction()
            finally:
                events._retrieve_consolidation_candidates = original_candidates
            assert f.sibling_id not in plan.candidate_revisions[2]
            assert plan.receipt_support[0].frame_id == f.sibling_id
            original_create = f.store.create_i_frame

            def mutate_support(*args, **kwargs):
                f.store._update(f.sibling_id, f.scope, trust="quarantined")
                return original_create(*args, **kwargs)

            f.store.create_i_frame = mutate_support
            with pytest.raises(ValueError, match="changed|corroboration"):
                commit_source_event(plan, store=f.store)
            assert snapshot(f.db) == before
            f.store.create_i_frame = original_create
            result = commit_source_event(plan, store=f.store)
            f.assert_publication(plan, result)
        elif case == "explicit-target-after-corroboration":
            plan, _ = f.correction(
                specs=[f.fixed, f.note, f.sibling, f.sibling],
                decisions=[
                    choice("SUPERSEDE", f.old_id),
                    choice("ADD"),
                    choice("ADD"),
                    choice("NOOP", f.sibling_id),
                ],
                proposals=[
                    relation("mistaken_assertion", f.old_id),
                    relation("unresolved"),
                    relation("unrelated"),
                    relation("state_observation", f.sibling_id),
                ],
            )
            result = commit_source_event(plan, store=f.store)
            live = SemanticStore(f.store).snapshot(f.sibling_id, producer_contract=PRODUCER)
            assert live and len(live.receipts) == 2
            assert len({r.provenance_id for r in live.receipts}) == 2
            assert len(result.frames) == 3
            proof["live"] = live.model_dump(mode="json")
        else:
            raise AssertionError("Unknown receipt-coherence case: " + case)
        proof.update(passed=True, readback=f.readback())
        return proof
    finally:
        f.db.close()


@pytest.mark.parametrize("case", ("useful-complete", *CONTROLS))
def test_receipt_coherence(case, tmp_path):
    assert exercise(case, tmp_path / "state.sqlite")["passed"]
