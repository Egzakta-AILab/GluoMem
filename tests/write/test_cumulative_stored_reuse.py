"""Cumulative exact reuse: authored per-claim menus and real atomic storage."""

from copy import deepcopy
from datetime import datetime

import jsonschema
import pytest

from gluomem.config.app_config import AppConfig
from gluomem.frames.temporal import TemporalObservation, TemporalStore
from gluomem.llm.outputs import (
    CognifyResult,
    ExtractedClaimSpec,
    ProjectedConsolidationDecision,
    projected_decision_model,
)
from gluomem.write import gate
from gluomem.write.events import (
    SourceEvent,
    SourceEventMessage,
    commit_source_event,
    prepare_source_event,
)
from gluomem.write.gate import ScanVerdict
from tests._source_helpers import UnusedExtractorLanes


@pytest.fixture
def cfg():
    config = AppConfig(dspy={"model": "mock"}, _env_file=None, _secrets_dir=None)
    config.write.cognify.enabled = False
    return config


class SafeScanner:
    def scan(self, content, *, context=""):
        return ScanVerdict(score=0.0)


def choice(action, fid=None, index=None):
    return {
        "action": action,
        "target_frame_id": fid,
        "target_claim_index": index,
        "rationale": "Authored offline selection.",
    }


def spec(text, index, date=None, **options):
    return ExtractedClaimSpec(
        text=text, message_id=f"m{index}", verbatim_span=text, event_date=date, **options
    )


def envelope(specs, **options):
    return SourceEvent(
        event_id="review-menu",
        source_ref="review:menu",
        capture_policy="trusted-only",
        created_at="2026-11-03T08:00:00Z",
        messages=[
            SourceEventMessage(
                message_id=s.message_id,
                content=s.verbatim_span,
                timestamp="2026-11-03T07:00:00Z",
            )
            for s in specs
        ],
        **options,
    )


class AuthoredSelections(UnusedExtractorLanes):
    consolidation_projection_protocol = "evolving-per-claim-v1"

    def __init__(self, specs, choices, inspect=None):
        self.specs = specs
        self.choices = choices
        self.inspect = inspect
        self.menus = []

    def extract_event_claims(
        self, source_text: str, *, reference_date: datetime | None = None
    ) -> list[ExtractedClaimSpec]:
        return self.specs

    def consolidate_claim_with_projection(self, text, candidates, *, context, projection):
        self.menus.append(projection)
        if self.inspect is not None:
            self.inspect(projection)
        selected = self.choices[projection.claim_index]
        model = projected_decision_model(projection)
        jsonschema.validate(selected, model.model_json_schema())
        return model.model_validate(selected)

    def cognify(self, text):
        return CognifyResult()


def seeded(store):
    old = store.create_i_frame("prior", "Nora workshop is in Oslo.")
    TemporalStore(store).record_observation(
        old.id,
        TemporalObservation(observed_at="2025-01-02", effective_at="2025-01-01"),
    )
    return old


def rejects(projection, selected, boundary):
    target = next(
        t
        for t in projection.targets
        if (t.frame_id, t.claim_index)
        == (selected["target_frame_id"], selected["target_claim_index"])
    )
    if boundary == "menu":
        assert selected["action"] not in target.allowed_actions
    elif boundary == "schema":
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(selected, projected_decision_model(projection).model_json_schema())
    else:
        assert boundary == "parser"
        with pytest.raises(ValueError, match="outside this claim's legal projection"):
            projected_decision_model(projection).model_validate(selected)


ACTIVATION_CASES = [
    pytest.param("candidate", None, {"pinned": True}, id="pinned-held"),
    pytest.param("active", "2001-01-01T00:00:00Z", {"pinned": True}, id="pinned-expired"),
    pytest.param("candidate", None, {"source_type": "explicit_memory"}, id="verified-held"),
]


class FrozenScores:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.calls = 0

    def scan(self, content, *, context=""):
        self.calls += 1
        return gate.ScanVerdict(score=next(self.scores))


@pytest.mark.parametrize("status,expiry,options", ACTIVATION_CASES)
@pytest.mark.parametrize("empty_updates", [1, 2], ids=["one-no-update", "two-no-updates"])
def test_cumulative_stored_activation_survives_quarantine_and_fresh_choice_commits(
    frame_store, cfg, monkeypatch, status, expiry, options, empty_updates
):
    old = seeded(frame_store)
    held = frame_store.create_i_frame(
        "held", "Nora workshop is in Paris.", status=status, expires_at=expiry
    )
    specs = [spec(held.content, i) for i in range(empty_updates + 1)]
    specs.append(spec("Nora Paris workshop has a copper roof.", len(specs)))
    choices = [choice("ADD")] * empty_updates
    choices += [choice("SUPERSEDE", old.id), choice("ADD")]
    scanner = FrozenScores([0.0, *([1.0] * empty_updates), 0.0])
    cfg.safety.fail_mode = "quarantine"
    admission_trace = []
    admit = gate.admit

    def observed_admit(*args, **kwargs):
        result = admit(*args, **kwargs)
        admission_trace.append(result)
        return result

    monkeypatch.setattr(gate, "admit", observed_admit)
    before = frame_store.all_frames()

    def inspect(projection):
        assert frame_store.all_frames() == before
        assert not any(t.conditional for t in projection.targets)
        if projection.claim_index == len(specs) - 1:
            assert next(t for t in projection.targets if t.frame_id == old.id).allowed_actions == ()
            for action in ("NOOP", "REFINE", "SUPERSEDE"):
                for boundary in ("menu", "schema", "parser"):
                    rejects(projection, choice(action, old.id), boundary)

    facade = AuthoredSelections(specs, choices, inspect)
    plan = prepare_source_event(
        envelope(specs, **options),
        store=frame_store,
        cfg=cfg,
        scanner=scanner,
        extractors=facade,
    )
    assert frame_store.all_frames() == before
    assert scanner.calls == len(specs)
    frozen = deepcopy((plan.candidate_revisions, plan.temporal_revisions))
    assert [d.model_dump() for d in plan.decisions] == choices
    result = commit_source_event(plan, store=frame_store)
    assert (plan.candidate_revisions, plan.temporal_revisions) == frozen
    assert [d.model_dump() for d in plan.decisions] == choices
    assert scanner.calls == len(specs)
    assert len(admission_trace) == len(specs)
    first = admission_trace[0]
    assert first.deduped and first.frame is not None and first.frame.id == held.id
    assert first.policy_updates == {
        "status": "active",
        "pinned": options.get("pinned", False),
        "verified_by_user": options.get("source_type") == "explicit_memory",
        "expires_at": None,
    }
    for reused in admission_trace[1:-1]:
        assert reused.deduped and reused.frame is not None and reused.frame.id == held.id
        assert reused.policy_updates == {}
        assert reused.frame.status == "active" and reused.frame.trust == held.trust
    current = frame_store.get_frame(held.id)
    assert current.status == "active" and current.trust == held.trust
    assert current.pinned is options.get("pinned", False)
    assert current.verified_by_user is (options.get("source_type") == "explicit_memory")
    assert current.expires_at is None and current.supersedes == old.id
    retired = frame_store.get_frame(old.id)
    assert retired.status == "superseded" and retired.content == "[forgotten]"
    replacement, detail = result.frames
    assert replacement.id == held.id and detail.content == specs[-1].text
    assert {p.message_id: p.frame_id for p in result.provenance} == {
        **{s.message_id: held.id for s in specs[:-1]},
        specs[-1].message_id: detail.id,
    }


@pytest.mark.parametrize("status,expiry,options", ACTIVATION_CASES)
def test_original_cumulative_noop_choice_is_refused_without_repair(
    frame_store, cfg, status, expiry, options
):
    old = seeded(frame_store)
    held = frame_store.create_i_frame(
        "held", "Nora workshop is in Paris.", status=status, expires_at=expiry
    )
    specs = [spec(held.content, 0), spec(held.content, 1), spec("Nora workshop detail.", 2)]
    choices = [choice("ADD"), choice("SUPERSEDE", old.id), choice("NOOP", old.id)]
    cfg.safety.fail_mode = "quarantine"

    class UncheckedSelections(AuthoredSelections):
        def consolidate_claim_with_projection(self, text, candidates, *, context, projection):
            self.menus.append(projection)
            return ProjectedConsolidationDecision(**self.choices[projection.claim_index])

    facade = UncheckedSelections(specs, choices)
    before = frame_store.all_frames()
    with pytest.raises(ValueError, match="outside this claim's legal projection"):
        prepare_source_event(
            envelope(specs, **options),
            store=frame_store,
            cfg=cfg,
            scanner=FrozenScores([0.0, 1.0, 0.0]),
            extractors=facade,
        )
    assert len(facade.menus) == len(specs) and facade.choices == choices
    assert next(t for t in facade.menus[-1].targets if t.frame_id == old.id).allowed_actions == ()
    assert frame_store.all_frames() == before
    assert frame_store._backend.scalar("SELECT COUNT(*) FROM source_events") == 0
    assert frame_store._backend.scalar("SELECT COUNT(*) FROM memory_provenance") == 0


@pytest.mark.parametrize("status,expiry,options", ACTIVATION_CASES)
@pytest.mark.parametrize("existing_quarantine", [False, True])
def test_cumulative_quarantined_reuse_never_invents_activation(
    frame_store, cfg, status, expiry, options, existing_quarantine
):
    old = seeded(frame_store)
    held = frame_store.create_i_frame(
        "held",
        "Nora workshop is in Paris.",
        status=status,
        expires_at=expiry,
        trust="quarantined" if existing_quarantine else "normal",
    )
    specs = [spec(held.content, 0), spec(held.content, 1), spec("Nora Oslo is reaffirmed.", 2)]
    choices = [choice("ADD"), choice("SUPERSEDE", old.id), choice("NOOP", old.id)]
    cfg.safety.fail_mode = "quarantine"
    facade = AuthoredSelections(specs, choices)
    before = frame_store.all_frames()
    plan = prepare_source_event(
        envelope(specs, **options),
        store=frame_store,
        cfg=cfg,
        scanner=FrozenScores([0.0 if existing_quarantine else 1.0, 1.0, 0.0]),
        extractors=facade,
    )
    assert frame_store.all_frames() == before
    assert not any(t.conditional for t in facade.menus[-1].targets)
    assert (
        "NOOP" in next(t for t in facade.menus[-1].targets if t.frame_id == old.id).allowed_actions
    )
    frozen = deepcopy((plan.candidate_revisions, plan.temporal_revisions))
    commit_source_event(plan, store=frame_store)
    assert frame_store.get_frame(held.id) == held
    assert frame_store.get_frame(old.id) == old
    assert (plan.candidate_revisions, plan.temporal_revisions) == frozen
    assert [d.model_dump() for d in plan.decisions] == choices


@pytest.mark.parametrize("options", [{"pinned": True}, {"source_type": "explicit_memory"}])
def test_cumulative_activation_reusing_the_same_identity_does_not_consume_it(
    frame_store, cfg, options
):
    held = seeded(frame_store)
    specs = [spec(held.content, 0), spec(held.content, 1), spec("Nora Paris is reaffirmed.", 2)]
    choices = [choice("ADD"), choice("SUPERSEDE", held.id), choice("NOOP", held.id)]
    cfg.safety.fail_mode = "quarantine"
    facade = AuthoredSelections(specs, choices)
    plan = prepare_source_event(
        envelope(specs, **options),
        store=frame_store,
        cfg=cfg,
        scanner=FrozenScores([0.0, 1.0, 0.0]),
        extractors=facade,
    )
    assert not any(t.conditional for t in facade.menus[-1].targets)
    result = commit_source_event(plan, store=frame_store)
    assert len(result.frames) == 1 and result.frames[0].id == held.id
    assert result.frames[0].status == "active" and result.frames[0].supersedes is None
    assert len(result.provenance) == len(specs)
