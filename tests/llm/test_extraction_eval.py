"""Frozen source-backed extraction evaluator: golden scores, splits, leak guards.

Every expectation here is computed offline from the authored fixture; no
model is called. Golden rows pin the scalar and the error codes, so a change
to matching rules or lexicons shows up as a diff against this table.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from gluomem.llm.extraction_eval import (
    FIXTURE_PATH,
    SEALED_SPLIT,
    FixtureError,
    SealedSplitError,
    evaluator_fingerprint,
    load_fixture,
    normalize,
    parse_fixture,
    score_case,
    scorecard,
)
from gluomem.llm.outputs import ExtractedEvent, ExtractedFact


def _facts(*texts: str) -> list[dict]:
    return [{"text": t, "confidence": 1.0} for t in texts]


def _events(*rows: tuple[str, str | None]) -> list[dict]:
    return [{"text": t, "event_date": d, "confidence": 1.0} for t, d in rows]


ANA_GOLD = _facts(
    "Ana moved from Belgrade to Novi Sad in March 2026.",
    "Marko lives in Belgrade.",
    "Ana does not own a car.",
    "Ana might buy a bicycle next year.",
)

# (case_id, prediction, expected scalar, expected error counts)
GOLDEN = [
    pytest.param("en-ana-move", ANA_GOLD, 1.0, {}, id="verbatim"),
    pytest.param(
        "en-ana-move",
        _facts(
            "Ana relocated to Novi Sad.",
            "Ana's brother Marko is based in Belgrade.",
            "Ana doesn't have a car.",
            "Ana is considering getting a bike.",
        ),
        1.0,
        {},
        id="supported-paraphrase",
    ),
    pytest.param(
        "en-ana-move",
        _facts("Ana lives in Paris.", "Ana works at a bank."),
        0.0,
        {"unsupported": 2},
        id="valid-but-unsupported",
    ),
    pytest.param(
        "en-ana-move",
        _facts("Marko moved to Novi Sad.", "Marko lives in Belgrade.",
               "Ana does not own a car.", "Ana might buy a bicycle next year."),
        0.75,
        {"wrong_subject": 1},
        id="wrong-subject",
    ),
    pytest.param(
        "en-ana-move",
        _facts("She moved to Novi Sad.", "Marko lives in Belgrade.",
               "Ana does not own a car.", "Ana might buy a bicycle next year."),
        0.75,
        {"missing_subject": 1},
        id="pronoun-subject",
    ),
    pytest.param(
        "en-ana-move",
        _facts("Ana moved to Novi Sad.", "Marko lives in Belgrade.",
               "Ana owns a car.", "Ana might buy a bicycle next year."),
        0.75,
        {"negation_lost": 1},
        id="negation-loss",
    ),
    pytest.param(
        "en-ana-move",
        _facts("Ana moved to Novi Sad.", "Marko lives in Belgrade.",
               "Ana does not own a car.", "Ana bought a bicycle."),
        0.75,
        {"hypothetical_as_completed": 1},
        id="hypothetical-as-completed",
    ),
    pytest.param(
        "en-ana-move",
        [*ANA_GOLD, {"text": "Ana lives in Belgrade."}],
        0.8889,
        {"stale_as_current": 1},
        id="stale-as-current",
    ),
    pytest.param(
        "en-ana-move",
        [*ANA_GOLD, {"text": "Ana previously lived in Belgrade."}],
        1.0,
        {},
        id="superseded-marked-previous",
    ),
    pytest.param(
        "en-ana-move",
        _facts("Ana moved to Novi Sad.", "Marko used to live in Belgrade.",
               "Ana does not own a car.", "Ana might buy a bicycle next year."),
        0.75,
        {"current_as_stale": 1},
        id="current-as-stale",
    ),
    pytest.param(
        "sr-latn-ana-move",
        _facts(
            "Ana se preselila iz Beograda u Novi Sad.",
            "Marko živi u Beogradu.",
            "Ana nema automobil.",
            "Ana će možda kupiti bicikl.",
        ),
        1.0,
        {},
        id="sr-latn-native",
    ),
    pytest.param(
        "sr-latn-ana-move",
        _facts("Ana se preselila u Novi Sad.", "Marko živi u Beogradu.",
               "Ana ima automobil.", "Ana je kupila bicikl."),
        0.5,
        {"negation_lost": 1, "hypothetical_as_completed": 1},
        id="sr-latn-negation-and-modality",
    ),
    pytest.param("en-greeting", [], 1.0, {}, id="correct-abstention"),
    pytest.param(
        "sr-cyrl-greeting", _facts("Корисник поздравља."), 0.0, {"unsupported": 1},
        id="spurious-on-small-talk",
    ),
    pytest.param("en-ana-move", [], 0.0, {}, id="over-abstention"),
    pytest.param(
        "en-demo-dates",
        _events(("Project kickoff", "2026-02-03T00:00:00"),
                ("Client demo", "2026-03-24T00:00:00")),
        1.0,
        {},
        id="events-current-dates",
    ),
    pytest.param(
        "en-demo-dates",
        _events(("Project kickoff", "2026-02-03T00:00:00"),
                ("Client demo", "2026-03-10T00:00:00")),
        0.5,
        {"stale_as_current": 1},
        id="events-stale-date-as-current",
    ),
    pytest.param(
        "en-demo-dates",
        _events(("Project kickoff", "2026-02-04T00:00:00"),
                ("Client demo", None)),
        0.0,
        {"wrong_date": 1, "missing_date": 1},
        id="events-wrong-and-missing-date",
    ),
    pytest.param(
        "sr-cyrl-plant-migration",
        _events(("Фабрика у Крагујевцу је прешла на SAP.", "2025-03-01T00:00:00")),
        1.0,
        {},
        id="sr-cyrl-month-precision",
    ),
    pytest.param(
        "en-plant-migration",
        _facts("The Kragujevac plant migrated its ERP to SAP in 2025.",
               "The Čačak plant has not migrated yet.",
               "The Čačak plant plans to migrate in 2027."),
        1.0,
        {},
        id="plan-kept-as-plan",
    ),
]


@pytest.mark.parametrize(("case_id", "prediction", "expected", "errors"), GOLDEN)
def test_golden_scores(case_id: str, prediction: list, expected: float, errors: dict) -> None:
    case = load_fixture().case(case_id)
    result = score_case(case, prediction)
    assert result.score == pytest.approx(expected, abs=1e-4)
    assert dict(result.errors) == errors
    # Reproducible: identical inputs give identical scalar, components, feedback.
    again = score_case(case, prediction)
    assert (again.score, again.components, again.feedback) == (
        result.score,
        result.components,
        result.feedback,
    )


def test_unsupported_valid_output_is_penalized_not_parse_credited() -> None:
    case = load_fixture().case("en-ana-move")
    result = score_case(case, [ExtractedFact(text="Ana lives in Paris.")])
    assert result.components["schema_validity"] == 1.0
    assert result.components["support_precision"] == 0.0
    assert result.score == 0.0


def test_error_kinds_are_distinguishable_with_targeted_feedback() -> None:
    case = load_fixture().case("en-ana-move")
    cases = {
        "wrong_subject": ("Marko moved to Novi Sad.", "attributes to Marko"),
        "negation_lost": ("Ana owns a car.", "drops the negation"),
        "hypothetical_as_completed": ("Ana bought a bicycle.", "plan or possibility as done"),
        "stale_as_current": ("Ana lives in Belgrade.", "superseded state as current"),
        "unsupported": ("Ana works at a bank.", "not stated anywhere"),
    }
    for code, (text, advice) in cases.items():
        result = score_case(case, _facts(text))
        assert set(result.errors) == {code}, code
        assert advice in result.feedback, code


def test_components_separate_validity_support_and_fidelity() -> None:
    case = load_fixture().case("en-ana-move")
    result = score_case(
        case, _facts("Ana moved to Novi Sad.", "Ana owns a car.", "Ana works at a bank.")
    )
    c = result.components
    assert c["schema_validity"] == 1.0
    assert c["support_precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert c["support_recall"] == 0.5
    assert c["polarity_fidelity"] == 0.0
    assert c["faithful_precision"] == pytest.approx(1 / 3, abs=1e-4)
    assert c["faithful_recall"] == 0.25
    assert c["time_fidelity"] is None  # not applicable to fact cases
    assert c["abstention"] is None


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "Ana moved.",
        [{"text": ""}],
        [{"text": "x", "confidence": 2}],
        [{"text": "x", "event_date": "not-a-date"}],
        [42],
    ],
)
def test_invalid_schema_scores_zero(bad: object) -> None:
    result = score_case(load_fixture().case("en-ana-move"), bad)
    assert result.score == 0.0
    assert result.components["schema_validity"] == 0.0
    assert result.errors == {"invalid_schema": 1}


def test_typed_outputs_are_accepted() -> None:
    fx = load_fixture()
    facts = [ExtractedFact(text=d["text"]) for d in ANA_GOLD]
    assert score_case(fx.case("en-ana-move"), facts).score == 1.0
    events = [
        ExtractedEvent(text="Project kickoff", event_date=datetime(2026, 2, 3)),
        ExtractedEvent(text="Client demo", event_date=datetime(2026, 3, 24)),
    ]
    assert score_case(fx.case("en-demo-dates"), events).score == 1.0


def test_feedback_never_exposes_reasoning_or_secrets() -> None:
    case = load_fixture().case("en-ana-move")
    prediction = [
        {"text": "Ana's api_key=sk-live-abcdef1234567890 is stored.", "reasoning": "HIDDEN-COT"},
        {"text": "Bearer abc.def.ghi unlocks it."},
    ]
    result = score_case(case, prediction)
    assert "HIDDEN-COT" not in result.feedback
    assert "sk-live" not in result.feedback
    assert "abc.def.ghi" not in result.feedback
    assert "[redacted]" in result.feedback
    # Receipts carry codes and ids, never the predicted text.
    assert "Ana's" not in json.dumps(result.record())


def test_normalize_folds_diacritics_and_cyrillic() -> None:
    assert normalize("Чачак, Đurđevdan, živi") == ["cacak", "djurdjevdan", "zivi"]
    assert normalize("Ana doesn't") == ["ana", "does", "not"]


# --------------------------------------------------------------------------
# Splits, sealing and the frozen fixture
# --------------------------------------------------------------------------


def test_split_manifest_groups_by_source_family() -> None:
    manifest = load_fixture().split_manifest()
    families = [set(s["source_families"]) for s in manifest["splits"].values()]
    for i, a in enumerate(families):
        for b in families[i + 1 :]:
            assert not a & b
    assert manifest["policy"]["group_by"] == "source_family"
    assert manifest["splits"][SEALED_SPLIT]["cases"]
    assert manifest["splits"]["train"]["cases"]
    assert manifest["splits"]["dev"]["cases"]
    # Both languages of one source family stay together.
    sealed = manifest["splits"][SEALED_SPLIT]["cases"]
    assert {"en-jelena-role", "sr-latn-jelena-role"} <= set(sealed)


def test_sealed_cases_refuse_optimization_and_hide_feedback() -> None:
    case = load_fixture().case("en-jelena-role")
    with pytest.raises(SealedSplitError):
        score_case(case, _facts("Jelena leads the payments team."))
    report = score_case(case, _facts("Jelena leads the payments team."), purpose="report")
    assert report.score > 0
    assert report.feedback == ""


def test_case_inputs_carry_no_gold() -> None:
    for case in load_fixture().cases:
        inputs = case.inputs()
        assert set(inputs) <= {"text", "source_text", "reference_date"}
        blob = json.dumps(inputs, default=str)
        assert case.split not in blob
        for claim in case.claims:
            assert claim.claim_id not in inputs


def _fixture_doc() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_family_spanning_two_splits_is_rejected() -> None:
    doc = _fixture_doc()
    doc["cases"][1]["split"] = "train"  # sr-latn-ana-move leaves its family's dev split
    with pytest.raises(FixtureError, match="belongs to one split"):
        parse_fixture(doc, sha256="x")


def test_support_span_must_be_verbatim() -> None:
    doc = _fixture_doc()
    doc["cases"][0]["claims"][0]["support"] = "Ana moved to Paris."
    with pytest.raises(FixtureError, match="verbatim"):
        parse_fixture(doc, sha256="x")


def test_anchor_must_appear_in_its_support_span() -> None:
    doc = _fixture_doc()
    doc["cases"][0]["claims"][0]["anchors"] = [["paris"]]
    with pytest.raises(FixtureError, match="absent from its support span"):
        parse_fixture(doc, sha256="x")


def test_edited_gold_breaks_the_frozen_pin(tmp_path: Path) -> None:
    doc = _fixture_doc()
    doc["cases"][0]["claims"].pop()
    edited = tmp_path / "extraction_v1.json"
    edited.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(FixtureError, match="frozen pin"):
        load_fixture(edited, expected_sha256=load_fixture().sha256)


def test_evaluator_fingerprint_is_stable() -> None:
    assert evaluator_fingerprint() == evaluator_fingerprint(load_fixture())
    assert len(evaluator_fingerprint()) == 64


def test_scorecard_reports_capabilities_per_split() -> None:
    fx = load_fixture()
    dev = [
        score_case(fx.case("en-ana-move"), ANA_GOLD),
        score_case(fx.case("en-greeting"), []),
        score_case(fx.case("sr-cyrl-greeting"), _facts("Pozdrav.")),
    ]
    card = scorecard(dev)
    assert card["split"] == "dev"
    assert card["evaluator"] == evaluator_fingerprint()
    assert card["capabilities"]["negation"] == 1.0
    assert card["capabilities"]["abstention"] == 0.5
    assert card["languages"]["sr-Cyrl"] == 0.0
    mixed = [*dev, score_case(fx.case("en-jelena-role"), [], purpose="report")]
    with pytest.raises(ValueError, match="one scorecard per split"):
        scorecard(mixed)


def test_write_path_never_imports_the_evaluator() -> None:
    root = Path(__file__).resolve().parents[2] / "gluomem"
    forbidden = ("extraction_eval", "llm.campaign", "extraction_v1")
    scanned = 0
    for package in ("write", "frames", "kg", "search", "storage", "embedding"):
        for path in (root / package).rglob("*.py"):
            scanned += 1
            text = path.read_text(encoding="utf-8")
            assert not any(name in text for name in forbidden), path
    assert scanned > 10
