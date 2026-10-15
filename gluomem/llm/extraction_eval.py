"""gluomem.llm.extraction_eval — frozen, source-backed scorer for extraction.

Scores fact/event extractions against authored cases whose claims carry the
exact source span that supports them. It is deterministic and dspy-free: the
same inputs give the same scalar, components and feedback on any machine, and
nothing here calls a model.

Each predicted item is matched to the claim it best anchors to, then checked
for subject, polarity, modality (plan/possibility vs completed), update status
(current vs superseded) and date. An item that anchors to no claim is
unsupported, so a well-formed but invented fact scores zero, not a parse
credit. The scalar is F1 of faithful precision (items with no error) and
faithful recall (required claims covered by a faithful item); a case with no
claims rewards an empty extraction.

Splits are grouped by source family. ``sealed_test`` cases score without
feedback and are refused for optimization purposes, so test gold cannot reach
an optimizer, and only signature inputs (``Case.inputs``) ever reach a model.

Matching folds case and diacritics and transliterates Serbian Cyrillic to
Latin. Anchors and aliases are token patterns; ``*`` marks a stem prefix.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import cache
from pathlib import Path
from typing import Any, Literal

EVALUATOR_VERSION = 1
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "extraction_v1.json"
# Changing any gold claim, span, split or alias changes this pin on purpose:
# the evaluator is frozen, so a gold edit is a new evaluator version.
FIXTURE_SHA256 = "82edcf4d0dd17be84630dc699e9a7f81d8708d4da28af395d771207c15d8ad5c"

SPLITS = ("train", "dev", "sealed_test")
SEALED_SPLIT = "sealed_test"
TASKS = ("fact_extraction", "event_extraction")
Purpose = Literal["optimize", "report"]

ERROR_CAPABILITY: dict[str, str] = {
    "unsupported": "support",
    "wrong_subject": "subject",
    "missing_subject": "subject",
    "negation_lost": "negation",
    "negation_added": "negation",
    "hypothetical_as_completed": "modality",
    "completed_as_hypothetical": "modality",
    "stale_as_current": "update",
    "current_as_stale": "update",
    "missing_date": "temporal",
    "wrong_date": "temporal",
}
CAPABILITIES = ("support", "subject", "negation", "modality", "update", "temporal")

# Cue lexicons are part of the frozen evaluator (they feed the fingerprint).
LEXICON: dict[str, tuple[str, ...]] = {
    "negation": (
        "not",
        "no",
        "never",
        "none",
        "nobody",
        "nothing",
        "without",
        "nor",
        "neither",
        "ne",
        "nije",
        "nisu",
        "nisam",
        "nema",
        "nemaju",
        "nemam",
        "nikad",
        "nikada",
        "nista",
        "nece",
    ),
    "hypothetical": (
        "might",
        "could",
        "would",
        "maybe",
        "perhaps",
        "possibly",
        "plan",
        "plans",
        "planning",
        "intend*",
        "consider*",
        "hope*",
        "want*",
        "wish*",
        "will",
        "potential*",
        "likely",
        "expect*",
        "mozda",
        "planira*",
        "namerava*",
        "razmatra*",
        "razmislja*",
        "ce",
        "cu",
        "ces",
        "cemo",
        "cete",
        "bi",
        "hoce",
        "zeli*",
        "verovatno",
        "eventualno",
    ),
    # Marks a state as no longer current. Also satisfies superseded claims.
    "stale_strong": (
        "previously",
        "formerly",
        "former",
        "earlier",
        "originally",
        "initially",
        "before",
        "until",
        "used to",
        "no longer",
        "not anymore",
        "anymore",
        "ranije",
        "prethodno",
        "nekada",
        "nekad",
        "bivs*",
        "prvobitno",
        "pre toga",
        "vise ne",
        "dotad",
    ),
    # Past or origin markers; enough for a superseded claim, never an error
    # on a current one ("moved from Belgrade to Novi Sad").
    "stale_weak": (
        "from",
        "lived",
        "worked",
        "was",
        "were",
        "had",
        "iz",
        "ziveo",
        "zivela",
        "radio",
        "radila",
        "bio",
        "bila",
        "bili",
    ),
}
# Tokens of these phrases are consumed before negation is read, so "no
# longer" marks an update rather than a negation.
_NEGATION_EXEMPT = ("no longer", "not anymore", "vise ne")

_CYRILLIC = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "ђ": "dj",
    "е": "e",
    "ж": "z",
    "з": "z",
    "и": "i",
    "ј": "j",
    "к": "k",
    "л": "l",
    "љ": "lj",
    "м": "m",
    "н": "n",
    "њ": "nj",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "ћ": "c",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "c",
    "ч": "c",
    "џ": "dz",
    "ш": "s",
}
_SECRET = re.compile(
    r"(?i)\b(?:sk-[a-z0-9_-]{8,}|bearer\s+\S+|[a-z0-9_]*(?:api_?key|token|secret|password)"
    r"\s*[=:]\s*\S+|[a-f0-9]{32,}|[a-z0-9+/]{40,}={0,2})"
)
_MAX_QUOTE = 160


class FixtureError(ValueError):
    """The extraction fixture is malformed, edited or violates split policy."""


class SealedSplitError(PermissionError):
    """A sealed-test case was requested for an optimization purpose."""


# --------------------------------------------------------------------------
# Normalization and token-pattern matching
# --------------------------------------------------------------------------


def normalize(text: str) -> list[str]:
    """Lowercase, transliterate Cyrillic, fold diacritics, split into tokens."""
    lowered = text.lower().replace("n't", " not").replace("’", "'")
    latin = "".join(_CYRILLIC.get(ch, ch) for ch in lowered).replace("đ", "dj")
    folded = "".join(
        ch for ch in unicodedata.normalize("NFKD", latin) if not unicodedata.combining(ch)
    )
    return re.findall(r"[a-z0-9]+", folded)


def _pattern(alternative: str) -> tuple[str, ...]:
    parts = alternative.split()
    out: list[str] = []
    for part in parts:
        star = part.endswith("*")
        tokens = normalize(part.rstrip("*"))
        if not tokens:
            raise FixtureError(f"empty token pattern in {alternative!r}")
        out.extend(tokens[:-1])
        out.append(tokens[-1] + ("*" if star else ""))
    return tuple(out)


def _token_matches(pattern: str, token: str) -> bool:
    if pattern.endswith("*"):
        return token.startswith(pattern[:-1])
    return token == pattern


def _find(pattern: tuple[str, ...], tokens: Sequence[str]) -> int:
    """Start index of the first match of ``pattern`` in ``tokens`` or -1."""
    width = len(pattern)
    for start in range(len(tokens) - width + 1):
        if all(_token_matches(p, tokens[start + i]) for i, p in enumerate(pattern)):
            return start
    return -1


def _any(patterns: Iterable[tuple[str, ...]], tokens: Sequence[str]) -> bool:
    return any(_find(p, tokens) >= 0 for p in patterns)


@cache
def _lexicon(name: str) -> tuple[tuple[str, ...], ...]:
    return tuple(_pattern(alt) for alt in LEXICON[name])


def _strip_phrases(tokens: list[str], phrases: Iterable[str]) -> list[str]:
    out = list(tokens)
    for phrase in phrases:
        pattern = _pattern(phrase)
        while (start := _find(pattern, out)) >= 0:
            del out[start : start + len(pattern)]
    return out


@dataclass(frozen=True)
class Cues:
    negated: bool
    hypothetical: bool
    stale_strong: bool
    stale_weak: bool

    @classmethod
    def read(cls, tokens: list[str]) -> Cues:
        polarity_tokens = _strip_phrases(tokens, _NEGATION_EXEMPT)
        return cls(
            negated=_any(_lexicon("negation"), polarity_tokens),
            hypothetical=_any(_lexicon("hypothetical"), tokens),
            stale_strong=_any(_lexicon("stale_strong"), tokens),
            stale_weak=_any(_lexicon("stale_weak"), tokens),
        )


# --------------------------------------------------------------------------
# Fixture model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    claim_id: str
    anchors: tuple[tuple[tuple[str, ...], ...], ...]
    support: str
    subject: str | None = None
    negated: bool = False
    modality: Literal["asserted", "hypothetical"] = "asserted"
    status: Literal["current", "superseded"] = "current"
    date: str | None = None
    required: bool = True

    @property
    def capabilities(self) -> tuple[str, ...]:
        tags = ["support"]
        if self.subject:
            tags.append("subject")
        if self.negated:
            tags.append("negation")
        if self.modality == "hypothetical":
            tags.append("modality")
        if self.status == "superseded":
            tags.append("update")
        if self.date:
            tags.append("temporal")
        return tuple(tags)


@dataclass(frozen=True)
class Case:
    case_id: str
    task: str
    language: str
    split: str
    source_family: str
    source: str
    entities: Mapping[str, tuple[tuple[str, ...], ...]]
    claims: tuple[Claim, ...]
    reference_date: str | None = None

    def inputs(self) -> dict[str, Any]:
        """Only the task signature inputs; never claims, spans or the split."""
        if self.task == "event_extraction":
            ref = datetime.fromisoformat(self.reference_date or "")
            return {"source_text": self.source, "reference_date": ref}
        return {"text": self.source}


@dataclass(frozen=True)
class Fixture:
    name: str
    version: int
    sha256: str
    split_policy: Mapping[str, Any]
    cases: tuple[Case, ...]

    def case(self, case_id: str) -> Case:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(f"unknown extraction case {case_id!r}")

    def select(self, *, task: str | None = None, splits: Iterable[str] = SPLITS) -> list[Case]:
        wanted = tuple(splits)
        unknown = set(wanted) - set(SPLITS)
        if unknown:
            raise FixtureError(f"unknown splits {sorted(unknown)}; expected {SPLITS}")
        return [c for c in self.cases if c.split in wanted and (task is None or c.task == task)]

    def split_manifest(self) -> dict[str, Any]:
        """Versioned grouping and per-split digests, for receipts."""
        splits: dict[str, Any] = {}
        for split in SPLITS:
            cases = [c for c in self.cases if c.split == split]
            ids = sorted(c.case_id for c in cases)
            splits[split] = {
                "cases": ids,
                "source_families": sorted({c.source_family for c in cases}),
                "digest": _digest(ids),
            }
        return {
            "fixture": self.name,
            "version": self.version,
            "fixture_sha256": self.sha256,
            "policy": dict(self.split_policy),
            "splits": splits,
        }


def _digest(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _parse_claim(raw: Mapping[str, Any], case_id: str, source: str) -> Claim:
    claim_id = str(raw["claim_id"])
    where = f"{case_id}/{claim_id}"
    support = str(raw["support"])
    if support not in source:
        raise FixtureError(f"{where}: support span is not a verbatim source span")
    groups = raw.get("anchors") or []
    if not groups or any(not g for g in groups):
        raise FixtureError(f"{where}: needs at least one non-empty anchor group")
    anchors = tuple(tuple(_pattern(a) for a in group) for group in groups)
    support_tokens = normalize(support)
    for group in anchors:
        if not _any(group, support_tokens):
            raise FixtureError(f"{where}: anchor group {group} absent from its support span")
    modality = raw.get("modality", "asserted")
    status = raw.get("status", "current")
    if modality not in ("asserted", "hypothetical") or status not in ("current", "superseded"):
        raise FixtureError(f"{where}: bad modality {modality!r} or status {status!r}")
    claim_date = raw.get("date")
    if claim_date is not None:
        _parse_partial_date(str(claim_date), where)
    return Claim(
        claim_id=claim_id,
        anchors=anchors,
        support=support,
        subject=raw.get("subject"),
        negated=bool(raw.get("negated", False)),
        modality=modality,
        status=status,
        date=None if claim_date is None else str(claim_date),
        required=bool(raw.get("required", True)),
    )


def _parse_partial_date(value: str, where: str) -> tuple[int, ...]:
    parts = value.split("-")
    try:
        nums = tuple(int(p) for p in parts)
        date(nums[0], nums[1] if len(nums) > 1 else 1, nums[2] if len(nums) > 2 else 1)
    except (ValueError, IndexError) as exc:
        raise FixtureError(f"{where}: date {value!r} is not YYYY, YYYY-MM or YYYY-MM-DD") from exc
    return nums


def _parse_case(raw: Mapping[str, Any]) -> Case:
    case_id = str(raw["case_id"])
    task, split = raw["task"], raw["split"]
    if task not in TASKS:
        raise FixtureError(f"{case_id}: task {task!r} not in {TASKS}")
    if split not in SPLITS:
        raise FixtureError(f"{case_id}: split {split!r} not in {SPLITS}")
    source = str(raw["source"])
    entities = {
        str(name): tuple(_pattern(a) for a in aliases)
        for name, aliases in (raw.get("entities") or {}).items()
    }
    claims = tuple(_parse_claim(c, case_id, source) for c in raw.get("claims") or [])
    if len({c.claim_id for c in claims}) != len(claims):
        raise FixtureError(f"{case_id}: duplicate claim ids")
    for claim in claims:
        if claim.subject is not None and claim.subject not in entities:
            raise FixtureError(f"{case_id}/{claim.claim_id}: subject has no aliases")
    if task == "event_extraction":
        try:
            datetime.fromisoformat(str(raw.get("reference_date")))
        except ValueError as exc:
            raise FixtureError(f"{case_id}: event cases need an ISO reference_date") from exc
    return Case(
        case_id=case_id,
        task=task,
        language=str(raw["language"]),
        split=split,
        source_family=str(raw["source_family"]),
        source=source,
        entities=entities,
        claims=claims,
        reference_date=raw.get("reference_date"),
    )


def parse_fixture(data: Mapping[str, Any], *, sha256: str) -> Fixture:
    """Validate a fixture document; enforces source-family split disjointness."""
    cases = tuple(_parse_case(c) for c in data.get("cases") or [])
    if not cases:
        raise FixtureError("fixture has no cases")
    if len({c.case_id for c in cases}) != len(cases):
        raise FixtureError("duplicate case ids")
    family_split: dict[str, str] = {}
    for case in cases:
        seen = family_split.setdefault(case.source_family, case.split)
        if seen != case.split:
            raise FixtureError(
                f"source family {case.source_family!r} appears in splits "
                f"{seen!r} and {case.split!r}; a family belongs to one split"
            )
    policy = data.get("split_policy")
    if not isinstance(policy, Mapping) or policy.get("group_by") != "source_family":
        raise FixtureError("split_policy must declare group_by: source_family")
    return Fixture(
        name=str(data["fixture"]),
        version=int(data["version"]),
        sha256=sha256,
        split_policy=policy,
        cases=cases,
    )


@cache
def load_fixture(path: Path = FIXTURE_PATH, *, expected_sha256: str | None = None) -> Fixture:
    """Load and validate the frozen fixture; the bundled one must match its pin."""
    blob = Path(path).read_bytes()
    actual = hashlib.sha256(blob).hexdigest()
    pin = FIXTURE_SHA256 if Path(path) == FIXTURE_PATH else expected_sha256
    if pin is not None and actual != pin:
        raise FixtureError(
            f"{path}: sha256 {actual} does not match the frozen pin {pin}; gold edits "
            f"need a new evaluator version"
        )
    return parse_fixture(json.loads(blob), sha256=actual)


def evaluator_fingerprint(fixture: Fixture | None = None) -> str:
    """Identity of the frozen evaluator: version, fixture bytes and cue lexicons."""
    fixture = fixture or load_fixture()
    return _digest(
        {
            "evaluator_version": EVALUATOR_VERSION,
            "fixture_sha256": fixture.sha256,
            "lexicon": {k: list(v) for k, v in sorted(LEXICON.items())},
            "negation_exempt": list(_NEGATION_EXEMPT),
        }
    )


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    text: str
    event_date: datetime | None = None


def coerce_items(raw: Any) -> list[Item] | None:
    """Typed ExtractedFact/ExtractedEvent objects or dicts -> items; None if invalid.

    Only ``text`` and ``event_date`` are read. Any ``reasoning`` or other field
    on the prediction is ignored and never quoted in feedback.
    """
    if raw is None or not isinstance(raw, list | tuple):
        return None
    items: list[Item] = []
    for entry in raw:
        if hasattr(entry, "model_dump"):
            entry = entry.model_dump()
        if not isinstance(entry, Mapping):
            return None
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        confidence = entry.get("confidence", 1.0)
        if not isinstance(confidence, int | float) or not 0 <= confidence <= 1:
            return None
        when = entry.get("event_date")
        if isinstance(when, str):
            try:
                when = datetime.fromisoformat(when)
            except ValueError:
                return None
        if when is not None and not isinstance(when, datetime):
            return None
        items.append(Item(text=text, event_date=when))
    return items


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    claim_id: str
    errors: tuple[str, ...]
    detail: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ItemResult:
    text: str
    best: Match | None

    @property
    def faithful(self) -> bool:
        return self.best is not None and not self.best.errors

    @property
    def errors(self) -> tuple[str, ...]:
        return ("unsupported",) if self.best is None else self.best.errors


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    split: str
    language: str
    score: float
    components: Mapping[str, float | None]
    errors: Mapping[str, int]
    covered: tuple[str, ...]
    missed: tuple[str, ...]
    items: tuple[ItemResult, ...]
    capability_counts: Mapping[str, Mapping[str, int]]
    feedback: str
    no_fact: bool = False

    def record(self) -> dict[str, Any]:
        """JSON-safe receipt row (no item texts, so no user content)."""
        return {
            "case_id": self.case_id,
            "split": self.split,
            "language": self.language,
            "score": self.score,
            "components": dict(self.components),
            "errors": dict(self.errors),
            "covered": list(self.covered),
            "missed": list(self.missed),
        }


def _date_parts(value: datetime | str) -> tuple[int, ...]:
    if isinstance(value, datetime):
        return (value.year, value.month, value.day)
    return tuple(int(p) for p in value.split("-"))


def _check(case: Case, claim: Claim, item: Item, tokens: list[str], cues: Cues) -> Match | None:
    if not all(_any(group, tokens) for group in claim.anchors):
        return None
    errors: list[str] = []
    detail: dict[str, str] = {}
    if claim.subject is not None:
        if not _any(case.entities[claim.subject], tokens):
            others = [
                name
                for name, aliases in case.entities.items()
                if name != claim.subject and _any(aliases, tokens)
            ]
            errors.append("wrong_subject" if others else "missing_subject")
            if others:
                detail["named"] = others[0]
            detail["subject"] = claim.subject
    if claim.negated and not cues.negated:
        errors.append("negation_lost")
    elif cues.negated and not claim.negated:
        errors.append("negation_added")
    if claim.modality == "hypothetical" and not cues.hypothetical:
        errors.append("hypothetical_as_completed")
    elif cues.hypothetical and claim.modality == "asserted":
        errors.append("completed_as_hypothetical")
    if claim.status == "superseded" and not (cues.stale_strong or cues.stale_weak):
        errors.append("stale_as_current")
    elif claim.status == "current" and cues.stale_strong:
        errors.append("current_as_stale")
    if claim.date is not None and case.task == "event_extraction":
        if item.event_date is None:
            errors.append("missing_date")
        else:
            want = _date_parts(claim.date)
            got = _date_parts(item.event_date)[: len(want)]
            if got != want:
                errors.append("wrong_date")
                detail["got_date"] = "-".join(f"{n:02d}" for n in got)
            detail["date"] = claim.date
    return Match(claim.claim_id, tuple(errors), detail)


def _rank(match: Match, index: int) -> tuple[int, int, int, int]:
    # Fewest errors wins. Ties go to the claim about the named subject, then
    # to a matching date: two update claims often differ only in who or when.
    errors = set(match.errors)
    subject_off = 1 if {"wrong_subject", "missing_subject"} & errors else 0
    date_off = 1 if {"wrong_date", "missing_date"} & errors else 0
    return (len(match.errors), subject_off, date_off, index)


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else round(num / den, 4)


def _quote(text: str) -> str:
    redacted = _SECRET.sub("[redacted]", " ".join(text.split()))
    return redacted if len(redacted) <= _MAX_QUOTE else redacted[: _MAX_QUOTE - 1] + "…"


_ADVICE = {
    "unsupported": "is not stated anywhere in the source; extract only what the text says",
    "wrong_subject": "attributes to {named} what the source says about {subject}",
    "missing_subject": "does not name {subject}; each fact must stand alone without pronouns",
    "negation_lost": "drops the negation in the source; keep 'not'/'never' when it negates",
    "negation_added": "negates something the source asserts",
    "hypothetical_as_completed": (
        "records a plan or possibility as done; keep the hedge (might, plans to)"
    ),
    "completed_as_hypothetical": "hedges something the source states as done",
    "stale_as_current": ("presents a superseded state as current; say it was previous or changed"),
    "current_as_stale": "marks a current state as past",
    "missing_date": "has no event_date although the source gives {date}",
    "wrong_date": "has event_date {got_date} but the source gives {date}",
}


def _feedback(
    case: Case, items: Sequence[ItemResult], missed: Sequence[Claim], score: float
) -> str:
    by_id = {c.claim_id: c for c in case.claims}
    lines = [f"Score {score:.2f}."]
    if not case.claims:
        if items:
            lines.append(
                "The source holds no durable facts (greeting or small talk); return an empty list."
            )
        else:
            lines.append("Correctly extracted nothing from a source with no durable facts.")
        return "\n".join(lines)
    if not items:
        lines.append("Extracted nothing, but the source states facts.")
    for result in items:
        if result.faithful:
            continue
        if result.best is None:
            lines.append(f"'{_quote(result.text)}' {_ADVICE['unsupported']}.")
            continue
        claim = by_id[result.best.claim_id]
        for code in result.best.errors:
            slots = {"named": "?", "subject": "?", "date": "?", "got_date": "?"}
            advice = _ADVICE[code].format_map({**slots, **result.best.detail})
            lines.append(f"'{_quote(result.text)}' {advice} (source: '{_quote(claim.support)}').")
    for claim in missed:
        lines.append(f"Missed: no faithful fact covers '{_quote(claim.support)}'.")
    if len(lines) == 1:
        lines.append("Every extracted item is supported and faithful; nothing was missed.")
    return "\n".join(lines)


def score_case(case: Case, raw_items: Any, *, purpose: Purpose = "optimize") -> CaseScore:
    """Score one prediction for ``case``.

    ``purpose="optimize"`` refuses sealed-test cases. ``purpose="report"``
    scores them but withholds feedback text, so a sealed result can be
    recorded once without describing the gold.
    """
    if purpose not in ("optimize", "report"):
        raise ValueError(f"purpose must be 'optimize' or 'report', not {purpose!r}")
    sealed = case.split == SEALED_SPLIT
    if sealed and purpose == "optimize":
        raise SealedSplitError(
            f"case {case.case_id!r} is in {SEALED_SPLIT}; it cannot be used for optimization"
        )
    items = coerce_items(raw_items)
    required = [c for c in case.claims if c.required]
    capability_counts = {cap: {"tp": 0, "fp": 0, "fn": 0} for cap in CAPABILITIES}

    if items is None:
        components: dict[str, float | None] = {
            "schema_validity": 0.0,
            "support_precision": None,
            "support_recall": 0.0 if required else None,
            "abstention": None,
        }
        feedback = (
            ""
            if sealed
            else (
                "Score 0.00.\nOutput is not a list of items with non-empty text, "
                "confidence in [0, 1] and an ISO event_date or null."
            )
        )
        return CaseScore(
            case.case_id,
            case.split,
            case.language,
            0.0,
            components,
            {"invalid_schema": 1},
            (),
            tuple(c.claim_id for c in required),
            (),
            capability_counts,
            feedback,
            no_fact=not case.claims,
        )

    results: list[ItemResult] = []
    faithful_claims: set[str] = set()
    anchored_claims: set[str] = set()
    for item in items:
        tokens = normalize(item.text)
        cues = Cues.read(tokens)
        matches = [
            (m, i)
            for i, claim in enumerate(case.claims)
            if (m := _check(case, claim, item, tokens, cues)) is not None
        ]
        anchored_claims.update(m.claim_id for m, _ in matches)
        faithful_claims.update(m.claim_id for m, _ in matches if not m.errors)
        best = min(matches, key=lambda mi: _rank(*mi))[0] if matches else None
        results.append(ItemResult(item.text, best))

    by_id = {c.claim_id: c for c in case.claims}
    errors: dict[str, int] = {}
    for result in results:
        for code in result.errors:
            errors[code] = errors.get(code, 0) + 1
            capability_counts[ERROR_CAPABILITY[code]]["fp"] += 1
    for claim in case.claims:
        hit = claim.claim_id in faithful_claims
        if not hit and not claim.required:
            continue
        for cap in claim.capabilities:
            capability_counts[cap]["tp" if hit else "fn"] += 1
    missed = [c for c in required if c.claim_id not in faithful_claims]

    n_items = len(results)
    faithful_items = sum(r.faithful for r in results)
    if not case.claims:
        score = 1.0 if n_items == 0 else 0.0
    elif n_items == 0:
        score = 0.0
    else:
        recall = 1.0 if not required else (len(required) - len(missed)) / len(required)
        score = _f1(faithful_items / n_items, recall)
    score = round(score, 4)

    def fidelity(codes: tuple[str, ...], applies: Callable[[ItemResult], bool]) -> float | None:
        anchored = [r for r in results if r.best is not None]
        relevant = [r for r in anchored if applies(r) or set(codes) & set(r.errors)]
        return _rate(sum(not set(codes) & set(r.errors) for r in relevant), len(relevant))

    def claim_of(r: ItemResult) -> Claim:
        assert r.best is not None
        return by_id[r.best.claim_id]

    components = {
        "schema_validity": 1.0,
        "support_precision": _rate(sum(r.best is not None for r in results), n_items),
        "support_recall": _rate(
            sum(c.claim_id in anchored_claims for c in required), len(required)
        ),
        "faithful_precision": _rate(faithful_items, n_items),
        "faithful_recall": _rate(len(required) - len(missed), len(required)),
        "subject_fidelity": fidelity(
            ("wrong_subject", "missing_subject"), lambda r: claim_of(r).subject is not None
        ),
        "polarity_fidelity": fidelity(
            ("negation_lost", "negation_added"), lambda r: claim_of(r).negated
        ),
        "modality_fidelity": fidelity(
            ("hypothetical_as_completed", "completed_as_hypothetical"),
            lambda r: claim_of(r).modality == "hypothetical",
        ),
        "update_fidelity": fidelity(
            ("stale_as_current", "current_as_stale"),
            lambda r: claim_of(r).status == "superseded",
        ),
        "time_fidelity": fidelity(
            ("missing_date", "wrong_date"), lambda r: claim_of(r).date is not None
        ),
        "abstention": (
            (1.0 if n_items == 0 else 0.0) if not case.claims else (0.0 if n_items == 0 else None)
        ),
    }
    feedback = "" if sealed else _feedback(case, results, missed, score)
    return CaseScore(
        case_id=case.case_id,
        split=case.split,
        language=case.language,
        score=score,
        components=components,
        errors=dict(sorted(errors.items())),
        covered=tuple(sorted(faithful_claims)),
        missed=tuple(c.claim_id for c in missed),
        items=tuple(results),
        capability_counts=capability_counts,
        feedback=feedback,
        no_fact=not case.claims,
    )


def scorecard(scores: Sequence[CaseScore], *, fixture: Fixture | None = None) -> dict[str, Any]:
    """Aggregate case scores without averaging across splits.

    Capability scores are F1 over claims tagged with a capability (true
    positives and misses) and items carrying that capability's error (false
    positives). ``abstention`` is the share of no-fact cases left empty.
    """
    splits = {s.split for s in scores}
    if len(splits) > 1:
        raise ValueError(f"one scorecard per split; got {sorted(splits)}")
    totals = {cap: {"tp": 0, "fp": 0, "fn": 0} for cap in CAPABILITIES}
    for s in scores:
        for cap, counts in s.capability_counts.items():
            for key, n in counts.items():
                totals[cap][key] += n
    capabilities: dict[str, float | None] = {}
    for cap, t in totals.items():
        den = 2 * t["tp"] + t["fp"] + t["fn"]
        capabilities[cap] = None if den == 0 else round(2 * t["tp"] / den, 4)
    no_fact = [s for s in scores if s.no_fact]
    capabilities["abstention"] = (
        None if not no_fact else round(sum(s.score for s in no_fact) / len(no_fact), 4)
    )
    languages: dict[str, list[float]] = {}
    for s in scores:
        languages.setdefault(s.language, []).append(s.score)
    return {
        "evaluator": evaluator_fingerprint(fixture),
        "split": next(iter(splits), None),
        "n_cases": len(scores),
        "mean_score": round(sum(s.score for s in scores) / len(scores), 4) if scores else None,
        "capabilities": capabilities,
        "languages": {k: round(sum(v) / len(v), 4) for k, v in sorted(languages.items())},
        "cases": [s.record() for s in scores],
    }
