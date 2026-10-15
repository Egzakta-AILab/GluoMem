"""gluomem.bench.adapters.longmemeval — SHA-pinned LongMemEval adapter (dspy-free).

LongMemEval (arXiv:2410.10813, ICLR 2025): 500 questions, each owning its own
multi-session chat history ("haystack"). Two pinned variants share one schema
and one adapter class:

- ``longmemeval_oracle`` — evidence-only haystacks (~1.9 sessions/question);
  the retrieval-ceiling variant.
- ``longmemeval_s`` — ~50 sessions / ~115k tokens per question; history that
  does NOT fit a context window (the headline variant).

Category map (``Sample.category``); the paper's "7 abilities" = 6
``question_type`` literals + abstention as a cross-cutting flag. Abstention is
marked ONLY by an ``_abs`` suffix on ``question_id`` (never a question_type) and
is checked FIRST — it overrides the nominal type:

    1=single-session-user      2=single-session-assistant
    3=single-session-preference 4=multi-session
    5=temporal-reasoning       6=knowledge-update    7=abstention

Schema notes (verified against the published files): ``haystack_dates`` /
``haystack_session_ids`` / ``haystack_sessions`` are positionally PARALLEL
arrays (index i is one session); a session is a list of
``{"role", "content", "has_answer"}`` turns; ``answer`` is str OR int (counting
questions) and is always cast to str for gold; timestamps are session-level
strings ``YYYY/MM/DD (Ddd) HH:MM``. Judge rubrics per category live here too
(``JUDGE_RUBRICS``) — transcribed from the official ``evaluate_qa.py``
per-type prompts so the trio judge grades each type the way the benchmark
authors do (temporal off-by-one-day forgiveness, knowledge-update accepts
old+new, abstention correct iff the prediction declines).
"""

from __future__ import annotations

import json
import re
import shutil
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gluomem.bench.adapters.locomo import DatasetError, _sha256_file
from gluomem.bench.types import Sample
from gluomem.write.ingest import Conversation, Session, Turn

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig, BenchmarkDatasetEntry

CATEGORY_NAMES: dict[int, str] = {
    1: "single-session-user",
    2: "single-session-assistant",
    3: "single-session-preference",
    4: "multi-session",
    5: "temporal-reasoning",
    6: "knowledge-update",
    7: "abstention",
}
_TYPE_TO_CATEGORY: dict[str, int] = {name: cat for cat, name in CATEGORY_NAMES.items() if cat != 7}

# Per-category grading rubrics for the typed judge (JudgeQATyped). Semantics
# transcribed from the official evaluate_qa.py per-question-type prompts.
JUDGE_RUBRICS: dict[str, str] = {
    "single-session-user": (
        "Mark correct only if the prediction contains the gold answer's "
        "information. Different wording, extra detail, or a different date "
        "format conveying the same information is correct; a partial or "
        "missing answer is incorrect."
    ),
    "single-session-assistant": (
        "The gold information was stated by the ASSISTANT earlier in the "
        "conversation history. Mark correct only if the prediction conveys "
        "the gold answer's information; partial or missing answers are "
        "incorrect."
    ),
    "single-session-preference": (
        "The gold describes the user's previously stated preference or "
        "constraint. Mark correct only if the prediction's answer or "
        "recommendation correctly incorporates that preference; a generic "
        "answer that ignores the stated preference is incorrect."
    ),
    "multi-session": (
        "Answering requires combining information from multiple sessions. "
        "Mark correct only if the prediction contains the gold answer's "
        "information (for counts, the exact number). Extra detail is fine; "
        "partial or missing answers are incorrect."
    ),
    "temporal-reasoning": (
        "Mark correct only if the prediction conveys the gold answer's "
        "information. For durations or day counts, accept answers within one "
        "day of the gold (off-by-one on the number of days is correct). "
        "Other date or time facts must match the gold."
    ),
    "knowledge-update": (
        "The gold is the UPDATED value of a fact that changed over the "
        "conversation history. Mark correct if the prediction states the "
        "updated value, even if it also mentions the outdated one. A "
        "prediction giving only the outdated value is incorrect."
    ),
    "abstention": (
        "This question is unanswerable from the conversation history (false "
        "premise or absent information). Mark correct ONLY if the prediction "
        "says the information is not available, that it does not know, or "
        "otherwise declines to answer. Any substantive answer is incorrect."
    ),
}

# "2023/04/10 (Mon) 23:07" — weekday token dropped before parsing (locale-safe).
_DATE_RE = re.compile(r"\s*\(\w+\)\s*")


def _parse_session_date(raw: str, *, where: str) -> datetime:
    cleaned = _DATE_RE.sub(" ", str(raw)).strip()
    try:
        return datetime.strptime(cleaned, "%Y/%m/%d %H:%M")  # noqa: DTZ007
    except ValueError:
        raise DatasetError(f"{where}: unparseable session date {raw!r}") from None


class LongMemEvalAdapter:
    """One adapter class for both pinned LongMemEval variants."""

    def __init__(
        self,
        cfg: AppConfig,
        dataset_key: str,
        *,
        path: str | Path | None = None,
    ) -> None:
        self._cfg = cfg
        self.name = dataset_key
        entry = cfg.benchmark.datasets.get(dataset_key)
        if entry is None:
            raise DatasetError(
                f"benchmark.datasets has no entry {dataset_key!r} — add "
                f"url/sha256/file to the config"
            )
        self._entry: BenchmarkDatasetEntry = entry
        self._path = (
            Path(path) if path is not None else Path(cfg.benchmark.datasets_dir) / self._entry.file
        ).expanduser()
        self._sha: str | None = None
        self._raw: list[dict[str, Any]] | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def sha(self) -> str:
        if self._sha is None:
            if not self._path.is_file():
                raise DatasetError(
                    f"dataset file not found: {self._path} — run "
                    f"`gluomem bench fetch --dataset {self.name}`"
                )
            self._sha = _sha256_file(self._path)
        return self._sha

    def verify(self) -> None:
        expected = self._entry.sha256
        if self.sha != expected:
            raise DatasetError(
                f"{self._path}: sha256 mismatch — got {self.sha}, pinned "
                f"{expected}; refusing to score against unpinned data"
            )

    def fetch(self, *, force: bool = False) -> Path:
        """Download the pinned URL, verify, atomically install (streamed —
        the S file is ~1 GB)."""
        if self._path.is_file() and not force:
            self.verify()
            return self._path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with (
            urllib.request.urlopen(self._entry.url) as resp,
            tmp.open(  # noqa: S310
                "wb"
            ) as out,
        ):
            shutil.copyfileobj(resp, out, length=1 << 20)
        got = _sha256_file(tmp)
        if got != self._entry.sha256:
            tmp.unlink(missing_ok=True)
            raise DatasetError(
                f"fetched {self._entry.url}: sha256 mismatch — got {got}, "
                f"pinned {self._entry.sha256}; upstream changed, re-pin "
                f"deliberately or investigate"
            )
        tmp.replace(self._path)
        self._sha = got
        return self._path

    # --- parsing --------------------------------------------------------------

    def _raw_questions(self) -> list[dict[str, Any]]:
        if self._raw is None:
            self.verify()
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise DatasetError(f"{self._path}: expected a JSON list")
            self._raw = data
        return self._raw

    @staticmethod
    def _category_of(question_id: str, question_type: str) -> int:
        # _abs is checked FIRST: abstention overrides the nominal type.
        if question_id.endswith("_abs"):
            return 7
        category = _TYPE_TO_CATEGORY.get(question_type)
        if category is None:
            raise DatasetError(f"{question_id}: unknown question_type {question_type!r}")
        return category

    def load(self) -> list[Sample]:
        """All 500 questions in file order (every row is eligible)."""
        samples: list[Sample] = []
        for raw in self._raw_questions():
            qid = str(raw.get("question_id", "")).strip()
            question = str(raw.get("question", "")).strip()
            if not qid or not question:
                raise DatasetError(f"{self._path}: row missing question_id/question: {raw!r:.120}")
            category = self._category_of(qid, str(raw.get("question_type", "")))
            gold = str(raw.get("answer", "")).strip()  # answer is str OR int
            if not gold:
                raise DatasetError(f"{qid}: empty answer")
            samples.append(
                Sample(
                    id=qid,
                    conv_id=qid,  # each question owns its haystack
                    question=question,
                    gold=gold,
                    category=category,
                    category_name=CATEGORY_NAMES[category],
                    evidence=tuple(str(e) for e in raw.get("answer_session_ids", []) or []),
                )
            )
        return samples

    def eligible_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for s in self.load():
            counts[s.category] = counts.get(s.category, 0) + 1
        return counts

    def excluded_counts(self) -> dict[str, int]:
        """Every LongMemEval row is eligible; malformed rows raise instead."""
        return {}

    # --- conversation / dia bridges -------------------------------------------

    def conversations(self) -> list[Conversation]:
        """One Conversation per question: its haystack sessions in file order.

        The three haystack arrays are positionally parallel; a length mismatch
        is data corruption and raises. Turns with empty content are dropped;
        assistant turns ingest like any speaker (the single-session-assistant
        questions need them retrievable).
        """
        out: list[Conversation] = []
        for raw in self._raw_questions():
            qid = str(raw.get("question_id"))
            dates = raw.get("haystack_dates") or []
            session_ids = raw.get("haystack_session_ids") or []
            sessions_raw = raw.get("haystack_sessions") or []
            if not (len(dates) == len(session_ids) == len(sessions_raw)):
                raise DatasetError(
                    f"{qid}: parallel haystack arrays disagree "
                    f"(dates={len(dates)} ids={len(session_ids)} "
                    f"sessions={len(sessions_raw)})"
                )
            sessions: list[Session] = []
            for i, (date_raw, sid, turns_raw) in enumerate(
                zip(dates, session_ids, sessions_raw, strict=True)
            ):
                turns = [
                    Turn(
                        speaker=str(t.get("role", "user")),
                        text=text,
                        dia_id=f"{sid}:{ti}",
                    )
                    for ti, t in enumerate(turns_raw)
                    if (text := str(t.get("content", "")).strip())
                ]
                if not turns:
                    continue
                sessions.append(
                    Session(
                        index=i + 1,
                        date_time=_parse_session_date(date_raw, where=f"{qid} session[{i}]"),
                        turns=turns,
                    )
                )
            out.append(Conversation(conv_id=qid, speakers=("user", "assistant"), sessions=sessions))
        return out

    def dia_content(self) -> dict[str, str]:
        """dia_id -> raw-turn frame content (``"{speaker}: {text}"``)."""
        out: dict[str, str] = {}
        for conv in self.conversations():
            for session in conv.sessions:
                for turn in session.turns:
                    if turn.dia_id is not None:
                        out[turn.dia_id] = f"{turn.speaker}: {turn.text}"
        return out

    def content_to_dias(self) -> dict[str, tuple[str, ...]]:
        """Collision-aware reverse map (shared filler sessions repeat content)."""
        acc: dict[str, list[str]] = {}
        for dia_id, content in self.dia_content().items():
            acc.setdefault(content, []).append(dia_id)
        return {content: tuple(dias) for content, dias in acc.items()}


__all__ = [
    "CATEGORY_NAMES",
    "JUDGE_RUBRICS",
    "LongMemEvalAdapter",
]
