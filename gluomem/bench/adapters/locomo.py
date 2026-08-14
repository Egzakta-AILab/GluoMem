"""gluomem.bench.adapters.locomo — SHA-pinned LoCoMo dataset adapter (dspy-free).

LoCoMo (arXiv:2402.17753): 10 multi-session conversations, ~1,986 QA rows.
Empirical category map (derived from scored artifacts, not the paper's docs):

    1=multi-hop  2=temporal  3=open-ended  4=single-hop  5=adversarial

Category 5 gold answers are frequently missing/"undefined"; those rows are
ineligible and excluded from sampling (a refusal axis, not a QA axis).

The stratified sample mirrors the sibling hive-mind protocol (N = 80 x 4,
seed 42, xorshift32 + Fisher-Yates) so headline numbers are directly
comparable across the two harnesses. Each category bucket gets its own
generator seeded ``seed + category``, making a bucket's draw independent of
the configured category order.

``dia_content``/``content_to_dias`` reproduce EXACTLY the raw-turn frame
content ingest writes (``"{speaker}: {text}"`` with the blip caption
already folded by ``load_locomo``) — the bridge that maps retrieved frames
back to evidence dia_ids, since frames carry no dia_id column.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gluomem.bench.types import Sample
from gluomem.write.ingest import Conversation, load_locomo

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gluomem.bench.types import DatasetAdapter
    from gluomem.config.app_config import AppConfig, BenchmarkDatasetEntry

CATEGORY_NAMES: dict[int, str] = {
    1: "multi-hop",
    2: "temporal",
    3: "open-ended",
    4: "single-hop",
    5: "adversarial",
}


class DatasetError(RuntimeError):
    """SHA mismatch, malformed rows, missing files — always loud."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class LoCoMoAdapter:
    """``DatasetAdapter`` for the LoCoMo QA benchmark."""

    name = "locomo"

    def __init__(self, cfg: AppConfig, *, path: str | Path | None = None) -> None:
        self._cfg = cfg
        self._entry: BenchmarkDatasetEntry = cfg.benchmark.datasets["locomo"]
        self._path = (
            Path(path)
            if path is not None
            else Path(cfg.benchmark.datasets_dir) / self._entry.file
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
                    f"`gluomem bench fetch --dataset locomo`"
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
        """Download the pinned source URL, verify, atomically install."""
        if self._path.is_file() and not force:
            self.verify()
            return self._path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with urllib.request.urlopen(self._entry.url) as resp:  # noqa: S310
            tmp.write_bytes(resp.read())
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

    def _raw_samples(self) -> list[dict[str, Any]]:
        if self._raw is None:
            self.verify()
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise DatasetError(f"{self._path}: expected a JSON list")
            self._raw = data
        return self._raw

    def load(self) -> list[Sample]:
        """All ELIGIBLE QA rows in file order.

        Eligible = category parses to 1..5, category != 5 (adversarial gold
        is unreliable), and a non-empty non-"undefined" answer exists.
        """
        samples: list[Sample] = []
        for raw in self._raw_samples():
            conv_id = str(raw.get("sample_id"))
            for idx, row in enumerate(raw.get("qa", [])):
                category = self._parse_category(row, conv_id, idx)
                answer = row.get("answer")
                gold = str(answer).strip() if answer is not None else ""
                if category == 5 or not gold or gold.lower() == "undefined":
                    continue
                question = str(row.get("question", "")).strip()
                if not question:
                    raise DatasetError(
                        f"{conv_id} qa[{idx}]: eligible row without a question"
                    )
                samples.append(
                    Sample(
                        id=f"{conv_id}:qa:{idx}",
                        conv_id=conv_id,
                        question=question,
                        gold=gold,
                        category=category,
                        category_name=CATEGORY_NAMES[category],
                        evidence=tuple(
                            str(e) for e in row.get("evidence", []) or []
                        ),
                    )
                )
        return samples

    @staticmethod
    def _parse_category(row: dict[str, Any], conv_id: str, idx: int) -> int:
        raw = row.get("category")
        try:
            if not isinstance(raw, (int, str)):
                raise ValueError(f"bad type {type(raw).__name__}")
            category = int(raw)
        except ValueError:
            raise DatasetError(
                f"{conv_id} qa[{idx}]: unparseable category {raw!r}"
            ) from None
        if category not in CATEGORY_NAMES:
            raise DatasetError(
                f"{conv_id} qa[{idx}]: category {category} outside 1..5"
            )
        return category

    def eligible_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for s in self.load():
            counts[s.category] = counts.get(s.category, 0) + 1
        return counts

    # --- conversation / dia bridges --------------------------------------------

    def conversations(self) -> list[Conversation]:
        """Typed conversations via the SAME parser ingest uses (caption
        folding included) — single source of truth for turn content."""
        self.verify()
        return load_locomo(self._path)

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
        """Collision-aware reverse map (repeated statements share content)."""
        acc: dict[str, list[str]] = {}
        for dia_id, content in self.dia_content().items():
            acc.setdefault(content, []).append(dia_id)
        return {content: tuple(dias) for content, dias in acc.items()}


# --- stratified sampling ------------------------------------------------------


def _xorshift32(state: int) -> int:
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFFFFFF
    return state & 0xFFFFFFFF


def _shuffled(items: list[Sample], seed: int) -> list[Sample]:
    """Fisher-Yates driven by xorshift32; deterministic for a given seed."""
    out = list(items)
    state = seed & 0xFFFFFFFF or 1  # xorshift32 must never start at 0
    for i in range(len(out) - 1, 0, -1):
        state = _xorshift32(state)
        j = state % (i + 1)
        out[i], out[j] = out[j], out[i]
    return out


def stratified_sample(
    samples: Sequence[Sample],
    *,
    per_category: int,
    categories: Sequence[int],
    seed: int,
) -> list[Sample]:
    """``per_category`` rows per category; short buckets fail loud."""
    out: list[Sample] = []
    for category in categories:
        bucket = [s for s in samples if s.category == category]
        if len(bucket) < per_category:
            raise DatasetError(
                f"category {category} "
                f"({CATEGORY_NAMES.get(category, '?')}) has only "
                f"{len(bucket)} eligible rows; per_category={per_category} — "
                f"shrink benchmark.sample.per_category or drop the category"
            )
        out.extend(_shuffled(bucket, seed + category)[:per_category])
    return out


# --- sample manifest ------------------------------------------------------------


def write_sample_manifest(
    samples: Sequence[Sample], *, adapter: DatasetAdapter, cfg: AppConfig, path: Path
) -> None:
    """Commitable record of exactly which QA rows the headline runs score."""
    sample_cfg = cfg.benchmark.sample
    counts: dict[str, int] = {}
    for s in samples:
        counts[str(s.category)] = counts.get(str(s.category), 0) + 1
    manifest = {
        "dataset": adapter.name,
        "dataset_sha": adapter.sha,
        "seed": sample_cfg.seed,
        "per_category": sample_cfg.per_category,
        "categories": list(sample_cfg.categories),
        "counts": counts,
        "ids": [s.id for s in samples],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_sample_manifest(path: Path, *, adapter: DatasetAdapter) -> list[Sample]:
    """Resolve committed ids back to Samples; sha drift or unknown ids raise."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    recorded_sha = manifest.get("dataset_sha")
    if recorded_sha != adapter.sha:
        raise DatasetError(
            f"{path}: manifest pinned dataset_sha {recorded_sha}, but "
            f"{adapter.path} hashes to {adapter.sha} — dataset changed under "
            f"the committed sample"
        )
    by_id = {s.id: s for s in adapter.load()}
    samples: list[Sample] = []
    for sample_id in manifest["ids"]:
        sample = by_id.get(sample_id)
        if sample is None:
            raise DatasetError(f"{path}: manifest id {sample_id!r} not in dataset")
        samples.append(sample)
    return samples


__all__ = [
    "CATEGORY_NAMES",
    "DatasetError",
    "LoCoMoAdapter",
    "load_sample_manifest",
    "stratified_sample",
    "write_sample_manifest",
]
