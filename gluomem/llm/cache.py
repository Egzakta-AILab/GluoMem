"""gluomem.llm.cache — CachedExtractors: pin one extraction, replay it across arms.

Extraction via deepseek-flash over OpenRouter at temperature 0 is NOT
reproducible run-to-run (provider routing / MoE batching) — an identical-config
re-ingest can swing the LoCoMo headline ~12pp overall / ~29pp temporal. So any
ingest-time ablation that measures a per-arm delta is really measuring provider
noise unless every arm shares ONE extraction.

This wrapper freezes one extraction to a JSONL transcript keyed on the
extraction INPUTS (not the rendered prompt, not the write-side config), so an
ablation whose change is POST extraction — KG resolution, dedup, frame model —
replays the identical extraction across every arm and the delta is noise-free.

Modes:
- ``record``     — build/extend the cache: a hit returns the cached output, a
  miss calls the inner extractor and persists it (resumable, deduped).
- ``readthrough``— reuse a pre-built cache: same behaviour as record but the few
  live misses (e.g. a drop-mode KG change that shifts an entity-overlap-triggered
  contradiction pair) are counted in ``cache_misses`` as a cleanliness diagnostic.
- ``replay``     — strict: a miss raises ``ExtractionCacheMiss`` (NOT an
  ``ExtractionError``, so the ingest degrade-to-raw path does not swallow it —
  a miss aborts loudly, proving the cache is complete for this run).

dspy-free: wraps any ``Extractors`` and only touches the pydantic output types.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from gluomem.llm.outputs import (
    CognifyResult,
    ContradictionVerdict,
    ExtractedEvent,
    ExtractedFact,
)

if TYPE_CHECKING:
    from datetime import datetime

    from gluomem.llm.outputs import Extractors

# Bump when the cached OUTPUT schema changes so stale caches miss loudly rather
# than deserialize into a wrong-shaped object.
_SCHEMA_VER = "1"
_SEP = "\x00"
_MODES = ("off", "record", "replay", "readthrough")

T = TypeVar("T")


class ExtractionCacheMiss(RuntimeError):
    """A strict-``replay`` lookup missed.

    Deliberately NOT a subclass of ``ExtractionError``: the ingest path catches
    ``ExtractionError`` and degrades the session to raw turns, which would
    silently produce a DIFFERENT DB on a cache miss. This bubbles up and aborts.
    """


def _key(method: str, *parts: str) -> str:
    """Content hash over (schema version, method, canonical inputs)."""
    h = hashlib.sha256()
    h.update(_SCHEMA_VER.encode())
    h.update(_SEP.encode())
    h.update(method.encode())
    for part in parts:
        h.update(_SEP.encode())
        h.update(part.encode())
    return h.hexdigest()


def _encode(method: str, value: object) -> object:
    """Typed output → JSON-native (dict / list of dict)."""
    if method in ("extract_facts", "extract_profile", "extract_events"):
        return [v.model_dump(mode="json") for v in value]  # type: ignore[union-attr]
    return value.model_dump(mode="json")  # type: ignore[union-attr]


def _decode(method: str, output: object) -> object:
    """JSON-native → typed output (validator chosen by method)."""
    if method in ("extract_facts", "extract_profile"):
        return [ExtractedFact.model_validate(x) for x in output]  # type: ignore[union-attr]
    if method == "extract_events":
        return [ExtractedEvent.model_validate(x) for x in output]  # type: ignore[union-attr]
    if method == "cognify":
        return CognifyResult.model_validate(output)
    if method == "check_contradiction":
        return ContradictionVerdict.model_validate(output)
    raise ValueError(f"unknown cached method {method!r}")


class CachedExtractors:
    """Freeze one extraction to disk and replay it across ablation arms.

    Implements the ``Extractors`` protocol by delegating to ``inner`` and
    interposing a content-addressed cache. See the module docstring for modes.
    """

    def __init__(self, inner: Extractors, *, path: str, mode: str) -> None:
        if mode not in _MODES:
            raise ValueError(f"extraction_cache.mode {mode!r} invalid; one of {list(_MODES)}")
        if mode != "off" and not path:
            raise ValueError("extraction_cache.path is required when mode != 'off'")
        self._inner = inner
        self._mode = mode
        self._path = Path(path).expanduser() if path else None
        self._lock = threading.Lock()
        self._store: dict[str, tuple[str, object]] = {}
        self.cache_misses = 0
        # Proxy usage so the runner's telemetry reads the inner accumulator
        # (record: real tokens; replay/readthrough hits: no call, stays ~0).
        self.usage = getattr(inner, "usage", None)
        if mode != "off":
            self._load()

    # --- store I/O -----------------------------------------------------------

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._store[rec["key"]] = (rec["method"], rec["output"])

    def _persist(self, method: str, key: str, value: object) -> None:
        output = _encode(method, value)
        with self._lock:
            if key in self._store:  # another thread won the race; keep first
                return
            self._store[key] = (method, output)
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {"key": key, "method": method, "output": output},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    def _lookup(self, key: str) -> object | None:
        rec = self._store.get(key)  # GIL-atomic; lock-free hot path
        if rec is None:
            return None
        return _decode(rec[0], rec[1])

    def _serve(self, method: str, key: str, call: Callable[[], T]) -> T:
        hit = self._lookup(key)
        if hit is not None:
            return hit  # type: ignore[return-value]
        if self._mode == "replay":
            raise ExtractionCacheMiss(
                f"no cached {method} for key {key[:12]}… — the cache is "
                "incomplete for this run (an input changed vs the recorded "
                "extraction); use mode='readthrough' to serve misses live."
            )
        # record | readthrough: call inner, persist, count the live call.
        with self._lock:
            self.cache_misses += 1
        value = call()
        self._persist(method, key, value)
        return value

    def cache_sha(self) -> str:
        """Order-independent sha256 of the whole cache (provenance)."""
        h = hashlib.sha256()
        for key in sorted(self._store):
            method, output = self._store[key]
            h.update(key.encode())
            h.update(_SEP.encode())
            h.update(method.encode())
            h.update(_SEP.encode())
            h.update(json.dumps(output, sort_keys=True, ensure_ascii=False).encode())
            h.update(b"\n")
        return h.hexdigest()

    # --- Extractors protocol -------------------------------------------------

    def extract_facts(self, text: str) -> list[ExtractedFact]:
        return self._serve(
            "extract_facts",
            _key("extract_facts", text),
            lambda: self._inner.extract_facts(text),
        )

    def extract_events(self, source_text: str, reference_date: datetime) -> list[ExtractedEvent]:
        return self._serve(
            "extract_events",
            _key("extract_events", source_text, reference_date.isoformat()),
            lambda: self._inner.extract_events(source_text, reference_date),
        )

    def extract_profile(self, text: str, *, speaker: str) -> list[ExtractedFact]:
        return self._serve(
            "extract_profile",
            _key("extract_profile", text, speaker),
            lambda: self._inner.extract_profile(text, speaker=speaker),
        )

    def cognify(self, text: str) -> CognifyResult:
        return self._serve(
            "cognify",
            _key("cognify", text),
            lambda: self._inner.cognify(text),
        )

    def check_contradiction(
        self, new_statement: str, stored_statement: str
    ) -> ContradictionVerdict:
        return self._serve(
            "check_contradiction",
            _key("check_contradiction", new_statement, stored_statement),
            lambda: self._inner.check_contradiction(new_statement, stored_statement),
        )


__all__ = ["CachedExtractors", "ExtractionCacheMiss"]
