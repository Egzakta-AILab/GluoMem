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
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from gluomem.llm.outputs import (
    ACTION_TARGET_WRITER_CONTRACT,
    SUPPORTED_WRITER_CONTRACTS,
    ClaimConsolidationContext,
    CognifyResult,
    ConsolidationDecision,
    ConsolidationProjection,
    ContextualConsolidation,
    ContradictionVerdict,
    ExtractedClaimSpec,
    ExtractedEvent,
    ExtractedFact,
    ProjectedConsolidation,
    ProjectedConsolidationDecision,
    StoredCandidate,
    projected_decision_model,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from gluomem.llm.observed_identity import ObservedIdentityAlignment
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
    if method in ("extract_facts", "extract_profile", "extract_events", "extract_event_claims"):
        return [v.model_dump(mode="json") for v in value]  # type: ignore[union-attr]
    return value.model_dump(mode="json")  # type: ignore[union-attr]


def _decode(method: str, output: object) -> object:
    """JSON-native → typed output (validator chosen by method)."""
    if method in ("extract_facts", "extract_profile"):
        return [ExtractedFact.model_validate(x) for x in output]  # type: ignore[union-attr]
    if method == "extract_events":
        return [ExtractedEvent.model_validate(x) for x in output]  # type: ignore[union-attr]
    if method == "extract_event_claims":
        return [ExtractedClaimSpec.model_validate(x) for x in output]  # type: ignore[union-attr]
    if method == "cognify":
        return CognifyResult.model_validate(output)
    if method == "check_contradiction":
        return ContradictionVerdict.model_validate(output)
    if method in ("consolidate_claim", "consolidate_claim_with_context"):
        return ConsolidationDecision.model_validate(output)
    if method == "consolidate_claim_with_projection":
        return ProjectedConsolidationDecision.model_validate(output)
    raise ValueError(f"unknown cached method {method!r}")


class CachedExtractors:
    """Freeze one extraction to disk and replay it across ablation arms.

    Implements the ``Extractors`` protocol by delegating to ``inner`` and
    interposing a content-addressed cache. See the module docstring for modes.
    """

    def __new__(
        cls,
        inner: Extractors,
        *,
        path: str,
        mode: str,
        writer_contract: str | None = None,
        max_rationale_length: int = 512,
        semantic_producer_contract: str | None = None,
        identity_rationale_length: int = 512,
    ):
        if cls is CachedExtractors and writer_contract == "semantic-state-writer-v4":
            if getattr(inner, "writer_capability", None) != writer_contract or not isinstance(
                inner, ProjectedConsolidation
            ):
                raise ValueError("Semantic cache requires the inner writer capability")
            if semantic_producer_contract is not None:
                from gluomem.llm.observed_identity import (
                    PRODUCER_CONTRACTS,
                    ObservedIdentityAlignment,
                )

                if (
                    semantic_producer_contract not in PRODUCER_CONTRACTS
                    or not isinstance(inner, ObservedIdentityAlignment)
                    or inner.semantic_producer_capability != semantic_producer_contract
                ):
                    raise ValueError("Observed cache requires its inner identity capability")
                wrapper = (
                    _ObservedBothCached
                    if isinstance(inner, ContextualConsolidation)
                    else _ObservedCached
                )
                return object.__new__(wrapper)
            return object.__new__(_SemanticCached)
        if cls is CachedExtractors:
            projected = isinstance(inner, ProjectedConsolidation)
            contextual = isinstance(inner, ContextualConsolidation)
            cls = (
                _BothCached
                if projected and contextual
                else _ProjectedCached
                if projected
                else _ContextCached
                if contextual
                else cls
            )
        return object.__new__(cls)

    def __init__(
        self,
        inner: Extractors,
        *,
        path: str,
        mode: str,
        writer_contract: str | None = None,
        max_rationale_length: int = 512,
        semantic_producer_contract: str | None = None,
        identity_rationale_length: int = 512,
    ) -> None:
        if writer_contract is not None and writer_contract not in SUPPORTED_WRITER_CONTRACTS:
            raise ValueError("unknown writer contract")
        self._semantic_producer_contract = semantic_producer_contract
        self._identity_rationale_length = min(512, identity_rationale_length)
        self._writer_contract = writer_contract
        self._max_rationale_length = min(512, max_rationale_length)
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

    def _persist(
        self, method: str, key: str, value: object, *, generation: int | None = None
    ) -> None:
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
        if self._writer_contract == ACTION_TARGET_WRITER_CONTRACT:
            key = _key(method, self._writer_contract, str(self._max_rationale_length), key)
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

    def extract_event_claims(
        self, source_text: str, *, reference_date: datetime | None = None
    ) -> list[ExtractedClaimSpec]:
        ref_str = reference_date.isoformat() if reference_date is not None else ""
        return self._serve(
            "extract_event_claims",
            _key("extract_event_claims", source_text, ref_str),
            lambda: self._inner.extract_event_claims(source_text, reference_date=reference_date),
        )

    def consolidate_claim(
        self, new_claim: str, candidates: Sequence[StoredCandidate]
    ) -> ConsolidationDecision:
        cand_json = json.dumps(
            [c.model_dump(mode="json") for c in candidates],
            sort_keys=True,
            ensure_ascii=False,
        )
        return self._serve(
            "consolidate_claim",
            _key("consolidate_claim", new_claim, cand_json),
            lambda: self._inner.consolidate_claim(new_claim, candidates),
        )


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


class _ContextCached(CachedExtractors):
    def consolidate_claim_with_context(
        self,
        new_claim: str,
        candidates: Sequence[StoredCandidate],
        *,
        context: ClaimConsolidationContext,
    ) -> ConsolidationDecision:
        # __new__ exposes this facade only after the runtime capability check.
        inner = cast(ContextualConsolidation, self._inner)
        return self._serve(
            "consolidate_claim_with_context",
            _key(
                "consolidate_claim_with_context",
                new_claim,
                _canonical([c.model_dump(mode="json") for c in candidates]),
                context.model_dump_json(),
            ),
            lambda: inner.consolidate_claim_with_context(new_claim, candidates, context=context),
        )


class _ProjectedCached(CachedExtractors):
    @property
    def consolidation_projection_protocol(self):
        return cast(ProjectedConsolidation, self._inner).consolidation_projection_protocol

    def consolidate_claim_with_projection(
        self,
        new_claim: str,
        candidates: Sequence[StoredCandidate],
        *,
        context: ClaimConsolidationContext,
        projection: ConsolidationProjection,
    ) -> ProjectedConsolidationDecision:
        inner = cast(ProjectedConsolidation, self._inner)
        result = self._serve(
            "consolidate_claim_with_projection",
            _key(
                "consolidate_claim_with_projection",
                self.consolidation_projection_protocol,
                new_claim,
                _canonical([c.model_dump(mode="json") for c in candidates]),
                context.model_dump_json(),
                projection.model_dump_json(),
            ),
            lambda: inner.consolidate_claim_with_projection(
                new_claim,
                candidates,
                context=context,
                projection=projection,
            ),
        )
        if self._writer_contract == ACTION_TARGET_WRITER_CONTRACT:
            return projected_decision_model(
                projection,
                writer_contract=self._writer_contract,
                max_rationale_length=self._max_rationale_length,
            ).model_validate(result.model_dump())
        return projection.validate_decision(result)


class _BothCached(_ProjectedCached, _ContextCached):
    pass


class _WriterCapability(Protocol):
    @property
    def writer_capability(self) -> str: ...


_semantic_caches: weakref.WeakSet[_TransientSemanticCache] = weakref.WeakSet()
_semantic_caches_lock = threading.Lock()


def clear_semantic_caches():
    """Conservative process-local invalidation on any erasure/retraction."""
    with _semantic_caches_lock:
        for cache in list(_semantic_caches):
            with cache._lock:
                cache._store.clear()
                cache._semantic_generation += 1


class _TransientSemanticCache(CachedExtractors):
    """V4 is invocation-private: no disk transcript, no public content digest.

    A new wrapper/process cannot replay this cache. Strict replay misses rather
    than reading predecessor plaintext transcripts. Record/readthrough memoize
    only in this wrapper's memory; all inputs still bind each lookup.
    """

    persistent_cache_supported = False

    def __init__(self, *args, **kwargs):
        from secrets import token_bytes

        self._semantic_secret = token_bytes(32)
        self._semantic_generation = 0
        super().__init__(*args, **kwargs)
        with _semantic_caches_lock:
            _semantic_caches.add(self)

    @property
    def writer_capability(self):
        # The semantic factory already requires this exact declared capability.
        return cast(_WriterCapability, self._inner).writer_capability

    def _load(self):
        pass  # V4 never reads a persistent extraction transcript.

    def _persist(
        self, method: str, key: str, value: object, *, generation: int | None = None
    ) -> None:
        if generation is None:
            raise TypeError("Semantic cache persistence requires an observed generation")
        with self._lock:
            if generation == self._semantic_generation:
                self._store.setdefault(key, (method, _encode(method, value)))

    def _lookup(self, key):
        from gluomem.llm.semantic import SemanticClaimSpec

        record = self._store.get(key)
        if record is None:
            return None
        method, output = record
        if method == "extract_event_claims":
            # Transient records are only produced by _encode, never loaded from disk.
            return [SemanticClaimSpec.model_validate(x) for x in cast(list[object], output)]
        return _decode(method, output)

    def _serve(self, method: str, key: str, call: Callable[[], T]) -> T:
        import hmac

        from gluomem.write.source_binding import ACTIVE_SOURCE_ENVELOPE

        source = ACTIVE_SOURCE_ENVELOPE.get() if method == "extract_event_claims" else None
        with self._lock:
            generation = self._semantic_generation
        material = _canonical(
            [self.writer_capability, self._max_rationale_length, method, key, source]
        )
        private_key = hmac.new(self._semantic_secret, material.encode(), "sha256").hexdigest()
        if self._mode != "off":
            hit = self._lookup(private_key)
            if hit is not None:
                return cast(T, hit)
            if self._mode == "replay":
                raise ExtractionCacheMiss("No invocation-private semantic cache entry")
        self.cache_misses += 1
        result = call()
        if self._mode != "off":
            self._persist(method, private_key, result, generation=generation)
        return result

    def cache_sha(self):
        import hmac

        return hmac.new(
            self._semantic_secret, "".join(sorted(self._store)).encode(), "sha256"
        ).hexdigest()


class _SemanticCached(_TransientSemanticCache, _BothCached):
    def consolidate_claim_with_projection(self, new_claim, candidates, *, context, projection):
        result = super().consolidate_claim_with_projection(
            new_claim, candidates, context=context, projection=projection
        )
        return projected_decision_model(
            projection,
            writer_contract=self.writer_capability,
            max_rationale_length=self._max_rationale_length,
        ).model_validate(result.model_dump())


class _ObservedCached(_TransientSemanticCache, _ProjectedCached):
    """Capability-specific transient cache; no unsupported contextual method."""

    def consolidate_claim_with_projection(self, new_claim, candidates, *, context, projection):
        result = super().consolidate_claim_with_projection(
            new_claim, candidates, context=context, projection=projection
        )
        return projected_decision_model(
            projection,
            writer_contract=self.writer_capability,
            max_rationale_length=self._max_rationale_length,
        ).model_validate(result.model_dump())

    @property
    def semantic_producer_capability(self):
        return cast("ObservedIdentityAlignment", self._inner).semantic_producer_capability

    def _lookup(self, key):
        from gluomem.llm.observed_identity import (
            PROPERTY_IDENTITY_CONTRACT,
            IdentityProposal,
            PropertyIdentityProposal,
        )

        rec = self._store.get(key)
        if rec is not None and rec[0] == "extract_event_claims":
            from gluomem.llm.observed_identity import observed_claim_type

            if self._semantic_producer_contract == "observed-assertion-scope-v4":
                from gluomem.llm.assertion_scope import scoped_claim_type

                return [scoped_claim_type().model_validate(x) for x in cast(list[object], rec[1])]
            return [observed_claim_type().model_validate(x) for x in cast(list[object], rec[1])]
        if rec is not None and rec[0] == "align_claim_identity":
            proposal = (
                PropertyIdentityProposal
                if self._semantic_producer_contract
                in (PROPERTY_IDENTITY_CONTRACT, "observed-assertion-scope-v4")
                else IdentityProposal
            )
            return proposal.model_validate(rec[1])
        return super()._lookup(key)

    def _serve(self, method: str, key: str, call: Callable[[], T]) -> T:
        key = _key(
            method,
            cast(str, self._semantic_producer_contract),
            str(self._identity_rationale_length),
            key,
        )
        return super()._serve(method, key, call)

    def align_claim_identity(self, new_claim, candidates, *, context, projection):
        projection.validate_candidates(candidates, context)
        from gluomem.llm.observed_identity import identity_proposal_model

        inner = cast("ObservedIdentityAlignment", self._inner)
        result = self._serve(
            "align_claim_identity",
            _key(
                "align_claim_identity",
                new_claim,
                _canonical([c.model_dump(mode="json") for c in candidates]),
                context.model_dump_json(),
                projection.model_dump_json(),
            ),
            lambda: inner.align_claim_identity(
                new_claim, candidates, context=context, projection=projection
            ),
        )
        return identity_proposal_model(
            projection,
            context,
            max_rationale_length=self._identity_rationale_length,
            producer_contract=self._semantic_producer_contract,
        ).model_validate(result.model_dump())


class _ObservedBothCached(_ObservedCached, _ContextCached):
    pass


__all__ = ["CachedExtractors", "ExtractionCacheMiss"]
