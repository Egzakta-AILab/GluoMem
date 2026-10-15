"""gluomem.llm — DSPy LLM layer.

IMPORT DISCIPLINE: importing ``gluomem.llm`` NEVER imports dspy. The package
``__init__`` re-exports only the dspy-free seam (``gluomem/llm/outputs.py``,
``gluomem/llm/fake.py``); the dspy-backed implementation (``modules.py`` etc.)
is imported lazily inside ``make_extractors`` only when the configured
backend is ``"dspy"``. This keeps ``gluomem/write/`` and the offline test gate
entirely dspy-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gluomem.llm.cache import CachedExtractors, ExtractionCacheMiss
from gluomem.llm.fake import FakeExtractors, NullExtractors
from gluomem.llm.outputs import (
    CognifyResult,
    ConsolidationAction,
    ConsolidationDecision,
    ContradictionVerdict,
    ExtractedClaimSpec,
    ExtractedEvent,
    ExtractedFact,
    ExtractionError,
    Extractors,
    KGEntity,
    KGRelation,
    StoredCandidate,
)

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig


def make_extractors(cfg: AppConfig) -> Extractors:
    """Build the configured ``Extractors`` backend (mirrors ``make_embedder``).

    Dispatches on ``cfg.write.extraction_backend`` (validated at config load):

    - ``"dspy"`` — the real DSPy-backed extractors; the ONLY path that imports
      dspy (lazily, inside this function body).
    - ``"fake"`` — deterministic rule-based ``FakeExtractors`` (loud banner);
      requires ``components.allow_fake``.
    - ``"none"`` — ``NullExtractors``; no extraction (raw-turns-only baseline).

    When ``cfg.write.extraction_cache.mode != "off"`` the built backend is
    wrapped in ``CachedExtractors`` so one extraction can be frozen and replayed
    across ingest-time ablation arms (see ``gluomem.llm.cache``).
    """
    backend = cfg.write.extraction_backend
    inner: Extractors
    if backend == "fake":
        from gluomem.components import require_fake_allowed

        require_fake_allowed(cfg, "write.extraction_backend='fake'")
        inner = FakeExtractors()
    elif backend == "none":
        inner = NullExtractors()
    else:
        # backend == "dspy" (enum-validated): lazy import keeps dspy out of
        # every other code path.
        from gluomem.extras import require

        require("llm", "write.extraction_backend=dspy")
        from gluomem.llm.modules import DspyExtractors

        inner = DspyExtractors(cfg)

    cache = cfg.write.extraction_cache
    if cache.mode != "off":
        from gluomem.llm.cache import CachedExtractors

        return CachedExtractors(inner, path=cache.path, mode=cache.mode)
    return inner


__all__ = [
    "CachedExtractors",
    "CognifyResult",
    "ConsolidationAction",
    "ConsolidationDecision",
    "ContradictionVerdict",
    "ExtractedClaimSpec",
    "ExtractedEvent",
    "ExtractedFact",
    "ExtractionCacheMiss",
    "ExtractionError",
    "Extractors",
    "FakeExtractors",
    "KGEntity",
    "KGRelation",
    "NullExtractors",
    "StoredCandidate",
    "make_extractors",
]
