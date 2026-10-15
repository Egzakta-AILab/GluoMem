"""gluomem.write — admission gate + multi-session conversation ingestion.

All memory writes funnel through ``gate.admit()`` (scan + dedup + index on
every path). This package is dspy-free: extraction goes through the
``gluomem.llm.Extractors`` seam (``make_extractors``).
"""

from gluomem.write.conversation import (
    Conversation,
    IngestReport,
    Session,
    Turn,
    ingest_conversation,
)
from gluomem.write.gate import (
    AdmitResult,
    CompositeScanner,
    FakeInjectionClassifier,
    GlinerInjectionClassifier,
    InjectionClassifier,
    InjectionScanner,
    Scanner,
    ScanVerdict,
    admit,
    make_scanner,
)

__all__ = [
    "AdmitResult",
    "CompositeScanner",
    "Conversation",
    "FakeInjectionClassifier",
    "GlinerInjectionClassifier",
    "IngestReport",
    "InjectionClassifier",
    "InjectionScanner",
    "ScanVerdict",
    "Scanner",
    "Session",
    "Turn",
    "admit",
    "ingest_conversation",
    "make_scanner",
]
