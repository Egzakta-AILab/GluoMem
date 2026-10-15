"""gluomem.write — admission gate + benchmark-ingest pipeline.

All memory writes funnel through ``gate.admit()`` (scan + dedup + index on
every path). This package is dspy-free: extraction goes through the
``gluomem.llm.Extractors`` seam (``make_extractors``).
"""

from gluomem.write.gate import (
    AdmitResult,
    InjectionScanner,
    Scanner,
    ScanVerdict,
    admit,
    make_scanner,
)

__all__ = [
    "AdmitResult",
    "InjectionScanner",
    "ScanVerdict",
    "Scanner",
    "admit",
    "make_scanner",
]
