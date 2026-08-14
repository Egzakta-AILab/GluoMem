"""gluomem.write — admission gate + benchmark-ingest pipeline.

All memory writes funnel through ``gate.admit()`` (scan + dedup + index on
every path). This package is dspy-free: extraction goes through the
``gluomem.llm.Extractors`` seam (``make_extractors``).
"""

from gluomem.write.gate import AdmitResult, InjectionScanner, ScanVerdict, admit

__all__ = ["AdmitResult", "InjectionScanner", "ScanVerdict", "admit"]
