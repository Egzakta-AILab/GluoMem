"""Architecture invariant — every write funnels through gate.admit().

Mechanically enforces the "no write reaches the store unscanned" rule: no
write-path module other than gate.py may call ``store.create_*_frame`` (or
``set_importance``, which only the frame-model dispatch may use via gate-side
orchestration in ingest.py — allowed there because demotion is not a write of
new content, but creation calls stay gate-only).
"""

from __future__ import annotations

from pathlib import Path

import gluomem.write

_CREATE_TOKENS = ("create_i_frame(", "create_p_frame(", "create_b_frame(")


def test_no_direct_frame_creation_outside_gate() -> None:
    pkg_dir = Path(gluomem.write.__file__).parent
    offenders: list[str] = []
    for path in sorted(pkg_dir.glob("*.py")):
        if path.name == "gate.py":
            continue
        text = path.read_text(encoding="utf-8")
        for token in _CREATE_TOKENS:
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert offenders == [], (
        "store.create_* called outside gate.py — all writes must go through "
        f"admit(): {offenders}"
    )
