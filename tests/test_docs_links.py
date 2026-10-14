"""Relative Markdown links in the repository's documents must resolve."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)
_FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)


def _markdown_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [ROOT / p for p in out if (ROOT / p).is_file()]


def _slug(heading: str) -> str:
    # GitHub's anchor rule: lowercase, drop punctuation except '-' and '_', spaces -> '-'.
    text = re.sub(r"[`*]", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    body = _FENCE.sub("", path.read_text(encoding="utf-8"))
    return {_slug(h) for h in _HEADING.findall(body)}


@pytest.mark.parametrize("doc", _markdown_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(doc: Path) -> None:
    body = _FENCE.sub("", doc.read_text(encoding="utf-8"))
    broken: list[str] = []
    for target in _LINK.findall(body):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#") and not target[1:]:
            continue
        path_part, _, anchor = target.partition("#")
        resolved = (doc.parent / path_part).resolve() if path_part else doc
        if not resolved.exists():
            broken.append(f"{target} (missing {resolved.relative_to(ROOT)})")
            continue
        if anchor and resolved.suffix == ".md" and anchor not in _anchors(resolved):
            broken.append(f"{target} (no heading #{anchor})")
    assert not broken, f"{doc.relative_to(ROOT)}: broken links: {broken}"
