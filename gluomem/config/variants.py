"""gluomem.config.variants — variant overlay loader + merge utilities.

The precedence chain (lowest → highest) is::

    pydantic field defaults < conf/config.yaml < variant overlay
        < GLUOMEM_-prefixed env ('__' nested delimiter) < explicit overrides

This module provides the pure utilities that implement the middle of that
chain: deep-merge of the variant overlay over the file layer, and the
dotted-key → nested-dict conversion used by ``with_overrides``.

The actual ``AppConfig`` cross-validators and ``with_overrides`` method live on
:class:`gluomem.config.app_config.AppConfig`; this module holds the reusable
merge helpers so the precedence logic is testable in isolation.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dict; return ``{}`` if missing or empty."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` over ``base`` (returns a new dict).

    Nested dicts are merged key-by-key; non-dict values in ``overlay`` replace
    the corresponding value in ``base``.
    """
    out = deepcopy(base)
    for key, val in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = deepcopy(val)
    return out


def dotted_to_nested(overrides: dict[str, Any]) -> dict[str, Any]:
    """Convert dotted keys (``"search.rrf_k"``) into a nested dict."""
    nested: dict[str, Any] = {}
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node = nested
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return nested


def load_variant_overlay(
    name: str, variants_dir: Path | None = None
) -> dict[str, Any]:
    """Load ``conf/variants/{name}.yaml`` as a plain dict.

    Parameters
    ----------
    name:
        The variant name (without ``.yaml``).
    variants_dir:
        Directory holding variant YAMLs. Defaults to ``conf/variants``.
    """
    vdir = variants_dir if variants_dir is not None else _default_variants_dir()
    return read_yaml(vdir / f"{name}.yaml")


def merge_variant(
    file_layer: dict[str, Any],
    name: str,
    variants_dir: Path | None = None,
) -> dict[str, Any]:
    """Deep-merge the named variant overlay over ``file_layer``."""
    overlay = load_variant_overlay(name, variants_dir)
    if not overlay:
        return file_layer
    return deep_merge(file_layer, overlay)


def _default_variants_dir() -> Path:
    """Return the default ``conf/variants`` directory."""
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "conf" / "variants"
