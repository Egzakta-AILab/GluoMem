"""Variant overlay loader and configuration merge utilities."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
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


@lru_cache(maxsize=1)
def _cached_default_config() -> dict[str, Any]:
    text = (files("gluomem.config") / "default_config.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return {}
    return data


def read_default_config() -> dict[str, Any]:
    """Return a deep copy of packaged ``gluomem/config/default_config.yaml``."""
    return deepcopy(_cached_default_config())


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` over ``base`` (returns a new dict)."""
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


def _default_variants_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "conf" / "variants"


def load_variant_overlay(
    name: str, variants_dir: Path | None = None
) -> dict[str, Any]:
    """Load ``conf/variants/{name}.yaml`` as a plain dict."""
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
