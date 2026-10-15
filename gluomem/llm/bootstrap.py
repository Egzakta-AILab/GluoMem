"""gluomem.llm.bootstrap — compiled-program artifact store (save/load/hash-pin).

Optimization happens OFFLINE (``gluomem/llm/optimize.py``); runtime NEVER calls
``.compile()``. Artifacts are persisted with ``module.save(dir,
save_program=True)`` (dspy writes ``program.pkl`` + ``metadata.json`` with
dependency versions) plus our own ``manifest.gluomem.json`` carrying the task
name, the sha256 of ``program.pkl``, and the model ref it was compiled for.

Loading is read-only with a DOUBLE hash check: the recomputed sha256 of
``program.pkl`` must match the manifest AND, when ``dspy.artifacts.pins``
pins the task, the config pin too. Any mismatch is an ``ArtifactError`` —
never a silent fallback onto a tampered artifact.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import dspy

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig

_MANIFEST_NAME = "manifest.gluomem.json"
_PROGRAM_NAME = "program.pkl"


class ArtifactError(RuntimeError):
    """A compiled-program artifact is missing, tampered, or pin-mismatched."""


def artifact_dir(root: str | Path, task: str) -> Path:
    return Path(root).expanduser() / task


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_artifact(module: dspy.Module, root: str | Path, task: str, *, model_ref: str) -> Path:
    """Persist a compiled module + manifest; returns the artifact directory."""
    target = artifact_dir(root, task)
    target.mkdir(parents=True, exist_ok=True)
    module.save(str(target), save_program=True)
    program = target / _PROGRAM_NAME
    if not program.exists():
        raise ArtifactError(f"dspy save wrote no {_PROGRAM_NAME} in {target}")
    manifest = {
        "task": task,
        "sha256": _sha256(program),
        "dspy_version": dspy.__version__,
        "model_ref": model_ref,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (target / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def load_artifact(root: str | Path, task: str, *, pins: dict[str, str]) -> dspy.Module:
    """Load a pinned artifact read-only; raise ``ArtifactError`` on any mismatch."""
    target = artifact_dir(root, task)
    program = target / _PROGRAM_NAME
    manifest_path = target / _MANIFEST_NAME
    if not program.exists() or not manifest_path.exists():
        raise ArtifactError(f"no artifact for task '{task}' under {target}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = _sha256(program)
    expected = manifest.get("sha256")
    if actual != expected:
        raise ArtifactError(
            f"artifact hash mismatch for task '{task}': {_PROGRAM_NAME} is "
            f"{actual}, manifest says {expected} — artifact tampered or corrupt"
        )
    pinned = pins.get(task)
    if pinned is not None and pinned != actual:
        raise ArtifactError(
            f"artifact pin mismatch for task '{task}': config pins {pinned}, artifact is {actual}"
        )
    return dspy.load(str(target), allow_pickle=True)


def load_or_build(task: str, cfg: AppConfig) -> dspy.Module:
    """The runtime entry: pinned artifact when present, else zero-shot module.

    ``dspy.artifacts.require=True`` turns a missing artifact into an error
    instead of the zero-shot fallback. Tampered/pin-mismatched
    artifacts ALWAYS raise regardless of ``require``.
    """
    # Local import: bootstrap must not import modules at module level
    # (modules.py imports load_or_build from here).
    from gluomem.llm.modules import build_module

    target = artifact_dir(cfg.dspy.artifacts.dir, task)
    if (target / _PROGRAM_NAME).exists():
        return load_artifact(cfg.dspy.artifacts.dir, task, pins=cfg.dspy.artifacts.pins)
    if cfg.dspy.artifacts.require:
        raise ArtifactError(
            f"dspy.artifacts.require=true but no compiled artifact exists for "
            f"task '{task}' under {target}; run `make optimize-prompts` first"
        )
    return build_module(task, cfg)
