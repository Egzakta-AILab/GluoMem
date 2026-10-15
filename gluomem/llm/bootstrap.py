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
_JSON_MANIFEST_NAME = "manifest.json"
_PROGRAM_NAME = "program.pkl"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ArtifactError(RuntimeError):
    """A compiled-program artifact is missing, tampered, or pin-mismatched."""


def _resolve_root(root: str | Path) -> Path:
    path = Path(root).expanduser()
    if not path.is_absolute() and not path.exists() and (_PROJECT_ROOT / path).exists():
        return _PROJECT_ROOT / path
    return path


def artifact_dir(root: str | Path, task: str) -> Path:
    return Path(root).expanduser() / task


def artifact_json_path(root: str | Path, task: str) -> Path:
    return Path(root).expanduser() / f"{task}.json"


def _sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix == ".json":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


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
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def save_json_artifact(module: dspy.Module, root: str | Path, task: str, *, model_ref: str) -> Path:
    """Persist a DSPy state JSON artifact (`<task>.json`) and update `<root>/manifest.json`."""
    root_path = Path(root).expanduser()
    root_path.mkdir(parents=True, exist_ok=True)
    json_path = artifact_json_path(root_path, task)
    module.save(str(json_path), save_program=False)
    normalized = (
        json.dumps(
            json.loads(json_path.read_text(encoding="utf-8")),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    json_path.write_text(normalized, encoding="utf-8", newline="\n")
    digest = _sha256(json_path)

    manifest_path = root_path / _JSON_MANIFEST_NAME
    manifest: dict[str, object] = {}
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            manifest = dict(loaded)
    artifacts_map = manifest.get("artifacts")
    entries: dict[str, object] = dict(artifacts_map) if isinstance(artifacts_map, dict) else {}
    entries[task] = {
        "task": task,
        "file": f"{task}.json",
        "sha256": digest,
        "dspy_version": dspy.__version__,
        "model_ref": model_ref,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    manifest["version"] = 1
    manifest["artifacts"] = entries
    manifest[task] = digest
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return json_path


def _expected_json_digest(manifest: object, task: str) -> str | None:
    if not isinstance(manifest, dict):
        return None
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, dict):
        entry = artifacts.get(task)
        if isinstance(entry, dict) and isinstance(entry.get("sha256"), str):
            return entry["sha256"]
        if isinstance(entry, str):
            return entry
    top = manifest.get(task)
    if isinstance(top, str):
        return top
    if isinstance(top, dict) and isinstance(top.get("sha256"), str):
        return top["sha256"]
    return None


def load_artifact(
    root: str | Path,
    task: str,
    *,
    pins: dict[str, str],
    cfg: AppConfig | None = None,
) -> dspy.Module:
    """Load a pinned artifact read-only; raise ``ArtifactError`` on any mismatch."""
    resolved_root = _resolve_root(root)
    target = artifact_dir(resolved_root, task)
    program = target / _PROGRAM_NAME
    manifest_path = target / _MANIFEST_NAME
    if program.exists() or manifest_path.exists():
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
                f"artifact pin mismatch for task '{task}': config pins {pinned}, "
                f"artifact is {actual}"
            )
        return dspy.load(str(target), allow_pickle=True)

    json_path = artifact_json_path(resolved_root, task)
    json_manifest_path = resolved_root / _JSON_MANIFEST_NAME
    if json_path.exists():
        if not json_manifest_path.exists():
            raise ArtifactError(f"no {_JSON_MANIFEST_NAME} for task '{task}' under {resolved_root}")
        manifest = json.loads(json_manifest_path.read_text(encoding="utf-8"))
        actual = _sha256(json_path)
        expected = _expected_json_digest(manifest, task)
        if actual != expected:
            raise ArtifactError(
                f"artifact hash mismatch for task '{task}': {json_path.name} is "
                f"{actual}, manifest says {expected} — artifact tampered or corrupt"
            )
        pinned = pins.get(task)
        if pinned is not None and pinned != actual:
            raise ArtifactError(
                f"artifact pin mismatch for task '{task}': config pins {pinned}, "
                f"artifact is {actual}"
            )
        from gluomem.config.app_config import load_config
        from gluomem.llm.modules import build_module

        module = build_module(task, cfg if cfg is not None else load_config())
        module.load(str(json_path))
        return module

    raise ArtifactError(f"no artifact for task '{task}' under {target}")


def load_or_build(task: str, cfg: AppConfig) -> dspy.Module:
    """The runtime entry: pinned artifact when present, else zero-shot module.

    ``dspy.artifacts.require=True`` turns a missing artifact into an error
    instead of the zero-shot fallback. Tampered/pin-mismatched
    artifacts ALWAYS raise regardless of ``require``.
    """
    # Local import: bootstrap must not import modules at module level
    # (modules.py imports load_or_build from here).
    from gluomem.llm.modules import build_module

    resolved_root = _resolve_root(cfg.dspy.artifacts.dir)
    target = artifact_dir(resolved_root, task)
    json_path = artifact_json_path(resolved_root, task)
    if (target / _PROGRAM_NAME).exists() or json_path.exists():
        return load_artifact(resolved_root, task, pins=cfg.dspy.artifacts.pins, cfg=cfg)
    if cfg.dspy.artifacts.require:
        raise ArtifactError(
            f"dspy.artifacts.require=true but no compiled artifact exists for "
            f"task '{task}' under {target}; run `make optimize-prompts` first"
        )
    return build_module(task, cfg)
