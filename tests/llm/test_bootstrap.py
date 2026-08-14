"""Artifact machinery tests — save/load roundtrip, tamper + pin guards, require mode.

All offline: modules are zero-shot (never compiled) and calls run under
DummyLM. Artifacts load read-only with a hash check.
"""

from __future__ import annotations

import json
from pathlib import Path

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from gluomem.config.app_config import load_config
from gluomem.llm.bootstrap import (
    ArtifactError,
    artifact_dir,
    load_artifact,
    load_or_build,
    save_artifact,
)
from gluomem.llm.modules import build_module


def _cfg(artifacts_dir: Path, **over):
    overrides = {"dspy.artifacts.dir": str(artifacts_dir), **over}
    return load_config().with_overrides(overrides)


def _saved(tmp_path: Path) -> Path:
    cfg = _cfg(tmp_path / "artifacts")
    module = build_module("fact_extraction", cfg)
    return save_artifact(
        module, cfg.dspy.artifacts.dir, "fact_extraction", model_ref="deepseek-flash"
    )


def test_save_writes_program_and_manifest(tmp_path: Path) -> None:
    target = _saved(tmp_path)
    assert (target / "program.pkl").exists()
    manifest = json.loads((target / "manifest.gluomem.json").read_text())
    assert manifest["task"] == "fact_extraction"
    assert manifest["model_ref"] == "deepseek-flash"
    assert manifest["dspy_version"] == dspy.__version__
    assert len(manifest["sha256"]) == 64


def test_roundtrip_loaded_module_still_works_under_dummylm(tmp_path: Path) -> None:
    target = _saved(tmp_path)
    module = load_artifact(target.parent, "fact_extraction", pins={})
    lm = DummyLM([{"facts": [{"text": "Alice lives in Boston.", "confidence": 1.0}]}])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        pred = module(text="Alice lives in Boston. It rains.")
    assert pred.facts[0].text == "Alice lives in Boston."


def test_tampered_program_raises(tmp_path: Path) -> None:
    target = _saved(tmp_path)
    program = target / "program.pkl"
    program.write_bytes(program.read_bytes() + b"tamper")
    with pytest.raises(ArtifactError, match="hash mismatch"):
        load_artifact(target.parent, "fact_extraction", pins={})


def test_config_pin_mismatch_raises(tmp_path: Path) -> None:
    target = _saved(tmp_path)
    with pytest.raises(ArtifactError, match="pin mismatch"):
        load_artifact(
            target.parent, "fact_extraction", pins={"fact_extraction": "0" * 64}
        )


def test_load_or_build_falls_back_to_zero_shot(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path / "empty")
    module = load_or_build("cognify", cfg)
    assert isinstance(module, dspy.Predict)


def test_load_or_build_prefers_existing_artifact(tmp_path: Path) -> None:
    target = _saved(tmp_path)
    cfg = _cfg(target.parent)
    module = load_or_build("fact_extraction", cfg)
    # Loaded from disk (dspy.load), not freshly built — spot-check it is usable.
    lm = DummyLM([{"facts": []}])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        assert list(module(text="x").facts) == []


def test_require_mode_raises_on_missing_artifact(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path / "empty", **{"dspy.artifacts.require": True})
    with pytest.raises(ArtifactError, match="optimize-prompts"):
        load_or_build("fact_extraction", cfg)


def test_artifact_dir_layout() -> None:
    assert artifact_dir("artifacts/dspy", "cognify") == Path("artifacts/dspy/cognify")
