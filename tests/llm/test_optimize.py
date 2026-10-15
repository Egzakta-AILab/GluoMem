"""DSPy artifact bootstrap save, load, tamper, pin, and require tests."""

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


def test_artifact_save_load_tamper_pin_and_require(tmp_path: Path) -> None:
    assert artifact_dir("artifacts/dspy", "cognify") == Path("artifacts/dspy/cognify")
    cfg = load_config().with_overrides({"dspy.artifacts.dir": str(tmp_path / "artifacts")})
    module = build_module("fact_extraction", cfg)
    target = save_artifact(
        module, cfg.dspy.artifacts.dir, "fact_extraction", model_ref="deepseek-flash"
    )
    assert (target / "program.pkl").exists()
    manifest = json.loads((target / "manifest.gluomem.json").read_text())
    assert manifest["task"] == "fact_extraction" and len(manifest["sha256"]) == 64

    loaded = load_artifact(target.parent, "fact_extraction", pins={})
    lm = DummyLM([{"facts": [{"text": "Alice lives in Boston.", "confidence": 1.0}]}])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        assert loaded(text="Alice lives in Boston.").facts[0].text == "Alice lives in Boston."

    with pytest.raises(ArtifactError, match="pin mismatch"):
        load_artifact(target.parent, "fact_extraction", pins={"fact_extraction": "0" * 64})

    assert isinstance(load_or_build("fact_extraction", cfg), dspy.Predict)
    empty_cfg = load_config().with_overrides({"dspy.artifacts.dir": str(tmp_path / "empty")})
    assert isinstance(load_or_build("cognify", empty_cfg), dspy.Predict)

    req_cfg = empty_cfg.with_overrides({"dspy.artifacts.require": True})
    with pytest.raises(ArtifactError, match="optimize-prompts"):
        load_or_build("fact_extraction", req_cfg)

    program = target / "program.pkl"
    program.write_bytes(program.read_bytes() + b"tamper")
    with pytest.raises(ArtifactError, match="hash mismatch"):
        load_artifact(target.parent, "fact_extraction", pins={})
