"""Providers fail closed and readiness is inspectable without secrets.

No configuration reaches the mock embedder by fallback: an unknown model, an
unsupported provider, a missing dim, a dim mismatch or a missing extra raises.
Test doubles need ``components.allow_fake``. Nothing here downloads or calls a
model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gluomem import extras
from gluomem.cli import main
from gluomem.components import (
    ComponentNotReadyError,
    FakeComponentError,
    ProviderUnavailableError,
    assert_ready,
    component_report,
)
from gluomem.config.app_config import load_config
from gluomem.embedding.providers import MiniLMEmbedder, MockEmbedder, make_embedder
from gluomem.embedding.reranker import CrossEncoderReranker, make_reranker
from gluomem.frames.store import FrameStore
from gluomem.llm import make_extractors
from gluomem.storage.db import MindDB

_FAKES = {
    "embedding.model": "mock",
    "write.extraction_backend": "fake",
    "benchmark.answer.backend": "fake",
    "benchmark.judge.backend": "fake",
}


@pytest.fixture
def extras_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend every extra is installed (nothing is imported or loaded)."""
    monkeypatch.setattr(extras, "_missing", lambda extra: [])


@pytest.fixture
def extras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extras, "_missing", lambda extra: list(extras.EXTRAS[extra][0]))


# --- embedder factory --------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding.model": "ollama", "storage.vector_dim": 768},  # no implementation
        {"models.minilm.provider": "onnx"},  # declared provider is not built
        {
            "models.nodim": {"provider": "inprocess", "provider_model": "x"},
            "embedding.model": "nodim",
        },
    ],
    ids=["ollama", "unsupported-provider", "no-dim"],
)
def test_unbuildable_embedder_raises_never_mock(
    overrides: dict[str, object], extras_installed: None
) -> None:
    cfg = load_config().with_overrides(overrides)
    with pytest.raises(ProviderUnavailableError):
        make_embedder(cfg)


def test_unknown_embedding_model_rejected_at_config_load() -> None:
    with pytest.raises(ValueError, match="Unknown model reference 'nope'"):
        load_config().with_overrides({"embedding.model": "nope"})


def test_supported_embedder_builds_real_class_unloaded(extras_installed: None) -> None:
    emb = make_embedder(load_config())
    assert isinstance(emb, MiniLMEmbedder)
    assert emb._model is None, "construction must not download the model"


def test_missing_extra_raises_instead_of_mock(extras_missing: None) -> None:
    with pytest.raises(extras.MissingExtraError):
        make_embedder(load_config())


def test_mock_requires_allow_fake() -> None:
    cfg = load_config().with_overrides({"embedding.model": "mock"})
    with pytest.raises(FakeComponentError):
        make_embedder(cfg)
    allowed = cfg.with_overrides({"components.allow_fake": True})
    assert isinstance(make_embedder(allowed), MockEmbedder)


# --- reranker factory --------------------------------------------------------


def test_reranker_provider_is_validated(extras_installed: None) -> None:
    cfg = load_config()
    rr = make_reranker(cfg)
    assert isinstance(rr, CrossEncoderReranker) and rr._model is None
    with pytest.raises(ProviderUnavailableError, match="onnx"):
        make_reranker(cfg.with_overrides({"models.reranker.provider": "onnx"}))
    assert make_reranker(cfg.with_overrides({"search.rerank.enabled": False})) is None


def test_reranker_missing_extra_raises(extras_missing: None) -> None:
    with pytest.raises(extras.MissingExtraError):
        make_reranker(load_config())


# --- other fakes -------------------------------------------------------------


def test_fake_backends_require_allow_fake() -> None:
    from gluomem.bench.runner import _make_answer_fn, _make_judge_fns

    cfg = load_config().with_overrides(_FAKES)
    for build in (make_extractors, _make_answer_fn, _make_judge_fns):
        with pytest.raises(FakeComponentError):
            build(cfg)
    allowed = cfg.with_overrides({"components.allow_fake": True})
    for build in (make_extractors, _make_answer_fn, _make_judge_fns):
        assert build(allowed) is not None


# --- readiness report --------------------------------------------------------


def test_report_describes_real_components_without_secrets(extras_installed: None) -> None:
    secret = "sk-or-very-secret-value"
    report = component_report(load_config(), env={"OPENROUTER_API_KEY": secret})
    assert secret not in json.dumps(report)
    by_role = {c["role"]: c for c in report["components"]}
    emb = by_role["embedder"]
    assert emb["implementation"] == "sentence-transformers"
    assert emb["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert emb["fingerprint"]["dim"] == 384 and not emb["fake"]
    assert by_role["reranker"]["implementation"] == "sentence-transformers.CrossEncoder"
    assert by_role["extraction:fact_extraction"]["api_key_env"] == "OPENROUTER_API_KEY"
    assert by_role["extraction:fact_extraction"]["api_key_set"] is True
    assert report["ready"] is True
    assert_ready(load_config(), env={"OPENROUTER_API_KEY": secret})


def test_report_marks_missing_key_and_extras_not_ready(extras_missing: None) -> None:
    report = component_report(load_config(), env={})
    by_role = {c["role"]: c for c in report["components"]}
    assert report["ready"] is False
    assert by_role["embedder"]["reason"] == "'embeddings' extra not installed"
    assert by_role["answer"]["ready"] is False


def test_fakes_never_claim_ready(extras_installed: None) -> None:
    cfg = load_config().with_overrides({**_FAKES, "components.allow_fake": True})
    report = component_report(cfg, env={})
    assert report["ready"] is False
    fakes = {c["role"] for c in report["components"] if c["fake"]}
    assert {"embedder", "answer", "extraction:fact_extraction"} <= fakes
    with pytest.raises(ComponentNotReadyError, match="test double"):
        assert_ready(cfg, env={})


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding.model": "ollama", "storage.vector_dim": 768},
        {"models.reranker.provider": "onnx"},
    ],
)
def test_unsupported_provider_not_ready(
    overrides: dict[str, object], extras_installed: None
) -> None:
    report = component_report(
        load_config().with_overrides(overrides), env={"OPENROUTER_API_KEY": "k"}
    )
    assert report["ready"] is False
    assert any("has no" in c["reason"] for c in report["components"])


def test_report_checks_store_fingerprint(tmp_path: Path, extras_installed: None) -> None:
    db = MindDB(tmp_path / "s.mind", dim=384)
    FrameStore(db, embedder=MockEmbedder(dim=384)).create_i_frame("g", "alpha")
    store = component_report(load_config(), db=db, env={})["store"]
    assert store["has_vectors"] is True
    assert store["fingerprint"]["implementation"] == "gluomem.mock"
    assert store["embedder_matches"] is False  # config says MiniLM
    mock_cfg = load_config().with_overrides(
        {"embedding.model": "mock", "components.allow_fake": True}
    )
    assert component_report(mock_cfg, db=db, env={})["store"]["embedder_matches"] is True
    db.close()


# --- CLI -----------------------------------------------------------------------


def test_cli_components_require_ready_exit_codes(
    capsys: pytest.CaptureFixture[str], extras_installed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k-cli")
    assert main(["components", "--require-ready"]) == 0
    out = capsys.readouterr().out
    assert "k-cli" not in out and json.loads(out)["ready"] is True
    rc = main(
        [
            "components",
            "--require-ready",
            "--override",
            "embedding.model=mock",
            "--override",
            "components.allow_fake=true",
        ]
    )
    assert rc == 3


def test_cli_rejects_fake_without_opt_in(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "locomo_tiny.json"
    rc = main(
        [
            "ingest",
            str(fixture),
            "--db",
            str(tmp_path / "x.mind"),
            "--override",
            "write.extraction_backend=none",
            "--override",
            "embedding.model=mock",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "components.allow_fake" in err and "Traceback" not in err


def test_cli_index_rebuild_and_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "r.mind"
    db = MindDB(path, dim=384)
    FrameStore(db, embedder=MockEmbedder(dim=384, normalize=True)).create_i_frame("g", "alpha")
    db.close()
    mock = ["--override", "embedding.model=mock", "--override", "components.allow_fake=true"]
    assert main(["index", "adopt", "--db", str(path), *mock]) == 2
    assert "different embedder" in capsys.readouterr().err
    assert main(["index", "rebuild", "--db", str(path), *mock]) == 0
    assert "rebuilt index for 1 frame(s)" in capsys.readouterr().out
    db = MindDB(path, dim=384)
    FrameStore(db, embedder=MockEmbedder(dim=384))
    db.close()
    assert main(["index", "rebuild", "--db", str(tmp_path / "missing.mind"), *mock]) == 1
