"""Component readiness, optional extras, fail-loud validation, and import boundaries."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from gluomem import extras
from gluomem.cli import _parse_override, main
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
}


@pytest.fixture
def no_model_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.util.find_spec
    hidden = {"dspy", "sentence_transformers"}

    def fake_find_spec(name: str, *args: object, **kwargs: object):  # noqa: ANN202
        if name in hidden:
            return None
        return real(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(extras.importlib.util, "find_spec", fake_find_spec)


@pytest.fixture
def extras_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extras, "_missing", lambda extra: [])


@pytest.fixture
def extras_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extras, "_missing", lambda extra: list(extras.EXTRAS[extra][0]))


def test_status_lists_every_extra() -> None:
    assert set(extras.status()) >= {"llm", "embeddings", "postgres", "service"}


def test_require_names_the_extra(no_model_packages: None) -> None:
    with pytest.raises(extras.MissingExtraError) as err:
        extras.require("llm", "thing")
    assert err.value.extra == "llm"
    assert "gluomem[llm]" in str(err.value)
    assert isinstance(err.value, ImportError)


def test_missing_extras_fail_instead_of_mocking(no_model_packages: None) -> None:
    with pytest.raises(extras.MissingExtraError, match="embeddings"):
        MiniLMEmbedder("sentence-transformers/all-MiniLM-L6-v2", dim=384).embed("hello")
    with pytest.raises(extras.MissingExtraError, match="embeddings"):
        CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2").score("q", ["d"])
    cfg = load_config().with_overrides({"write.extraction_backend": "dspy"})
    with pytest.raises(extras.MissingExtraError, match="llm"):
        make_extractors(cfg)


def test_fake_extraction_needs_no_extra(no_model_packages: None) -> None:
    cfg = load_config().with_overrides(
        {"write.extraction_backend": "fake", "components.allow_fake": True}
    )
    assert make_extractors(cfg) is not None


def test_cli_reports_missing_extra_without_traceback(
    no_model_packages: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "x.mind"
    db = MindDB(path, dim=384)
    db.close()
    rc = main(["index", "rebuild", "--db", str(path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "requires the 'embeddings' extra" in err and "Traceback" not in err


@pytest.mark.parametrize(
    "overrides",
    [
        {"models.minilm.provider": "onnx"},
        {
            "models.nodim": {"provider": "inprocess", "provider_model": "x"},
            "embedding.model": "nodim",
        },
    ],
    ids=["unsupported-provider", "no-dim"],
)
def test_unbuildable_embedder_raises_never_mock(
    overrides: dict[str, object], extras_installed: None
) -> None:
    cfg = load_config().with_overrides(overrides)
    with pytest.raises(ProviderUnavailableError):
        make_embedder(cfg)


def test_supported_embedder_and_reranker_build_unloaded(extras_installed: None) -> None:
    cfg = load_config()
    emb = make_embedder(cfg)
    assert isinstance(emb, MiniLMEmbedder) and emb._model is None
    rr = make_reranker(cfg)
    assert isinstance(rr, CrossEncoderReranker) and rr._model is None
    with pytest.raises(ProviderUnavailableError, match="onnx"):
        make_reranker(cfg.with_overrides({"models.reranker.provider": "onnx"}))
    assert make_reranker(cfg.with_overrides({"search.rerank.enabled": False})) is None


def test_factories_missing_extra_raise(extras_missing: None) -> None:
    cfg = load_config()
    with pytest.raises(extras.MissingExtraError):
        make_embedder(cfg)
    with pytest.raises(extras.MissingExtraError):
        make_reranker(cfg)


def test_fake_backends_require_allow_fake() -> None:
    cfg = load_config().with_overrides(_FAKES)
    for build in (make_embedder, make_extractors):
        with pytest.raises(FakeComponentError):
            build(cfg)
    allowed = cfg.with_overrides({"components.allow_fake": True})
    assert isinstance(make_embedder(allowed), MockEmbedder)
    assert make_extractors(allowed) is not None


def test_report_describes_real_components_without_secrets(extras_installed: None) -> None:
    secret = "sk-or-very-secret-value"
    report = component_report(load_config(), env={"OPENROUTER_API_KEY": secret})
    assert secret not in json.dumps(report)
    by_role = {c["role"]: c for c in report["components"]}
    emb = by_role["embedder"]
    assert emb["implementation"] == "sentence-transformers"
    assert emb["fingerprint"]["dim"] == 384 and not emb["fake"]
    assert by_role["reranker"]["implementation"] == "sentence-transformers.CrossEncoder"
    assert by_role["extraction:fact_extraction"]["api_key_set"] is True
    assert report["ready"] is True
    assert_ready(load_config(), env={"OPENROUTER_API_KEY": secret})


def test_report_not_ready_conditions(extras_installed: None, extras_missing: None) -> None:
    report = component_report(load_config(), env={})
    by_role = {c["role"]: c for c in report["components"]}
    assert report["ready"] is False
    assert by_role["embedder"]["reason"] == "'embeddings' extra not installed"
    assert by_role["extraction:fact_extraction"]["ready"] is False


def test_fakes_and_unsupported_providers_never_claim_ready(extras_installed: None) -> None:
    cfg = load_config().with_overrides({**_FAKES, "components.allow_fake": True})
    report = component_report(cfg, env={})
    assert report["ready"] is False
    with pytest.raises(ComponentNotReadyError, match="test double"):
        assert_ready(cfg, env={})

    bad_provider = load_config().with_overrides({"models.reranker.provider": "onnx"})
    rep2 = component_report(bad_provider, env={"OPENROUTER_API_KEY": "k"})
    assert rep2["ready"] is False
    assert any("has no" in c["reason"] for c in rep2["components"])


def test_report_checks_store_fingerprint(tmp_path: Path, extras_installed: None) -> None:
    db = MindDB(tmp_path / "s.mind", dim=384)
    FrameStore(db, embedder=MockEmbedder(dim=384)).create_i_frame("g", "alpha")
    store = component_report(load_config(), db=db, env={})["store"]
    assert store["has_vectors"] is True
    assert store["embedder_matches"] is False
    mock_cfg = load_config().with_overrides(
        {"embedding.model": "mock", "components.allow_fake": True}
    )
    assert component_report(mock_cfg, db=db, env={})["store"]["embedder_matches"] is True
    db.close()


def test_cli_components_and_index(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extras_installed: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k-cli")
    assert main(["components", "--require-ready"]) == 0
    out = capsys.readouterr().out
    assert "k-cli" not in out and json.loads(out)["ready"] is True
    assert (
        main(
            [
                "components",
                "--require-ready",
                "--override",
                "embedding.model=mock",
                "--override",
                "components.allow_fake=true",
            ]
        )
        == 3
    )

    path = tmp_path / "r.mind"
    db = MindDB(path, dim=384)
    FrameStore(db, embedder=MockEmbedder(dim=384, normalize=True)).create_i_frame("g", "alpha")
    db.close()

    rc = main(
        [
            "index",
            "rebuild",
            "--db",
            str(path),
            "--override",
            "embedding.model=mock",
        ]
    )
    assert rc == 2 and "components.allow_fake" in capsys.readouterr().err

    mock = ["--override", "embedding.model=mock", "--override", "components.allow_fake=true"]
    assert main(["index", "adopt", "--db", str(path), *mock]) == 2
    assert "different embedder" in capsys.readouterr().err
    assert main(["index", "rebuild", "--db", str(path), *mock]) == 0
    assert "rebuilt index for 1 frame(s)" in capsys.readouterr().out
    assert main(["index", "rebuild", "--db", str(tmp_path / "missing.mind"), *mock]) == 1


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"search.rrf_k": -5}, "rrf_k"),
        (
            {
                "scoring.profiles.balanced.weights": {
                    "temporal": 0.5,
                    "popularity": 0.5,
                    "contextual": 0.5,
                    "importance": 0.5,
                }
            },
            "sum",
        ),
        ({"storage.vector_dim": 512}, "512"),
        ({"write.conv_parallelism": 0}, "conv_parallelism"),
        ({"write.max_concurrent_llm": 0}, "max_concurrent_llm"),
        ({"write.dedup.mode": "fuzzy"}, "exact"),
        ({"write.dedup.similarity_threshold": 1.5}, "similarity_threshold"),
        ({"embedding.model": "does_not_exist"}, "does_not_exist"),
        ({"write.frame_model": "yolo"}, "frame_model"),
        ({"write.extraction_cache.mode": "bogus"}, "extraction_cache"),
        ({"write.extraction_cache.mode": "record"}, "path"),
        ({"kg.entity_type_mode": "purge"}, "entity_type_mode"),
        (
            {"kg.entity_type_mode": "drop", "kg.entity_type_allowlist": []},
            "entity_type_allowlist",
        ),
        ({"safety.fail_mode": "ignore"}, "fail_mode"),
        ({"write.extraction_backend": "gpt"}, "extraction_backend"),
        ({"write.lanes.profile.importance": "mega"}, "mega"),
        ({"write.subject.min_entity_overlap": 0}, "min_entity_overlap"),
        (
            {"safety.scanner.patterns": [{"regex": "([unclosed", "weight": 1.0}]},
            "compile",
        ),
        ({"write.subject.trigger": "vibes"}, "trigger"),
        ({"write.contradiction.top_k": 0}, "top_k"),
        ({"write.dedup.on_hit": "merge"}, "keep_longest"),
        ({"kg.predicate_families": [["LIVES_IN"]]}, "predicate_families"),
        (
            {
                "kg.predicate_families": [
                    ["LIVES_IN", "MOVED_TO"],
                    ["moved_to", "RESIDES_IN"],
                ]
            },
            "disjoint",
        ),
        (
            {
                "kg.entity_type_mode": "off",
                "kg.entity_type_allowlist": [],
                "kg.entity_type_prompt_enum": True,
            },
            "entity_type_prompt_enum",
        ),
    ],
)
def test_invalid_config_overrides_fail_loudly(overrides: dict[str, object], match: str) -> None:
    cfg = load_config()
    with pytest.raises(ValidationError, match=match):
        cfg.with_overrides(overrides)


def test_cli_override_parses_json_lists() -> None:
    assert _parse_override('safety.conflict_review.sources=["import"]') == (
        "safety.conflict_review.sources",
        ["import"],
    )
    with pytest.raises(Exception, match="JSON"):
        _parse_override("k=[1,2,")


def test_lazy_imports_do_not_load_heavy_packages() -> None:
    code = (
        "import sys; "
        "import gluomem.llm, gluomem.embedding.providers; "
        "loaded = [m for m in ('dspy', 'sentence_transformers') if m in sys.modules]; "
        "sys.exit(1 if loaded else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
