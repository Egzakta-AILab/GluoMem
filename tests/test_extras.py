"""Optional-extra contract: missing model packages fail loudly, never fall back."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from gluomem import extras
from gluomem.cli import main
from gluomem.config.app_config import load_config
from gluomem.embedding.providers import MiniLMEmbedder
from gluomem.embedding.reranker import CrossEncoderReranker
from gluomem.llm import make_extractors

_FIXTURE = Path(__file__).parent / "fixtures" / "locomo_tiny.json"


@pytest.fixture
def no_model_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.util.find_spec
    hidden = {"dspy", "sentence_transformers"}

    def fake_find_spec(name: str, *args: object, **kwargs: object):  # noqa: ANN202
        if name in hidden:
            return None
        return real(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(extras.importlib.util, "find_spec", fake_find_spec)


def test_status_lists_every_extra() -> None:
    assert set(extras.status()) == {"llm", "embeddings", "mem0"}


def test_require_names_the_extra(no_model_packages: None) -> None:
    with pytest.raises(extras.MissingExtraError) as err:
        extras.require("llm", "thing")
    assert err.value.extra == "llm"
    assert "gluomem[llm]" in str(err.value)
    assert isinstance(err.value, ImportError)


def test_minilm_without_embeddings_extra_fails_instead_of_mocking(
    no_model_packages: None,
) -> None:
    emb = MiniLMEmbedder("sentence-transformers/all-MiniLM-L6-v2", dim=384)
    with pytest.raises(extras.MissingExtraError, match="embeddings"):
        emb.embed("hello")


def test_cross_encoder_without_embeddings_extra_fails(no_model_packages: None) -> None:
    with pytest.raises(extras.MissingExtraError, match="embeddings"):
        CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2").score("q", ["d"])


def test_dspy_extraction_without_llm_extra_fails(no_model_packages: None) -> None:
    cfg = load_config().with_overrides({"write.extraction_backend": "dspy"})
    with pytest.raises(extras.MissingExtraError, match="llm"):
        make_extractors(cfg)


def test_fake_extraction_needs_no_extra(no_model_packages: None) -> None:
    cfg = load_config().with_overrides({"write.extraction_backend": "fake"})
    assert make_extractors(cfg) is not None


def test_cli_reports_missing_extra_without_traceback(
    no_model_packages: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["ingest", str(_FIXTURE), "--db", str(tmp_path / "x.mind")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "requires the 'llm' extra" in err
    assert "Traceback" not in err
