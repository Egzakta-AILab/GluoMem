"""Generated fixture pins, literal markers and malformed-input contracts."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from gluomem.bench.adversarial import (
    ADVERSARIAL_FIXTURES,
    AdversarialFixtureError,
    load_adversarial_fixture,
    load_registered_adversarial_fixture,
)
from gluomem.bench.fixture_helpers import contains_folded
from gluomem.bench.registry import REGISTRY
from gluomem.bench.replay import REPLAY_FIXTURES, ReplayFixtureError, load_registered_fixture


@pytest.mark.parametrize("entry_id", ["mab_fixture", "sr_en_kvark_v1", "minja_adversarial_v1"])
@pytest.mark.parametrize("unpinned", [False, True])
def test_generated_fixtures_require_matching_pins(
    entry_id: str, unpinned: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adversarial = entry_id == "minja_adversarial_v1"
    paths = ADVERSARIAL_FIXTURES if adversarial else REPLAY_FIXTURES
    loader = load_registered_adversarial_fixture if adversarial else load_registered_fixture
    error = AdversarialFixtureError if adversarial else ReplayFixtureError
    if unpinned:
        monkeypatch.setitem(REGISTRY, entry_id, replace(REGISTRY[entry_id], sha256=None))
    else:
        path = tmp_path / "tampered.json"
        path.write_bytes(paths[entry_id].read_bytes() + b"\n")
        monkeypatch.setitem(paths, entry_id, path)
    with pytest.raises(error, match="registry pin"):
        loader(entry_id)


@pytest.mark.parametrize(
    "needle,hay",
    [
        ("Avala 9", "Главни трезор је Авала 9."),
        ("Željko Ćirić", "Željko Ćirić je potpisao."),
        ("Djordje", "Потписао Ђорђе."),
    ],
)
def test_markers_preserve_multiword_order(needle: str, hay: str) -> None:
    assert contains_folded(needle, hay)
    assert not contains_folded("missing marker", hay)
    assert not contains_folded("", hay)


@pytest.mark.parametrize("payload", [{}, {"scenarios": [{}]}, {"scenarios": [None]}, []])
def test_malformed_adversarial_input_raises_domain_error(payload: object, tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AdversarialFixtureError):
        load_adversarial_fixture(path)
