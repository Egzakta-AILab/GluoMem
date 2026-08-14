"""LongMemEval adapter — schema parsing, category map, conversations, rubrics."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pytest

from gluomem.bench.adapters import make_adapter
from gluomem.bench.adapters.locomo import (
    DatasetError,
    load_sample_manifest,
    write_sample_manifest,
)
from gluomem.bench.adapters.longmemeval import (
    CATEGORY_NAMES,
    JUDGE_RUBRICS,
    LongMemEvalAdapter,
)
from gluomem.config.app_config import load_config

FIXTURE = Path(__file__).parent.parent / "fixtures" / "longmemeval_tiny.json"


@pytest.fixture
def cfg():
    sha = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    return load_config().with_overrides(
        {
            "benchmark.datasets.longmemeval_s.file": FIXTURE.name,
            "benchmark.datasets.longmemeval_s.sha256": sha,
            "benchmark.datasets_dir": str(FIXTURE.parent),
        }
    )


@pytest.fixture
def adapter(cfg) -> LongMemEvalAdapter:
    return LongMemEvalAdapter(cfg, "longmemeval_s")


def test_registry_builds_both_variants() -> None:
    cfg = load_config()
    assert make_adapter("longmemeval_oracle", cfg).name == "longmemeval_oracle"
    assert make_adapter("longmemeval_s", cfg).name == "longmemeval_s"


def test_load_categories_gold_and_abs_override(adapter) -> None:
    samples = adapter.load()
    assert [s.id for s in samples] == ["q_user_1", "q_multi_2", "q_ghost_3_abs"]
    by_id = {s.id: s for s in samples}
    assert by_id["q_user_1"].category == 1
    assert by_id["q_user_1"].category_name == "single-session-user"
    assert by_id["q_user_1"].conv_id == "q_user_1"  # question owns its haystack
    assert by_id["q_user_1"].evidence == ("sess_a",)
    # int answer cast to str for gold
    assert by_id["q_multi_2"].gold == "2"
    assert by_id["q_multi_2"].category_name == "multi-session"
    # _abs overrides the nominal temporal-reasoning type
    assert by_id["q_ghost_3_abs"].category == 7
    assert by_id["q_ghost_3_abs"].category_name == "abstention"


def test_eligible_counts(adapter) -> None:
    assert adapter.eligible_counts() == {1: 1, 4: 1, 7: 1}


def test_conversations_dates_turns_and_dia_ids(adapter) -> None:
    convs = {c.conv_id: c for c in adapter.conversations()}
    assert set(convs) == {"q_user_1", "q_multi_2", "q_ghost_3_abs"}
    c1 = convs["q_user_1"]
    assert c1.speakers == ("user", "assistant")
    assert len(c1.sessions) == 2
    # weekday token stripped, minutes preserved
    assert c1.sessions[0].date_time == datetime(2023, 5, 8, 10, 0)
    # assistant turn ingested alongside the user turn
    assert [t.speaker for t in c1.sessions[0].turns] == ["user", "assistant"]
    assert c1.sessions[0].turns[0].dia_id == "sess_a:0"
    # empty-content assistant turn dropped (q_multi_2 second session)
    c2 = convs["q_multi_2"]
    assert [t.dia_id for t in c2.sessions[1].turns] == ["sess_d:0"]


def test_dia_bridges(adapter) -> None:
    dia = adapter.dia_content()
    assert dia["sess_a:0"].startswith("user: I graduated")
    rev = adapter.content_to_dias()
    assert rev[dia["sess_a:0"]] == ("sess_a:0",)


def test_parallel_array_mismatch_raises(cfg, tmp_path: Path) -> None:
    import json

    rows = json.loads(FIXTURE.read_text())
    rows[0]["haystack_dates"] = rows[0]["haystack_dates"][:1]  # break parallelism
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(rows))
    sha = hashlib.sha256(bad.read_bytes()).hexdigest()
    cfg2 = cfg.with_overrides({"benchmark.datasets.longmemeval_s.sha256": sha})
    a = LongMemEvalAdapter(cfg2, "longmemeval_s", path=bad)
    with pytest.raises(DatasetError, match="parallel"):
        a.conversations()


def test_unknown_question_type_raises(cfg, tmp_path: Path) -> None:
    import json

    rows = json.loads(FIXTURE.read_text())
    rows[0]["question_type"] = "mystery"
    bad = tmp_path / "bad2.json"
    bad.write_text(json.dumps(rows))
    sha = hashlib.sha256(bad.read_bytes()).hexdigest()
    cfg2 = cfg.with_overrides({"benchmark.datasets.longmemeval_s.sha256": sha})
    a = LongMemEvalAdapter(cfg2, "longmemeval_s", path=bad)
    with pytest.raises(DatasetError, match="question_type"):
        a.load()


def test_sha_verify_rejects_drift(cfg) -> None:
    bad_cfg = cfg.with_overrides(
        {"benchmark.datasets.longmemeval_s.sha256": "0" * 64}
    )
    with pytest.raises(DatasetError, match="sha256 mismatch"):
        LongMemEvalAdapter(bad_cfg, "longmemeval_s").verify()


def test_sample_manifest_roundtrip(adapter, cfg, tmp_path: Path) -> None:
    samples = adapter.load()
    path = tmp_path / "manifest.json"
    write_sample_manifest(samples, adapter=adapter, cfg=cfg, path=path)
    loaded = load_sample_manifest(path, adapter=adapter)
    assert [s.id for s in loaded] == [s.id for s in samples]


def test_rubrics_cover_every_category() -> None:
    assert set(JUDGE_RUBRICS) == set(CATEGORY_NAMES.values())
    assert "declines" in JUDGE_RUBRICS["abstention"]


def test_sample_categories_validator_allows_7_rejects_8() -> None:
    cfg = load_config()
    cfg.with_overrides({"benchmark.sample.categories": [1, 2, 3, 4, 5, 6, 7]})
    with pytest.raises(Exception, match="1..7"):
        cfg.with_overrides({"benchmark.sample.categories": [8]})
