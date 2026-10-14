"""Benchmark types, dataset adapters (LoCoMo, LongMemEval), sampling, and registry."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from gluomem.bench.adapters import make_adapter
from gluomem.bench.adapters.locomo import (
    DatasetError,
    LoCoMoAdapter,
    _xorshift32,
    load_sample_manifest,
    stratified_sample,
    write_sample_manifest,
)
from gluomem.bench.adapters.longmemeval import (
    CATEGORY_NAMES,
    JUDGE_RUBRICS,
    LongMemEvalAdapter,
)
from gluomem.bench.registry import (
    REGISTRY,
    WITHIN_HISTORY_LABEL,
    RegistryError,
    SourceOverlapError,
    assert_disjoint,
    entry_digest,
    generalization_label,
    get_entry,
    manifest_record,
    overlap_report,
    protocol_id,
    registry_digest,
    require_runnable,
)
from gluomem.bench.replay import REPLAY_FIXTURES
from gluomem.bench.runner import run_benchmark
from gluomem.bench.types import (
    JudgeRecord,
    Prediction,
    ProtocolTag,
    RetrievedDoc,
    Sample,
    Score,
)
from gluomem.config.app_config import AppConfig, load_config
from tests.bench.conftest import FAKE_BACKENDS, FIXTURE, FIXTURE_SHA

LME_FIXTURE = Path(__file__).parent.parent / "fixtures" / "longmemeval_tiny.json"
LME_SHA = hashlib.sha256(LME_FIXTURE.read_bytes()).hexdigest()


def _tag(**kw: object) -> ProtocolTag:
    base: dict[str, object] = {
        "metric_family": "qa_accuracy",
        "judge_family": "trio_strict",
        "judge_count": 3,
        "backbone": "gpt5-mini+haiku+qwen3",
        "dataset_sha": "abc123",
    }
    base.update(kw)
    return ProtocolTag(**base)  # type: ignore[arg-type]


def _samples(category: int, n: int) -> list[Sample]:
    return [
        Sample(
            id=f"c1:qa:{category * 100 + i}",
            conv_id="c1",
            question="q",
            gold="g",
            category=category,
            category_name=str(category),
        )
        for i in range(n)
    ]


@pytest.fixture
def locomo_cfg() -> AppConfig:
    return load_config().with_overrides({"benchmark.datasets.locomo.sha256": FIXTURE_SHA})


@pytest.fixture
def locomo_adapter(locomo_cfg: AppConfig) -> LoCoMoAdapter:
    return LoCoMoAdapter(locomo_cfg, path=FIXTURE)


@pytest.fixture
def lme_cfg() -> AppConfig:
    return load_config().with_overrides(
        {
            "benchmark.datasets.longmemeval_s.file": LME_FIXTURE.name,
            "benchmark.datasets.longmemeval_s.sha256": LME_SHA,
            "benchmark.datasets_dir": str(LME_FIXTURE.parent),
        }
    )


@pytest.fixture
def lme_adapter(lme_cfg: AppConfig) -> LongMemEvalAdapter:
    return LongMemEvalAdapter(lme_cfg, "longmemeval_s")


def test_bench_types_frozen_equality_and_dspy_free_import() -> None:
    s = Sample(
        id="c1:qa:0",
        conv_id="c1",
        question="q",
        gold="g",
        category=4,
        category_name="single-hop",
        evidence=("D1:1",),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.gold = "tampered"  # type: ignore[misc]
    p = Prediction(sample_id=s.id, answer="a")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.answer = "tampered"  # type: ignore[misc]

    assert _tag() == _tag()
    assert hash(_tag()) == hash(_tag())
    assert _tag(backbone="haiku") != _tag()
    assert len({_tag(), _tag(), _tag(metric_family="precision@5")}) == 2

    sc = Score(value=1.0, tag=_tag(), n=1)
    assert sc.value == 1.0 and sc.n == 1
    rec = JudgeRecord(
        sample_id="c1:qa:0",
        backbone="haiku",
        model_slug="openrouter/anthropic/claude-haiku-4.5",
        correct=True,
        score=1.0,
        rationale="matches",
        question="q",
        gold="g",
        prediction="g indeed",
    )
    assert rec.correct is True
    d = RetrievedDoc(rank=0, text="Alice: hi", score=0.5)
    assert d.dia_id is None and d.frame_id is None

    code = "import gluomem.bench, sys; sys.exit(1 if 'dspy' in sys.modules else 0)"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_locomo_load_eligibility_categories_and_evidence(
    locomo_adapter: LoCoMoAdapter,
) -> None:
    samples = locomo_adapter.load()
    assert len(samples) == 5
    assert locomo_adapter.eligible_counts() == {1: 1, 2: 1, 3: 1, 4: 2}
    assert all(s.category != 5 for s in samples)

    by_id = {s.id: s for s in samples}
    assert by_id["conv-tiny-1:qa:2"].category == 3
    assert by_id["conv-tiny-1:qa:2"].category_name == "open-ended"
    assert by_id["conv-tiny-1:qa:0"].category_name == "multi-hop"
    assert by_id["conv-tiny-1:qa:1"].category_name == "temporal"
    assert samples[0].id == "conv-tiny-1:qa:0"
    assert samples[0].conv_id == "conv-tiny-1"
    assert samples[0].evidence == ("D1:2", "D3:1")


def test_locomo_sha_verify_missing_file_and_fetch(
    locomo_cfg: AppConfig, tmp_path: Path
) -> None:
    LoCoMoAdapter(locomo_cfg, path=FIXTURE).verify()
    bad = locomo_cfg.with_overrides({"benchmark.datasets.locomo.sha256": "0" * 64})
    with pytest.raises(DatasetError, match="sha256 mismatch"):
        LoCoMoAdapter(bad, path=FIXTURE).verify()

    with pytest.raises(DatasetError, match="bench fetch"):
        LoCoMoAdapter(locomo_cfg, path=tmp_path / "absent.json").load()

    src = tmp_path / "upstream.json"
    shutil.copyfile(FIXTURE, src)
    target = tmp_path / "data" / "locomo10.json"
    fetch_cfg = locomo_cfg.with_overrides({"benchmark.datasets.locomo.url": src.as_uri()})
    adapter = LoCoMoAdapter(fetch_cfg, path=target)
    assert adapter.fetch() == target and target.is_file()
    assert adapter.sha == FIXTURE_SHA

    src.write_text(src.read_text() + "\n", encoding="utf-8")
    adapter2 = LoCoMoAdapter(fetch_cfg, path=tmp_path / "data2" / "x.json")
    with pytest.raises(DatasetError, match="sha256 mismatch"):
        adapter2.fetch()
    assert not (tmp_path / "data2" / "x.json").exists()


def test_locomo_dia_bridges_match_ingested_raw_turns(
    locomo_cfg: AppConfig, locomo_adapter: LoCoMoAdapter, tmp_path: Path
) -> None:
    from gluomem.llm import NullExtractors
    from gluomem.storage.db import MindDB
    from gluomem.write.ingest import ingest_path

    ingest_cfg = locomo_cfg.with_overrides(
        {
            "write.extraction_backend": "none",
            "embedding.model": "mock",
            "components.allow_fake": True,
        }
    )
    db_path = tmp_path / "tiny.mind"
    ingest_path(FIXTURE, ingest_cfg, facade=NullExtractors(), db_path=db_path)
    with MindDB(db_path, dim=ingest_cfg.storage.vector_dim) as db:
        rows = db.conn.execute(
            "SELECT content FROM memory_frames WHERE gop_id LIKE '%:session:%'"
        ).fetchall()
    contents = {r[0] for r in rows}
    assert contents
    reverse = locomo_adapter.content_to_dias()
    assert not (contents - set(reverse))
    assert any("[shares photo:" in c for c in contents)
    assert reverse["Alice: I adopted a cat yesterday."] == ("D1:3", "D2:3")


def test_adapter_factory_and_longmemeval_parsing(
    lme_cfg: AppConfig, lme_adapter: LongMemEvalAdapter, tmp_path: Path
) -> None:
    cfg = load_config()
    with pytest.raises(ValueError, match="locomo"):
        make_adapter("membench", cfg)
    assert make_adapter("locomo", cfg).name == "locomo"
    assert make_adapter("longmemeval_oracle", cfg).name == "longmemeval_oracle"
    assert make_adapter("longmemeval_s", cfg).name == "longmemeval_s"

    samples = lme_adapter.load()
    assert [s.id for s in samples] == ["q_user_1", "q_multi_2", "q_ghost_3_abs"]
    by_id = {s.id: s for s in samples}
    assert by_id["q_user_1"].category == 1
    assert by_id["q_user_1"].category_name == "single-session-user"
    assert by_id["q_user_1"].evidence == ("sess_a",)
    assert by_id["q_multi_2"].gold == "2"
    assert by_id["q_multi_2"].category_name == "multi-session"
    assert by_id["q_ghost_3_abs"].category == 7
    assert by_id["q_ghost_3_abs"].category_name == "abstention"
    assert lme_adapter.eligible_counts() == {1: 1, 4: 1, 7: 1}

    convs = {c.conv_id: c for c in lme_adapter.conversations()}
    c1 = convs["q_user_1"]
    assert c1.speakers == ("user", "assistant")
    assert c1.sessions[0].date_time == datetime(2023, 5, 8, 10, 0)
    assert [t.speaker for t in c1.sessions[0].turns] == ["user", "assistant"]
    assert c1.sessions[0].turns[0].dia_id == "sess_a:0"
    assert [t.dia_id for t in convs["q_multi_2"].sessions[1].turns] == ["sess_d:0"]

    dia = lme_adapter.dia_content()
    assert dia["sess_a:0"].startswith("user: I graduated")
    assert lme_adapter.content_to_dias()[dia["sess_a:0"]] == ("sess_a:0",)
    assert set(JUDGE_RUBRICS) == set(CATEGORY_NAMES.values())
    assert "declines" in JUDGE_RUBRICS["abstention"]


def test_longmemeval_error_paths(lme_cfg: AppConfig, tmp_path: Path) -> None:
    rows = json.loads(LME_FIXTURE.read_text())
    rows[0]["haystack_dates"] = rows[0]["haystack_dates"][:1]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(rows))
    sha = hashlib.sha256(bad.read_bytes()).hexdigest()
    cfg2 = lme_cfg.with_overrides({"benchmark.datasets.longmemeval_s.sha256": sha})
    with pytest.raises(DatasetError, match="parallel"):
        LongMemEvalAdapter(cfg2, "longmemeval_s", path=bad).conversations()

    rows2 = json.loads(LME_FIXTURE.read_text())
    rows2[0]["question_type"] = "mystery"
    bad2 = tmp_path / "bad2.json"
    bad2.write_text(json.dumps(rows2))
    sha2 = hashlib.sha256(bad2.read_bytes()).hexdigest()
    cfg3 = lme_cfg.with_overrides({"benchmark.datasets.longmemeval_s.sha256": sha2})
    with pytest.raises(DatasetError, match="question_type"):
        LongMemEvalAdapter(cfg3, "longmemeval_s", path=bad2).load()

    bad_sha = lme_cfg.with_overrides({"benchmark.datasets.longmemeval_s.sha256": "0" * 64})
    with pytest.raises(DatasetError, match="sha256 mismatch"):
        LongMemEvalAdapter(bad_sha, "longmemeval_s").verify()


def test_stratified_sampling_and_manifest_roundtrip(
    locomo_cfg: AppConfig, locomo_adapter: LoCoMoAdapter, tmp_path: Path
) -> None:
    state = 42
    seq = [state := _xorshift32(state) for _ in range(3)]
    assert seq == [11355432, 2836018348, 476557059]

    pool = _samples(1, 50) + _samples(2, 50)
    a = stratified_sample(pool, per_category=10, categories=[1, 2], seed=42)
    b = stratified_sample(pool, per_category=10, categories=[1, 2], seed=42)
    c = stratified_sample(pool, per_category=10, categories=[1, 2], seed=43)
    assert [s.id for s in a] == [s.id for s in b]
    assert [s.id for s in a] != [s.id for s in c]

    ab = stratified_sample(pool, per_category=5, categories=[1, 2], seed=42)
    ba = stratified_sample(pool, per_category=5, categories=[2, 1], seed=42)
    assert [s.id for s in ab if s.category == 2] == [s.id for s in ba if s.category == 2]

    with pytest.raises(DatasetError, match="only 3 eligible"):
        stratified_sample(_samples(1, 3), per_category=10, categories=[1], seed=42)

    picked = stratified_sample(
        locomo_adapter.load(), per_category=1, categories=[1, 2, 3, 4], seed=42
    )
    manifest_path = tmp_path / "sample.json"
    write_sample_manifest(
        picked,
        adapter=locomo_adapter,
        cfg=locomo_cfg.with_overrides({"benchmark.sample.per_category": 1}),
        path=manifest_path,
    )
    loaded = load_sample_manifest(manifest_path, adapter=locomo_adapter)
    assert [s.id for s in loaded] == [s.id for s in picked]

    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["dataset_sha"] = "0" * 64
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(DatasetError, match="dataset changed"):
        load_sample_manifest(manifest_path, adapter=locomo_adapter)

    manifest_path.write_text(
        json.dumps({"dataset_sha": locomo_adapter.sha, "ids": ["conv-tiny-1:qa:999"]}),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="qa:999"):
        load_sample_manifest(manifest_path, adapter=locomo_adapter)


def test_benchmark_registry_contracts(bench_cfg: AppConfig, tmp_path: Path, capsys) -> None:
    datasets = load_config().benchmark.datasets
    for entry in REGISTRY.values():
        assert entry.source_ancestry and entry.scorers and entry.capabilities
        assert entry.model_roles and entry.rights
        assert entry.mode in ("native", "adapted")
        assert entry.readiness in ("ready", "fixture", "blocked")
        if entry.readiness != "blocked":
            assert entry.sha256 or entry.dataset_key
        if entry.dataset_key is not None:
            assert entry.dataset_key in datasets

    for bench_id, path in REPLAY_FIXTURES.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == get_entry(bench_id).sha256

    adapted = get_entry("locomo")
    native = replace(adapted, mode="native")
    assert protocol_id(adapted) == "adapted:locomo@v1"
    assert protocol_id(native) == "native:locomo@v1"
    assert entry_digest(adapted) != entry_digest(native)
    assert registry_digest() == registry_digest()

    pairs = {(o["a"], o["b"]): o["shared"] for o in overlap_report(REGISTRY)}
    assert pairs[("longmemeval_oracle", "memoryagentbench")] == ["longmemeval"]
    assert pairs[("longmemeval_s", "memoryagentbench")] == ["longmemeval"]
    assert pairs[("longmemeval_oracle", "longmemeval_s")] == ["longmemeval"]

    with pytest.raises(SourceOverlapError, match="longmemeval"):
        assert_disjoint(["longmemeval_s"], ["memoryagentbench"])
    assert_disjoint(["locomo", "mab_fixture"], ["longmemeval_s"])

    assert generalization_label(get_entry("locomo")) == WITHIN_HISTORY_LABEL
    with pytest.raises(RegistryError, match="blocked"):
        require_runnable(get_entry("memoryagentbench"))
    with pytest.raises(RegistryError, match="not registered"):
        get_entry("nope")

    cfg = bench_cfg.with_overrides(dict(FAKE_BACKENDS))
    with pytest.raises(RegistryError, match="blocked"):
        run_benchmark(cfg, dataset="memoryagentbench", retriever="bm25", out_root=tmp_path)

    paths = run_benchmark(cfg, dataset="locomo", retriever="bm25", out_root=tmp_path)
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["benchmark"] == json.loads(json.dumps(manifest_record(get_entry("locomo"))))
    assert manifest["protocol"] == "adapted:locomo@v1"

    from gluomem.cli import main

    capsys.readouterr()
    assert main(["bench", "registry"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["digest"] == registry_digest()
    assert {e["id"] for e in out["entries"]} == set(REGISTRY)
