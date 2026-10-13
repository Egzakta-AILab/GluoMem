"""Benchmark registry: protocol identity, source overlap, split labels, readiness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

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
from gluomem.config.app_config import AppConfig, load_config
from tests.bench.test_runner import FAKE_BACKENDS


def test_every_entry_records_the_required_fields() -> None:
    for entry in REGISTRY.values():
        assert entry.source_ancestry, entry.id
        assert entry.scorers, entry.id
        assert entry.capabilities, entry.id
        assert entry.model_roles, entry.id
        assert entry.rights, entry.id
        assert entry.mode in ("native", "adapted")
        assert entry.readiness in ("ready", "fixture", "blocked")
        if entry.readiness != "blocked":
            # Runnable entries have a pinned hash (registry or config).
            assert entry.sha256 or entry.dataset_key, entry.id


def test_config_pinned_entries_name_real_dataset_keys() -> None:
    datasets = load_config().benchmark.datasets
    for entry in REGISTRY.values():
        if entry.dataset_key is not None:
            assert entry.dataset_key in datasets, entry.id


def test_fixture_hash_is_pinned_in_the_registry() -> None:
    for bench_id, path in REPLAY_FIXTURES.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == get_entry(bench_id).sha256


def test_protocol_id_separates_native_and_adapted() -> None:
    adapted = get_entry("locomo")
    native = replace(adapted, mode="native")
    assert protocol_id(adapted) == "adapted:locomo@v1"
    assert protocol_id(native) == "native:locomo@v1"
    assert entry_digest(adapted) != entry_digest(native)


def test_digests_are_stable_and_content_addressed() -> None:
    assert registry_digest() == registry_digest()
    entry = get_entry("mab_fixture")
    assert entry_digest(entry) != entry_digest(replace(entry, version=2))


def test_overlap_detects_memoryagentbench_longmemeval_ancestry() -> None:
    pairs = {(o["a"], o["b"]): o["shared"] for o in overlap_report(REGISTRY)}
    assert pairs[("longmemeval_oracle", "memoryagentbench")] == ["longmemeval"]
    assert pairs[("longmemeval_s", "memoryagentbench")] == ["longmemeval"]
    assert pairs[("longmemeval_oracle", "longmemeval_s")] == ["longmemeval"]
    assert not any("locomo" in pair for pair in pairs)
    assert not any("mab_fixture" in pair for pair in pairs)


def test_held_out_set_must_be_source_disjoint() -> None:
    with pytest.raises(SourceOverlapError, match="longmemeval"):
        assert_disjoint(["longmemeval_s"], ["memoryagentbench"])
    assert_disjoint(["locomo", "mab_fixture"], ["longmemeval_s"])  # no raise


def test_question_splits_are_not_labeled_independent_history() -> None:
    assert generalization_label(get_entry("locomo")) == WITHIN_HISTORY_LABEL
    assert "NOT independent-history" in WITHIN_HISTORY_LABEL
    assert generalization_label(get_entry("mab_fixture")).startswith("grouped by history")


def test_blocked_entries_are_listed_but_refused() -> None:
    entry = get_entry("memoryagentbench")
    with pytest.raises(RegistryError, match="blocked"):
        require_runnable(entry)
    require_runnable(get_entry("locomo"))  # no raise
    with pytest.raises(RegistryError, match="not registered"):
        get_entry("nope")


def test_runner_refuses_a_blocked_benchmark(bench_cfg: AppConfig, tmp_path: Path) -> None:
    cfg = bench_cfg.with_overrides(dict(FAKE_BACKENDS))
    with pytest.raises(RegistryError, match="blocked"):
        run_benchmark(cfg, dataset="memoryagentbench", retriever="bm25", out_root=tmp_path)


def test_run_manifest_and_report_preserve_protocol_hashes(
    bench_cfg: AppConfig, tmp_path: Path
) -> None:
    cfg = bench_cfg.with_overrides(dict(FAKE_BACKENDS))
    paths = run_benchmark(cfg, dataset="locomo", retriever="bm25", out_root=tmp_path)
    manifest = json.loads(paths.manifest.read_text())
    record = manifest["benchmark"]
    assert record == json.loads(json.dumps(manifest_record(get_entry("locomo"))))
    assert manifest["protocol"] == "adapted:locomo@v1"
    assert record["generalization"] == WITHIN_HISTORY_LABEL
    entries = json.loads(paths.scores.read_text())
    assert {e["tag"]["protocol"] for e in entries} == {"adapted:locomo@v1"}
    assert {e["tag"]["dataset_sha"] for e in entries} == {manifest["dataset_sha"]}
    report = paths.report.read_text()
    assert "adapted:locomo@v1" in report
    assert record["digest"][:16] in report
    assert "judge models:" in report and "judge prompt sha:" in report
    assert WITHIN_HISTORY_LABEL in report


def test_registry_cli_prints_entries_and_overlaps(capsys) -> None:
    from gluomem.cli import main

    assert main(["bench", "registry"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["digest"] == registry_digest()
    assert {e["id"] for e in out["entries"]} == set(REGISTRY)
    assert {"a": "longmemeval_s", "b": "memoryagentbench", "shared": ["longmemeval"]} in out[
        "overlaps"
    ]
