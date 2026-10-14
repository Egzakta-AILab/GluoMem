"""GluoMem CLI entrypoint (``gluomem``).

``gluomem ingest``, ``gluomem bench``, ``gluomem components`` (effective
component readiness) and ``gluomem index`` (explicit embedding rebuild/adopt)
are live; ``tune``/``ablate`` remain stubs. The module top stays dspy-free —
LLM backends are resolved lazily inside handlers (``make_extractors`` for
ingest, the runner's backend dispatch for bench), so fake/none backends run fully offline, and the
``bench rescore``/``gate``/``sample``/``fetch`` handlers never touch dspy at
all.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig


def _parse_override(raw: str) -> tuple[str, object]:
    """``K=V`` → (dotted key, value); bools/ints/floats coerced, else string."""
    key, sep, value = raw.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(
            f"override {raw!r} must be dotted.key=value"
        )
    lowered = value.lower()
    if lowered in ("true", "false"):
        return key, lowered == "true"
    if value[:1] in ("[", "{"):
        import json

        try:
            return key, json.loads(value)
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(
                f"override {raw!r}: value looks like JSON but fails to "
                f"parse ({exc})"
            ) from None
    try:
        return key, int(value)
    except ValueError:
        pass
    try:
        return key, float(value)
    except ValueError:
        pass
    return key, value


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variant", default=None, help="config variant overlay name")
    parser.add_argument(
        "--override",
        action="append",
        type=_parse_override,
        metavar="K=V",
        help="dotted config override, e.g. dspy.tasks.answer.model=sonnet5",
    )


def _config_from(args: argparse.Namespace) -> AppConfig:
    from gluomem.config.app_config import load_config

    return load_config(
        variant=args.variant, overrides=dict(args.override or [])
    )


def _cmd_ingest(args: argparse.Namespace) -> int:
    from gluomem.llm import make_extractors
    from gluomem.write.ingest import ingest_path

    cfg = _config_from(args)
    facade = make_extractors(cfg)
    try:
        report = ingest_path(args.path, cfg, facade=facade, db_path=args.db)
    except (FileNotFoundError, ValueError) as exc:
        print(f"gluomem ingest: {exc}", file=sys.stderr)
        return 1
    print(
        f"ingested {report.conversations} conversation(s), "
        f"{report.sessions} session(s), {report.turns} turn(s): "
        f"I={report.i_frames} P={report.p_frames} B={report.b_frames} "
        f"deduped={report.deduped} quarantined={report.quarantined} "
        f"rejected={report.rejected} entities={report.entities} "
        f"relations={report.relations} retired_edges={report.retired_edges} "
        f"demoted={report.demoted}"
    )
    for err in report.extraction_errors:
        print(f"extraction error: {err}", file=sys.stderr)
    return 0


# --- components / index -------------------------------------------------------


def _open_existing_db(path: str, cfg: AppConfig):  # noqa: ANN202 — MindDB, imported lazily
    from pathlib import Path

    from gluomem.storage.db import MindDB

    if not Path(path).expanduser().is_file():
        raise FileNotFoundError(f"no mind db at {path}")
    return MindDB(path, dim=cfg.storage.vector_dim)


def _cmd_components(args: argparse.Namespace) -> int:
    import json

    from gluomem.components import component_report

    cfg = _config_from(args)
    db = None
    if args.db:
        try:
            db = _open_existing_db(args.db, cfg)
        except FileNotFoundError as exc:
            print(f"gluomem components: {exc}", file=sys.stderr)
            return 1
    try:
        report = component_report(cfg, db=db)
    finally:
        if db is not None:
            db.close()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_ready:
        store_ok = report.get("store", {}).get("embedder_matches") is not False
        return 0 if report["ready"] and store_ok else 3
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    from gluomem.embedding.compat import adopt_embedder
    from gluomem.embedding.providers import make_embedder
    from gluomem.frames.store import FrameStore

    if args.index_command not in ("rebuild", "adopt"):
        print("gluomem index: pick a subcommand (adopt, rebuild)", file=sys.stderr)
        return 1
    cfg = _config_from(args)
    try:
        db = _open_existing_db(args.db, cfg)
    except FileNotFoundError as exc:
        print(f"gluomem index: {exc}", file=sys.stderr)
        return 1
    try:
        embedder = make_embedder(cfg)
        if args.index_command == "rebuild":
            store = FrameStore.rebuild_index(db, embedder, chunking=cfg.chunking)
            n = db.conn.execute("SELECT COUNT(*) FROM memory_frames").fetchone()[0]
            print(f"rebuilt index for {n} frame(s); fingerprint {store.fingerprint.digest}")
        else:
            fp = adopt_embedder(db, embedder)
            print(f"adopted fingerprint {fp.digest} without re-embedding")
    finally:
        db.close()
    return 0


# --- bench subcommands ---------------------------------------------------------


def _cmd_bench_run(args: argparse.Namespace) -> int:
    from gluomem.bench.adapters.locomo import DatasetError
    from gluomem.bench.runner import run_benchmark

    cfg = _config_from(args)
    try:
        paths = run_benchmark(
            cfg,
            dataset=args.dataset,
            retriever=args.retriever,
            db_dir=args.db_dir,
            out_root=args.out,
            limit=args.limit,
        )
    except (DatasetError, FileNotFoundError, ValueError) as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    print(f"run dir: {paths.root}")
    if args.gate:
        import json

        entries = json.loads(paths.scores.read_text(encoding="utf-8"))
        return _gate_verdict(entries, cfg)
    return 0


def _gate_verdict(entries: list[dict], cfg: AppConfig) -> int:
    from gluomem.bench.report import check_gate

    gate = cfg.benchmark.ci_gate
    try:
        ok, entry = check_gate(entries, gate)
    except ValueError as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    if not ok:
        print(
            f"GATE FAIL metric={gate.metric} value={entry['value']:.4f} "
            f"min={gate.min}",
            file=sys.stderr,
        )
        return 1
    print(f"GATE PASS metric={gate.metric} value={entry['value']:.4f} min={gate.min}")
    return 0


def _cmd_bench_rescore(args: argparse.Namespace) -> int:
    from gluomem.bench.rescore import RescoreMismatchError, rescore_run

    try:
        entries = rescore_run(args.run_dir, check=args.check)
    except (FileNotFoundError, RescoreMismatchError) as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    verb = "verified" if args.check else "rewrote"
    print(f"rescore {verb} {len(entries)} score entries from {args.run_dir}")
    return 0


def _cmd_bench_gate(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    cfg = _config_from(args)
    scores_path = Path(args.run_dir) / "scores.json"
    try:
        entries = json.loads(scores_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    return _gate_verdict(entries, cfg)


def _cmd_bench_fetch(args: argparse.Namespace) -> int:
    from gluomem.bench.adapters import make_adapter
    from gluomem.bench.adapters.locomo import DatasetError

    cfg = _config_from(args)
    try:
        adapter = make_adapter(args.dataset, cfg)
        path = adapter.fetch(force=args.force)
    except (DatasetError, ValueError, OSError) as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    print(f"fetched {args.dataset} -> {path} (sha256 {adapter.sha})")
    return 0


def _cmd_bench_sample(args: argparse.Namespace) -> int:
    from pathlib import Path

    from gluomem.bench.adapters import make_adapter
    from gluomem.bench.adapters.locomo import (
        DatasetError,
        stratified_sample,
        write_sample_manifest,
    )

    cfg = _config_from(args)
    try:
        adapter = make_adapter(args.dataset, cfg)
        adapter.verify()
        samples = stratified_sample(
            adapter.load(),
            per_category=cfg.benchmark.sample.per_category,
            categories=cfg.benchmark.sample.categories,
            seed=cfg.benchmark.sample.seed,
        )
        manifest_path = Path(cfg.benchmark.sample.manifest)
        write_sample_manifest(
            samples, adapter=adapter, cfg=cfg, path=manifest_path
        )
    except (DatasetError, ValueError) as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    counts = adapter.eligible_counts()
    print(f"eligible per category: {counts}")
    print(f"wrote {len(samples)}-sample manifest -> {manifest_path}")
    return 0


def _conv_ids_from_manifest(path: str | None) -> set[str] | None:
    """Conversation ids referenced by a sample manifest (None = no filter).

    LoCoMo sample ids are ``{conv_id}:qa:{idx}``; LongMemEval ids ARE the
    conv_id — strip the qa suffix when present, pass through otherwise."""
    if path is None:
        return None
    import json
    from pathlib import Path

    try:
        ids = json.loads(Path(path).read_text(encoding="utf-8"))["ids"]
        return {i.split(":qa:")[0] for i in ids}
    except OSError as exc:
        raise ValueError(f"cannot read conv-ids manifest {path}: {exc}") from exc
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(
            f"invalid conv-ids manifest {path} (need a JSON object with an "
            f"'ids' list of strings): {exc!r}"
        ) from exc


def _cmd_bench_ingest(args: argparse.Namespace) -> int:
    from pathlib import Path

    from gluomem.bench.adapters.locomo import DatasetError

    cfg = _config_from(args)
    try:
        conv_ids = _conv_ids_from_manifest(args.conv_ids_from)
    except ValueError as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    if args.system == "mem0":
        # Competitor ingest: never touches make_extractors/dspy.
        from gluomem.bench.mem0 import ingest_mem0_per_conversation

        try:
            mem0_reports = ingest_mem0_per_conversation(
                cfg,
                dataset=args.dataset,
                db_dir=Path(args.db_dir),
                skip_existing=args.skip_existing,
                conv_ids=conv_ids,
            )
        except (DatasetError, ValueError, RuntimeError) as exc:
            print(f"gluomem bench: {exc}", file=sys.stderr)
            return 1
        print(
            f"ingested {len(mem0_reports)} conversation(s) into "
            f"{args.db_dir} (mem0)"
        )
        return 0

    from gluomem.bench.runner import ingest_per_conversation
    from gluomem.llm import make_extractors

    facade = make_extractors(cfg)
    try:
        reports = ingest_per_conversation(
            cfg,
            dataset=args.dataset,
            db_dir=Path(args.db_dir),
            facade=facade,
            skip_existing=args.skip_existing,
            conv_ids=conv_ids,
        )
    except (DatasetError, ValueError) as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    total_errors = [e for r in reports for e in r.extraction_errors]
    print(f"ingested {len(reports)} conversation(s) into {args.db_dir}")
    for err in total_errors:
        print(f"extraction error: {err}", file=sys.stderr)
    return 0


def _cmd_bench_verify(args: argparse.Namespace) -> int:
    from pathlib import Path

    from gluomem.bench.adapters.locomo import DatasetError
    from gluomem.bench.runner import verify_ingest_completeness

    cfg = _config_from(args)
    try:
        rows = verify_ingest_completeness(
            cfg,
            dataset=args.dataset,
            db_dir=Path(args.db_dir),
            write_sentinels=args.write_sentinels,
        )
    except (DatasetError, ValueError) as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    incomplete = [r for r in rows if r["status"] == "incomplete"]
    complete = sum(1 for r in rows if r["status"] == "complete")
    absent = sum(1 for r in rows if r["status"] == "absent")
    for r in incomplete:
        print(
            f"INCOMPLETE {r['conv_id']}: {r['found_sessions']}/"
            f"{r['expected_sessions']} sessions",
            file=sys.stderr,
        )
    print(
        f"verify {args.db_dir}: {complete} complete, "
        f"{len(incomplete)} incomplete, {absent} absent"
    )
    if incomplete:
        print(
            "delete the incomplete .mind files and re-ingest with "
            "--skip-existing",
            file=sys.stderr,
        )
        return 1
    return 0


def _add_bench_parsers(sub: argparse._SubParsersAction) -> None:
    bench = sub.add_parser("bench", help="run / rescore benchmark evaluations")
    bench_sub = bench.add_subparsers(dest="bench_command")

    run = bench_sub.add_parser("run", help="run a benchmark end-to-end")
    run.add_argument("--dataset", required=True, help="dataset name (locomo)")
    run.add_argument(
        "--retriever",
        required=True,
        choices=["gluomem", "full_context", "vector_rag", "bm25", "mem0"],
    )
    run.add_argument("--db-dir", default=None, help="per-conversation .mind dir")
    run.add_argument("--out", default=None, help="run-dir root (default: config)")
    run.add_argument("--limit", type=int, default=None, help="truncate sample")
    run.add_argument(
        "--gate", action="store_true", help="apply the CI gate to this run"
    )
    _add_config_args(run)

    rescore = bench_sub.add_parser(
        "rescore", help="re-derive scores/report offline (no model calls)"
    )
    rescore.add_argument("run_dir", help="a bench run directory")
    rescore.add_argument(
        "--check",
        action="store_true",
        help="verify committed artifacts instead of rewriting",
    )

    gate = bench_sub.add_parser("gate", help="apply the CI gate to a run dir")
    gate.add_argument("run_dir", help="a bench run directory")
    _add_config_args(gate)

    fetch = bench_sub.add_parser("fetch", help="download + SHA-verify a dataset")
    fetch.add_argument("--dataset", required=True)
    fetch.add_argument("--force", action="store_true", help="re-download")
    _add_config_args(fetch)

    sample = bench_sub.add_parser(
        "sample", help="write the committed stratified sample manifest"
    )
    sample.add_argument("--dataset", required=True)
    _add_config_args(sample)

    ingest = bench_sub.add_parser(
        "ingest", help="ingest a dataset into per-conversation memory stores"
    )
    ingest.add_argument("--dataset", required=True)
    ingest.add_argument("--db-dir", required=True)
    ingest.add_argument(
        "--system",
        choices=["gluomem", "mem0"],
        default="gluomem",
        help="memory system to ingest into (mem0 needs `uv sync --extra mem0`)",
    )
    ingest.add_argument(
        "--conv-ids-from",
        default=None,
        help=(
            "sample-manifest JSON; restrict ingest to the conversations its "
            "ids reference (LongMemEval: id == conv_id; LoCoMo: id prefix)"
        ),
    )
    ingest.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip conversations already ingested (resume after interruption)",
    )
    _add_config_args(ingest)

    verify = bench_sub.add_parser(
        "verify",
        help="audit a per-conversation .mind dir for silently-partial ingests",
    )
    verify.add_argument("--dataset", required=True)
    verify.add_argument("--db-dir", required=True)
    verify.add_argument(
        "--write-sentinels",
        action="store_true",
        help=(
            "write the completion sentinel for verified-complete conversations "
            "(bless a pre-sentinel ingest dir for --skip-existing resume)"
        ),
    )
    _add_config_args(verify)


_BENCH_HANDLERS = {
    "run": _cmd_bench_run,
    "rescore": _cmd_bench_rescore,
    "gate": _cmd_bench_gate,
    "fetch": _cmd_bench_fetch,
    "sample": _cmd_bench_sample,
    "ingest": _cmd_bench_ingest,
    "verify": _cmd_bench_verify,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gluomem",
        description="GluoMem — deterministic memory core (experiment runner).",
    )
    sub = parser.add_subparsers(dest="command")

    ingest = sub.add_parser(
        "ingest", help="ingest a LoCoMo-format benchmark file into memory"
    )
    ingest.add_argument("path", help="path to a LoCoMo-format JSON file")
    ingest.add_argument("--db", default=None, help="mind db path (default: config)")
    _add_config_args(ingest)

    _add_bench_parsers(sub)

    components = sub.add_parser(
        "components",
        help="print effective component implementations and readiness (no secrets)",
    )
    components.add_argument("--db", default=None, help="also check this mind db's fingerprint")
    components.add_argument(
        "--require-ready",
        action="store_true",
        help="exit 3 unless every component is real and available",
    )
    _add_config_args(components)

    index = sub.add_parser("index", help="explicit embedding index maintenance")
    index_sub = index.add_subparsers(dest="index_command")
    for name, text in (
        ("rebuild", "re-embed every frame with the configured embedder and record it"),
        ("adopt", "record the configured embedder for a pre-fingerprint file, no re-embed"),
    ):
        p = index_sub.add_parser(name, help=text)
        p.add_argument("--db", required=True, help="mind db path")
        _add_config_args(p)

    # Stub subcommands; not implemented yet.
    sub.add_parser("tune", help="offline parameter tuning")
    sub.add_parser("ablate", help="run ablation experiments")

    args = parser.parse_args(argv)
    from gluomem.components import FakeComponentError, ProviderUnavailableError
    from gluomem.embedding.compat import EmbeddingMismatchError
    from gluomem.extras import MissingExtraError

    try:
        return _dispatch(args, parser)
    except (
        MissingExtraError,
        ProviderUnavailableError,
        FakeComponentError,
        EmbeddingMismatchError,
    ) as exc:
        print(f"gluomem: {exc}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "ingest":
        return _cmd_ingest(args)
    if args.command == "components":
        return _cmd_components(args)
    if args.command == "index":
        return _cmd_index(args)
    if args.command == "bench":
        handler = _BENCH_HANDLERS.get(args.bench_command or "")
        if handler is None:
            print(
                "gluomem bench: pick a subcommand "
                f"({', '.join(sorted(_BENCH_HANDLERS))})",
                file=sys.stderr,
            )
            return 1
        return handler(args)
    print(
        f"gluomem: '{args.command}' is not implemented yet (arrives in M6+).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
