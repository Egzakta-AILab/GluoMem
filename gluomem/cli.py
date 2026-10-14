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
    parser.add_argument(
        "--override",
        action="append",
        type=_parse_override,
        metavar="K=V",
        help="dotted config override, e.g. dspy.tasks.answer.model=sonnet5",
    )


def _config_from(args: argparse.Namespace) -> AppConfig:
    from gluomem.config.app_config import load_config

    return load_config(overrides=dict(args.override or []))


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
            n = store.count_frames()
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


def _cmd_bench_replay(args: argparse.Namespace) -> int:
    from gluomem.bench.registry import RegistryError
    from gluomem.bench.replay import (
        ReplayFixtureError,
        ReplayNotReadyError,
        run_replay_fixture,
    )
    from gluomem.embedding.providers import make_embedder

    cfg = _config_from(args)
    try:
        receipt, digest = run_replay_fixture(
            cfg,
            benchmark_id=args.dataset,
            embedder=make_embedder(cfg),
            work_dir=args.work_dir,
            out_dir=args.out,
            candidate=args.candidate,
        )
    except (RegistryError, ReplayFixtureError, ReplayNotReadyError, FileExistsError) as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    print(receipt["label"])
    print(f"receipt {args.out}/receipt.json sha256 {digest}")
    return 0


def _cmd_bench_registry(args: argparse.Namespace) -> int:  # noqa: ARG001
    import json

    from gluomem.bench.registry import (
        REGISTRY,
        manifest_record,
        overlap_report,
        registry_digest,
    )

    print(
        json.dumps(
            {
                "digest": registry_digest(),
                "entries": [manifest_record(e) for _, e in sorted(REGISTRY.items())],
                "overlaps": overlap_report(REGISTRY),
            },
            indent=2,
            sort_keys=True,
        )
    )
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

    replay = bench_sub.add_parser(
        "replay",
        help="replay a registered fixture in event order and write a receipt (offline)",
    )
    replay.add_argument("--dataset", default="mab_fixture", help="registry fixture id")
    replay.add_argument("--work-dir", required=True, help="fresh dir for per-history stores")
    replay.add_argument("--out", required=True, help="receipt output dir")
    replay.add_argument("--candidate", default="baseline", help="candidate label")
    _add_config_args(replay)

    bench_sub.add_parser(
        "registry", help="print the benchmark registry, digests and source overlaps"
    )


def _resolve_tune_cache(args: argparse.Namespace, cfg: AppConfig):  # noqa: ANN202
    from pathlib import Path

    from gluomem.tune.candidate_cache import CandidateCache, build_cache_for_benchmark

    cache_dir = Path(args.cache) if getattr(args, "cache", None) else None
    rebuild = bool(getattr(args, "rebuild_cache", False))
    if cache_dir is not None and (cache_dir / "manifest.json").is_file() and not rebuild:
        return CandidateCache.load(cache_dir)
    if getattr(args, "db_dir", None):
        cache = build_cache_for_benchmark(
            cfg,
            db_dir=args.db_dir,
            dataset_name=getattr(args, "dataset", "locomo") or "locomo",
            sample_manifest=getattr(args, "sample_manifest", None),
            lane_fetch=int(getattr(args, "lane_fetch", 100)),
        )
        if cache_dir is not None:
            cache.save(cache_dir)
        return cache
    if cache_dir is not None:
        return CandidateCache.load(cache_dir)
    raise ValueError(
        "provide --cache <dir> (existing candidate cache) or --db-dir <dir> to build one"
    )


def _cmd_tune(args: argparse.Namespace) -> int:
    from gluomem import extras
    from gluomem.tune.objective import TuneConfig, run_optuna_tune

    extras.require("tune", "Offline retrieval parameter tuning")
    cfg = _config_from(args)
    try:
        cache = _resolve_tune_cache(args, cfg)
        tune_cfg = TuneConfig(
            n_trials=int(args.trials),
            k=int(args.k),
            metric=str(args.metric),
            profile=str(args.profile or cfg.search.default_profile),
            seed=int(args.seed),
            enable_pruner=not bool(args.no_prune),
        )
        result = run_optuna_tune(
            cache,
            cfg,
            tune_cfg=tune_cfg,
            out_dir=args.out,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"gluomem tune: {exc}", file=sys.stderr)
        return 1

    print(
        f"tuned {result.n_trials} trial(s) on {len(cache.query_ids())} query(s): "
        f"incumbent {result.metric}@{result.k}={result.incumbent_score:.4f} -> "
        f"best={result.best_score:.4f} (+{result.improvement:.4f}, "
        f"trial #{result.best_trial_number}, {result.elapsed_ms:.1f}ms, model_calls=0)"
    )
    return 0


def _cmd_ablate(args: argparse.Namespace) -> int:
    import json

    from gluomem.tune.ablate import format_ablation_table, run_ablation

    cfg = _config_from(args)
    try:
        cache = _resolve_tune_cache(args, cfg)
        report = run_ablation(
            cache,
            cfg,
            profile=args.profile,
            k=int(args.k),
            out_dir=args.out,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"gluomem ablate: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_ablation_table(report))
    return 0


def _add_tune_and_ablate_parsers(sub: argparse._SubParsersAction) -> None:
    tune = sub.add_parser(
        "tune", help="offline Tier-1 retrieval parameter tuning (Optuna TPE + Hyperband)"
    )
    tune.add_argument("--cache", default=None, help="candidate cache directory")
    tune.add_argument("--db-dir", default=None, help="per-conversation .mind dir to build cache")
    tune.add_argument("--dataset", default="locomo", help="dataset name when building cache")
    tune.add_argument("--sample-manifest", default=None, help="optional sample manifest JSON")
    tune.add_argument("--lane-fetch", type=int, default=100, help="per-lane candidate pool size")
    tune.add_argument("--rebuild-cache", action="store_true", help="force cache rebuild")
    tune.add_argument("--out", default=None, help="output dir for manifest/trials/best_params")
    tune.add_argument("--trials", type=int, default=200, help="number of Optuna trials")
    tune.add_argument("--k", type=int, default=10, help="cutoff k for metric@k")
    tune.add_argument(
        "--metric",
        choices=["ndcg", "recall", "hit"],
        default="ndcg",
        help="optimization objective metric",
    )
    tune.add_argument("--profile", default=None, help="scoring profile to tune")
    tune.add_argument("--seed", type=int, default=42, help="deterministic RNG seed")
    tune.add_argument("--no-prune", action="store_true", help="disable Hyperband pruner")
    _add_config_args(tune)

    ablate = sub.add_parser(
        "ablate", help="run deterministic retrieval signal and lane ablations over a cache"
    )
    ablate.add_argument("--cache", default=None, help="candidate cache directory")
    ablate.add_argument("--db-dir", default=None, help="per-conversation .mind dir to build cache")
    ablate.add_argument("--dataset", default="locomo", help="dataset name when building cache")
    ablate.add_argument("--sample-manifest", default=None, help="optional sample manifest JSON")
    ablate.add_argument("--lane-fetch", type=int, default=100, help="per-lane candidate pool size")
    ablate.add_argument("--rebuild-cache", action="store_true", help="force cache rebuild")
    ablate.add_argument("--out", default=None, help="output dir for ablation_report.json")
    ablate.add_argument("--k", type=int, default=10, help="cutoff k for metrics@k")
    ablate.add_argument("--profile", default=None, help="scoring profile to ablate")
    ablate.add_argument("--json", action="store_true", help="emit JSON instead of table")
    _add_config_args(ablate)


def _cmd_migrate(args: argparse.Namespace) -> int:
    from gluomem.migrations import get_current, get_heads, run_revision
    from gluomem.migrations.runner import run_migration

    subcmd = args.migrate_command
    if subcmd == "heads":
        print("\n".join(get_heads()))
        return 0
    if subcmd not in ("upgrade", "downgrade", "revision", "current"):
        print(
            "gluomem migrate: pick upgrade, downgrade, revision, heads or current", file=sys.stderr
        )
        return 1
    cfg = _config_from(args)
    schema = args.schema or cfg.storage.postgres_schema
    vector_dim = args.vector_dim if args.vector_dim is not None else cfg.storage.vector_dim
    sql = getattr(args, "sql", False)
    dsn = args.dsn or cfg.storage.dsn or ("postgresql+psycopg://localhost/gluomem" if sql else None)
    if not dsn:
        print("gluomem migrate: pass --dsn or set storage.dsn", file=sys.stderr)
        return 1
    try:
        if subcmd == "current":
            print("\n".join(get_current(dsn, schema=schema)))
        elif subcmd == "revision":
            revision = run_revision(
                dsn,
                args.message,
                schema=schema,
                vector_dim=vector_dim,
                autogenerate=args.autogenerate,
            )
            print(revision.path)
        else:
            out = run_migration(
                dsn,
                args.revision,
                direction=subcmd,
                schema=schema,
                vector_dim=vector_dim,
                sql=sql,
            )
            if sql:
                print(out.rstrip())
            else:
                print(
                    f"migrate {subcmd}: schema={schema} revision={args.revision} dim={vector_dim}"
                )
    except (ValueError, RuntimeError) as exc:
        print(f"gluomem migrate {subcmd}: {exc}", file=sys.stderr)
        return 1
    return 0


def _add_migrate_parsers(sub: argparse._SubParsersAction) -> None:
    migrate = sub.add_parser("migrate", help="run Alembic PostgreSQL schema migrations")
    commands = migrate.add_subparsers(dest="migrate_command")
    commands.add_parser("heads", help="print Alembic head revisions")
    for name in ("upgrade", "downgrade", "revision", "current"):
        parser = commands.add_parser(name, help=f"Alembic {name}")
        parser.add_argument("--dsn", default=None, help="PostgreSQL DSN (default: storage.dsn)")
        parser.add_argument("--schema", default=None, help="PostgreSQL schema")
        parser.add_argument("--vector-dim", type=int, default=None, help="vector dimension")
        if name in ("upgrade", "downgrade"):
            parser.add_argument(
                "revision", nargs="?", default="head" if name == "upgrade" else "base"
            )
            parser.add_argument("--sql", action="store_true", help="render offline SQL")
        elif name == "revision":
            parser.add_argument("--autogenerate", action="store_true")
            parser.add_argument("-m", "--message", required=True)
        _add_config_args(parser)


def _cmd_worker(args: argparse.Namespace) -> int:
    from gluomem.embedding.providers import make_embedder
    from gluomem.frames.store import FrameStore
    from gluomem.kg.graph import KnowledgeGraph
    from gluomem.service.jobs import JobQueue
    from gluomem.service.worker import MemoryWorker
    from gluomem.storage.db import MindDB

    cfg = _config_from(args)
    embedder = make_embedder(cfg)
    db = MindDB.from_config(cfg, path_override=args.db)
    try:
        store = FrameStore(
            db, embedder=embedder, chunking=cfg.chunking, lifecycle=cfg.lifecycle
        )
        kg = KnowledgeGraph(db)
        queue = JobQueue(
            db.backend,
            default_max_attempts=args.max_attempts,
        )
        worker = MemoryWorker(
            db,
            store,
            cfg=cfg,
            kg=kg,
            queue=queue,
            worker_id=args.worker_id,
            lease_seconds=args.lease_seconds,
        )
        if args.once:
            jobs = worker.run_until_empty(max_jobs=args.max_jobs)
            print(f"processed {len(jobs)} job(s)")
            return 0
        count = worker.run_forever(
            poll_interval=args.poll_interval,
            max_iterations=args.max_iterations,
        )
        print(f"processed {count} job(s)")
        return 0
    finally:
        db.close()


def _add_worker_parser(sub: argparse._SubParsersAction) -> None:
    worker = sub.add_parser(
        "worker", help="run durable memory_jobs background worker"
    )
    worker.add_argument("--db", default=None, help="mind db path (default: config)")
    worker.add_argument(
        "--worker-id", default=None, help="worker identifier (default: hostname-pid)"
    )
    worker.add_argument(
        "--lease-seconds",
        type=float,
        default=60.0,
        help="lease duration in seconds (default: 60)",
    )
    worker.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="idle poll interval in seconds (default: 1.0)",
    )
    worker.add_argument(
        "--once",
        action="store_true",
        help="drain currently queued jobs and exit",
    )
    worker.add_argument(
        "--max-jobs",
        type=int,
        default=100,
        help="maximum jobs to drain when --once is set",
    )
    worker.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="optional max poll iterations before exiting",
    )
    worker.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="default max attempts per job (default: 3)",
    )
    _add_config_args(worker)


def _cmd_serve(args: argparse.Namespace) -> int:
    from pathlib import Path

    from gluomem.embedding.providers import make_embedder
    from gluomem.extras import require
    from gluomem.frames.store import FrameStore
    from gluomem.kg.graph import KnowledgeGraph
    from gluomem.service.app import create_app
    from gluomem.service.jobs import JobQueue
    from gluomem.storage.db import MindDB

    require("service", "gluomem serve")
    import uvicorn

    cfg = _config_from(args)
    embedder = make_embedder(cfg)
    db = MindDB.from_config(cfg, path_override=args.db)
    try:
        store = FrameStore(
            db, embedder=embedder, chunking=cfg.chunking, lifecycle=cfg.lifecycle
        )
        kg = KnowledgeGraph(db)
        queue = JobQueue(db.backend, max_pending_per_scope=args.max_pending_per_scope)
        app = create_app(
            db,
            store,
            cfg=cfg,
            service_key=(
                Path(args.service_key_file).read_text().strip()
                if args.service_key_file else None
            ),
            kg=kg,
            queue=queue,
        )
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    finally:
        db.close()


def _add_serve_parser(sub: argparse._SubParsersAction) -> None:
    serve = sub.add_parser("serve", help="run authenticated GluoMem HTTP service")
    serve.add_argument("--db", default=None, help="mind db path (default: config)")
    serve.add_argument("--host", default="127.0.0.1", help="bind host")
    serve.add_argument("--port", type=int, default=8080, help="bind port")
    serve.add_argument(
        "--service-key-file",
        default=None,
        help="file containing bearer secret (default: GLUOMEM_SERVICE_KEY)",
    )
    serve.add_argument(
        "--max-pending-per-scope", type=int, default=50,
        help="per-scope pending job cap (default: 50)",
    )
    _add_config_args(serve)


_BENCH_HANDLERS = {
    "replay": _cmd_bench_replay,
    "registry": _cmd_bench_registry,
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

    _add_tune_and_ablate_parsers(sub)
    _add_migrate_parsers(sub)
    _add_worker_parser(sub)
    _add_serve_parser(sub)

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
    if args.command == "tune":
        return _cmd_tune(args)
    if args.command == "ablate":
        return _cmd_ablate(args)
    if args.command == "migrate":
        return _cmd_migrate(args)
    if args.command == "worker":
        return _cmd_worker(args)
    if args.command == "serve":
        return _cmd_serve(args)
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
        f"gluomem: '{args.command}' is not implemented yet.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
