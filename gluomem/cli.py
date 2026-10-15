"""GluoMem CLI entrypoint (``gluomem``).

Exposes operational commands for the memory engine:
- ``gluomem components`` — effective component implementations and readiness
- ``gluomem index`` — explicit embedding index rebuild and fingerprint adoption
- ``gluomem migrate`` — Alembic PostgreSQL schema migrations
- ``gluomem worker`` — durable background job worker
- ``gluomem serve`` — authenticated FastAPI HTTP service
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
        raise argparse.ArgumentTypeError(f"override {raw!r} must be dotted.key=value")
    lowered = value.lower()
    if lowered in ("true", "false"):
        return key, lowered == "true"
    if value[:1] in ("[", "{"):
        import json

        try:
            return key, json.loads(value)
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(
                f"override {raw!r}: value looks like JSON but fails to parse ({exc})"
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
        help="dotted config override, e.g. embedding.model=minilm",
    )


def _config_from(args: argparse.Namespace) -> AppConfig:
    from gluomem.config.app_config import load_config

    return load_config(overrides=dict(args.override or []))


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
        store = FrameStore(db, embedder=embedder, chunking=cfg.chunking, lifecycle=cfg.lifecycle)
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
    worker = sub.add_parser("worker", help="run durable memory_jobs background worker")
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
        store = FrameStore(db, embedder=embedder, chunking=cfg.chunking, lifecycle=cfg.lifecycle)
        kg = KnowledgeGraph(db)
        queue = JobQueue(db.backend, max_pending_per_scope=args.max_pending_per_scope)
        app = create_app(
            db,
            store,
            cfg=cfg,
            service_key=(
                Path(args.service_key_file).read_text().strip() if args.service_key_file else None
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
        "--max-pending-per-scope",
        type=int,
        default=50,
        help="per-scope pending job cap (default: 50)",
    )
    _add_config_args(serve)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gluomem",
        description="GluoMem — deterministic memory core.",
    )
    sub = parser.add_subparsers(dest="command")

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
    if args.command == "components":
        return _cmd_components(args)
    if args.command == "index":
        return _cmd_index(args)
    if args.command == "migrate":
        return _cmd_migrate(args)
    if args.command == "worker":
        return _cmd_worker(args)
    if args.command == "serve":
        return _cmd_serve(args)
    print(
        f"gluomem: '{args.command}' is not implemented yet.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
