"""Programmatic Alembic configuration and migration helpers."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

from gluomem.extras import require

_MIGRATIONS_DIR = Path(__file__).resolve().parent
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def validate_schema_name(schema: str) -> str:
    """Validate that ``schema`` is a safe SQL identifier."""
    if not _SAFE_IDENTIFIER_RE.fullmatch(schema):
        raise ValueError(f"unsafe PostgreSQL schema identifier: {schema!r}")
    return schema


def normalize_sqlalchemy_url(dsn: str) -> str:
    """Normalize a Postgres DSN into a SQLAlchemy psycopg v3 URL."""
    raw = dsn.strip()
    if not raw:
        raise ValueError("database DSN must not be empty")
    if raw.startswith("postgres://"):
        return "postgresql+psycopg://" + raw[len("postgres://") :]
    if raw.startswith("postgresql://"):
        return "postgresql+psycopg://" + raw[len("postgresql://") :]
    return raw


def make_alembic_config(
    dsn: str = "postgresql+psycopg://localhost/gluomem",
    *,
    schema: str = "gluomem",
    vector_dim: int = 384,
    connection: Any | None = None,
    output_buffer: io.StringIO | None = None,
) -> Any:
    """Build an in-memory ``alembic.config.Config`` pointing at ``gluomem/migrations``."""
    require("postgres", "Alembic PostgreSQL migrations")
    from alembic.config import Config

    validate_schema_name(schema)
    if int(vector_dim) <= 0:
        raise ValueError(f"vector_dim must be positive, got {vector_dim}")

    cfg = Config(output_buffer=output_buffer)
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", normalize_sqlalchemy_url(dsn))
    cfg.attributes["schema"] = schema
    cfg.attributes["vector_dim"] = int(vector_dim)
    if connection is not None:
        cfg.attributes["connection"] = connection
    return cfg


def get_heads() -> list[str]:
    """Return the current Alembic head revision IDs."""
    require("postgres", "Alembic PostgreSQL migrations")
    from alembic.script import ScriptDirectory

    cfg = make_alembic_config()
    script = ScriptDirectory.from_config(cfg)
    return list(script.get_heads())


def get_revisions() -> list[dict[str, Any]]:
    """Return all Alembic revisions ordered from base to head."""
    require("postgres", "Alembic PostgreSQL migrations")
    from alembic.script import ScriptDirectory

    cfg = make_alembic_config()
    script = ScriptDirectory.from_config(cfg)
    revisions: list[dict[str, Any]] = []
    for rev in reversed(list(script.walk_revisions())):
        revisions.append(
            {
                "revision": rev.revision,
                "down_revision": rev.down_revision,
                "doc": rev.doc,
            }
        )
    return revisions


def run_migration(
    dsn: str,
    revision: str = "head",
    *,
    direction: str = "upgrade",
    schema: str = "gluomem",
    vector_dim: int = 384,
    sql: bool = False,
    connection: Any | None = None,
) -> str:
    """Run an Alembic upgrade or downgrade, optionally rendering offline SQL."""
    if direction not in ("upgrade", "downgrade"):
        raise ValueError("direction must be 'upgrade' or 'downgrade'")
    buf = io.StringIO()
    cfg = make_alembic_config(
        dsn, schema=schema, vector_dim=vector_dim, connection=connection, output_buffer=buf
    )
    from alembic import command

    getattr(command, direction)(cfg, revision, sql=sql)
    return buf.getvalue()


def run_upgrade(dsn: str, revision: str = "head", **kwargs: Any) -> str:
    """Run ``alembic upgrade`` to ``revision``."""
    return run_migration(dsn, revision, direction="upgrade", **kwargs)


def run_downgrade(dsn: str, revision: str = "base", **kwargs: Any) -> str:
    """Run ``alembic downgrade`` to ``revision``."""
    return run_migration(dsn, revision, direction="downgrade", **kwargs)


def run_revision(
    dsn: str,
    message: str,
    *,
    schema: str = "gluomem",
    vector_dim: int = 384,
    autogenerate: bool = False,
) -> Any:
    """Generate an Alembic revision against the configured metadata."""
    cfg = make_alembic_config(dsn, schema=schema, vector_dim=vector_dim)
    from alembic import command

    return command.revision(
        cfg,
        message=message,
        autogenerate=autogenerate,
        rev_id="0001" if not get_heads() else None,
    )


def get_current(dsn: str, *, schema: str = "gluomem") -> list[str]:
    """Read installed revisions without creating or changing a schema."""
    require("postgres", "Alembic PostgreSQL migrations")
    from sqlalchemy import create_engine, text

    engine = create_engine(normalize_sqlalchemy_url(dsn))
    try:
        with engine.connect() as conn:
            schema = validate_schema_name(schema)
            if (
                conn.scalar(
                    text("SELECT to_regclass(:name)"), {"name": f"{schema}.alembic_version"}
                )
                is None
            ):
                return []
            return list(conn.scalars(text(f"SELECT version_num FROM {schema}.alembic_version")))
    finally:
        engine.dispose()


def render_migration_sql(
    *,
    direction: str = "upgrade",
    revision: str | None = None,
    schema: str = "gluomem",
    vector_dim: int = 384,
) -> str:
    """Render offline SQL for ``upgrade`` or ``downgrade`` without a live database."""
    if direction == "upgrade":
        target = revision or "head"
        return run_upgrade(
            "postgresql+psycopg://localhost/gluomem",
            target,
            schema=schema,
            vector_dim=vector_dim,
            sql=True,
        )
    if direction == "downgrade":
        target = revision or "0001:base"
        return run_downgrade(
            "postgresql+psycopg://localhost/gluomem",
            target,
            schema=schema,
            vector_dim=vector_dim,
            sql=True,
        )
    raise ValueError(f"direction must be 'upgrade' or 'downgrade', got {direction!r}")
