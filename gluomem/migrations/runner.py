"""Programmatic Alembic configuration and migration helpers."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

from gluomem.extras import require

_MIGRATIONS_DIR = Path(__file__).resolve().parent
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_schema_name(schema: str) -> str:
    """Validate that ``schema`` is a safe SQL identifier."""
    if not _SAFE_IDENTIFIER_RE.match(schema):
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


def get_heads(
    *,
    schema: str = "gluomem",
    vector_dim: int = 384,
) -> list[str]:
    """Return the current Alembic head revision IDs."""
    require("postgres", "Alembic PostgreSQL migrations")
    from alembic.script import ScriptDirectory

    cfg = make_alembic_config(schema=schema, vector_dim=vector_dim)
    script = ScriptDirectory.from_config(cfg)
    return list(script.get_heads())


def get_revisions(
    *,
    schema: str = "gluomem",
    vector_dim: int = 384,
) -> list[dict[str, Any]]:
    """Return all Alembic revisions ordered from base to head."""
    require("postgres", "Alembic PostgreSQL migrations")
    from alembic.script import ScriptDirectory

    cfg = make_alembic_config(schema=schema, vector_dim=vector_dim)
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


def run_upgrade(
    dsn: str,
    revision: str = "head",
    *,
    schema: str = "gluomem",
    vector_dim: int = 384,
    sql: bool = False,
    connection: Any | None = None,
) -> str:
    """Run ``alembic upgrade`` to ``revision`` (or render SQL when ``sql=True``)."""
    require("postgres", "Alembic PostgreSQL migrations")
    from alembic import command

    buf = io.StringIO()
    cfg = make_alembic_config(
        dsn,
        schema=schema,
        vector_dim=vector_dim,
        connection=connection,
        output_buffer=buf,
    )
    command.upgrade(cfg, revision, sql=sql)
    return buf.getvalue()


def run_downgrade(
    dsn: str,
    revision: str = "base",
    *,
    schema: str = "gluomem",
    vector_dim: int = 384,
    sql: bool = False,
    connection: Any | None = None,
) -> str:
    """Run ``alembic downgrade`` to ``revision`` (or render SQL when ``sql=True``)."""
    require("postgres", "Alembic PostgreSQL migrations")
    from alembic import command

    buf = io.StringIO()
    cfg = make_alembic_config(
        dsn,
        schema=schema,
        vector_dim=vector_dim,
        connection=connection,
        output_buffer=buf,
    )
    command.downgrade(cfg, revision, sql=sql)
    return buf.getvalue()


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
        target = revision or "0001_initial_scoped_schema:base"
        return run_downgrade(
            "postgresql+psycopg://localhost/gluomem",
            target,
            schema=schema,
            vector_dim=vector_dim,
            sql=True,
        )
    raise ValueError(f"direction must be 'upgrade' or 'downgrade', got {direction!r}")
