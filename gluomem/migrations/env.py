"""Alembic environment script for GluoMem PostgreSQL schema migrations."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine, pool, text

from gluomem.migrations.runner import validate_schema_name

config = context.config
schema: str = validate_schema_name(str(config.attributes.get("schema", "gluomem")))


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a DB connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=None,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table="alembic_version",
        version_table_schema=schema,
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def _run_with_connection(connection) -> None:  # noqa: ANN001
    dialect_name = connection.dialect.name
    version_schema = schema if dialect_name == "postgresql" else None
    if dialect_name == "postgresql":
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        connection.execute(text(f"SET search_path TO {schema}, public"))

    context.configure(
        connection=connection,
        target_metadata=None,
        version_table="alembic_version",
        version_table_schema=version_schema,
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live SQLAlchemy connection or engine."""
    provided_conn = config.attributes.get("connection")
    if provided_conn is not None:
        _run_with_connection(provided_conn)
        return

    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("sqlalchemy.url is not configured for online migration")
    connectable = create_engine(url, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        _run_with_connection(connection)
        connection.commit()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
