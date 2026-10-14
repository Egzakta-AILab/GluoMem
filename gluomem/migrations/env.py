"""Alembic environment for schema-isolated PostgreSQL migrations."""

from __future__ import annotations

from alembic import context
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine, inspect, pool, text

from gluomem.migrations.runner import validate_schema_name
from gluomem.storage.pg_schema import build_metadata

config = context.config
schema = validate_schema_name(str(config.attributes.get("schema", "gluomem")))
target_metadata = build_metadata(int(config.attributes.get("vector_dim", 384)))


def render_item(type_, obj, autogen_context):
    if type_ == "type" and isinstance(obj, Vector):
        autogen_context.imports.add("import pgvector.sqlalchemy")
        return "pgvector.sqlalchemy.Vector(dim=vector_dim)"
    return False


def include_name(name, type_, parent_names):
    if type_ == "schema":
        # The selected search_path is the default schema to the inspector.
        return name is None
    return True


def include_object(obj, name, type_, reflected, compare_to):
    return type_ != "table" or name in target_metadata.tables


def configure(**kwargs) -> None:
    context.configure(
        target_metadata=target_metadata,
        version_table_schema=schema,
        include_schemas=True,
        include_name=include_name,
        include_object=include_object,
        compare_type=True,
        render_item=render_item,
        **kwargs,
    )


def run_migrations_offline() -> None:
    """Emit SQL that can be applied to an empty database."""
    configure(
        url=config.get_main_option("sqlalchemy.url"),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        # The version table must not precede creation of its containing schema.
        context.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        context.execute(f"SET LOCAL search_path TO {schema}, public")
        context.run_migrations()


def run_with_connection(connection) -> None:
    connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
    connection.execute(text(f"SET LOCAL search_path TO {schema}, public"))
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public"))
    inspector = inspect(connection)
    tables = set(inspector.get_table_names(schema=schema))
    if tables & target_metadata.tables.keys() and "alembic_version" not in tables:
        raise RuntimeError("Unversioned PostgreSQL schema; recreate the database")
    # SQLAlchemy otherwise reflects unqualified tables as public.
    previous_schema = connection.dialect.default_schema_name
    connection.dialect.default_schema_name = schema
    try:
        configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()
    finally:
        connection.dialect.default_schema_name = previous_schema


def run_migrations_online() -> None:
    """Migrate using the supplied connection or a new transaction."""
    connection = config.attributes.get("connection")
    if connection is not None:
        run_with_connection(connection)
        return
    engine = create_engine(config.get_main_option("sqlalchemy.url"), poolclass=pool.NullPool)
    try:
        with engine.begin() as connection:
            run_with_connection(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
