"""gluomem.storage — relational backends, vector stores, FTS stores, and MindDB facade."""

from gluomem.storage.backend import (
    DatabaseBackend,
    FtsStore,
    VectorDimMismatchError,
    VectorStore,
    assert_dim,
)
from gluomem.storage.db import MindDB
from gluomem.storage.sqlite_backend import SqliteBackend, SqliteFts5Store
from gluomem.storage.sqlite_vec import SqliteVecStore

__all__ = [
    "DatabaseBackend",
    "FtsStore",
    "MindDB",
    "SqliteBackend",
    "SqliteFts5Store",
    "SqliteVecStore",
    "VectorDimMismatchError",
    "VectorStore",
    "assert_dim",
]
