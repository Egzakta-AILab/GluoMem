"""Shared pinning, store isolation and literal marker matching for fixtures."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

from gluomem.bench.registry import BenchmarkEntry, get_entry
from gluomem.config.app_config import AppConfig
from gluomem.embedding.base import Embedder
from gluomem.frames.store import FrameStore
from gluomem.search.multilingual import fold_to_ascii
from gluomem.storage.db import MindDB

T = TypeVar("T")


def load_pinned_fixture(
    entry_id: str,
    paths: Mapping[str, Path],
    parser: Callable[[Path], tuple[T, str]],
    err: type[ValueError],
) -> tuple[BenchmarkEntry, T, str]:
    """Load a generated fixture only when its bytes match a required registry pin."""
    entry = get_entry(entry_id)
    path = paths.get(entry_id)
    if path is None or not path.is_file():
        raise err(f"no fixture registered for {entry_id!r}")
    if entry.sha256 is None:
        raise err(f"fixture {entry_id!r} has no registry pin")
    parsed, sha = parser(path)
    if sha != entry.sha256:
        raise err(f"{path}: sha256 {sha} != registry pin {entry.sha256}")
    return entry, parsed, sha


@contextmanager
def fresh_store(
    work_dir: Path, name: str, cfg: AppConfig, embedder: Embedder
) -> Iterator[tuple[MindDB, FrameStore]]:
    """Open a fresh isolated fixture store and always close it."""
    if Path(name).name != name:
        raise ValueError("fixture store name must be a single path component")
    path = work_dir / f"{name}.mind"
    if path.exists():
        raise FileExistsError(f"{path} already exists; each fixture requires a fresh store")
    path.parent.mkdir(parents=True, exist_ok=True)
    db = MindDB(path, dim=cfg.storage.vector_dim)
    try:
        yield db, FrameStore(db, embedder=embedder, chunking=cfg.chunking)
    finally:
        db.close()


def contains_folded(needle: str, hay: str) -> bool:
    """Match a nonempty literal marker after script and diacritic folding."""
    folded = fold_to_ascii(needle)
    return bool(folded) and folded in fold_to_ascii(hay)
