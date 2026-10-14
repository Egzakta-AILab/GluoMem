"""Installed-wheel boundary check.

Run with the interpreter of an environment where the built wheel is installed,
from a directory outside the repository:

    python /path/to/scripts/wheel_smoke.py [--expect-core-only]

It proves that the installed package (not the checkout) loads its packaged
schema and default configuration, performs a deterministic write, reopen and
read, and, with ``--expect-core-only``, that model-backed components fail with
``MissingExtraError`` instead of silently using a mock. Prints a JSON receipt.
No network access and no model downloads.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-core-only", action="store_true")
    args = parser.parse_args()

    import gluomem
    from gluomem import extras
    from gluomem.config.app_config import load_config
    from gluomem.embedding.providers import MiniLMEmbedder, MockEmbedder
    from gluomem.frames.store import FrameStore
    from gluomem.search.hybrid import HybridSearch
    from gluomem.storage.db import MindDB

    package_dir = Path(gluomem.__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[1]
    if repo_root in package_dir.parents or Path.cwd().resolve() == repo_root:
        print(f"FAIL: gluomem imported from the checkout ({package_dir})", file=sys.stderr)
        return 1
    if not (package_dir / "storage" / "schema.sql").is_file():
        print("FAIL: storage/schema.sql is not packaged", file=sys.stderr)
        return 1
    if not (package_dir / "storage" / "schema_postgres.sql").is_file():
        print("FAIL: storage/schema_postgres.sql is not packaged", file=sys.stderr)
        return 1
    if not (package_dir / "config" / "default_config.yaml").is_file():
        print("FAIL: config/default_config.yaml is not packaged", file=sys.stderr)
        return 1

    cfg = load_config()
    dim = int(cfg.storage.vector_dim)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke.mind"
        embedder = MockEmbedder(dim=dim)
        db = MindDB(path, dim=dim)
        store = FrameStore(db, embedder=embedder, chunking=cfg.chunking)
        written = store.create_i_frame("smoke", "The deploy window is Tuesday at 10:00 UTC.")
        db.close()

        db = MindDB(path, dim=dim)
        store = FrameStore(db, embedder=embedder, chunking=cfg.chunking)
        search = HybridSearch(db, store, cfg, embedder=embedder)
        result = search.search("deploy window", limit=5)
        db.close()
    hit_ids = [h.frame.id for h in result.hits]
    if written.id not in hit_ids:
        print(f"FAIL: written frame {written.id} not recalled: {hit_ids}", file=sys.stderr)
        return 1

    extra_status = extras.status()
    if args.expect_core_only:
        if extra_status["llm"] or extra_status["embeddings"] or extra_status["postgres"]:
            print(f"FAIL: expected a core-only install, got {extra_status}", file=sys.stderr)
            return 1
        try:
            MiniLMEmbedder("sentence-transformers/all-MiniLM-L6-v2", dim=dim).embed("x")
        except extras.MissingExtraError:
            pass
        else:
            print("FAIL: MiniLM embedded without the embeddings extra", file=sys.stderr)
            return 1

    print(
        json.dumps(
            {
                "gluomem_version": gluomem.__version__,
                "package_dir": str(package_dir),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "extras": extra_status,
                "written_frame_id": written.id,
                "recalled_frame_ids": hit_ids,
                "result": "ok",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
