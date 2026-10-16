"""Offline operator-owned reconciliation; never infer later deletions from an old backup."""

import json
from pathlib import Path
from typing import Any

from gluomem.frames.store import FrameStore


def reconcile_restored_store(store: FrameStore, journal_path: Path | str) -> dict[str, Any]:
    """Keep the quarantined target blocked until the external journal is replayed."""
    from gluomem.frames.lifecycle import forget_scope, retract_source
    from gluomem.service.jobs import write_transaction
    from gluomem.service.readiness import WORKER_HEARTBEAT_KEY
    from gluomem.storage.scope import MemoryScope

    if store.db.in_transaction:
        raise ValueError("restore reconciliation requires an idle offline target")
    store.db.meta_set("operations.restore_state", "blocked")
    journal = json.loads(Path(journal_path).read_text(encoding="utf-8"))
    if (
        not isinstance(journal, dict)
        or journal.get("version") != 1
        or journal.get("complete") is not True
        or not isinstance(journal.get("actions"), list)
    ):
        raise ValueError("unsupported or incomplete deletion journal")
    actions = []
    for action in journal["actions"]:
        if (
            not isinstance(action, dict)
            or action.get("operation") not in ("retract_source", "forget_scope")
            or not isinstance(action.get("scope"), list)
            or len(action["scope"]) != 3
            or (
                action["operation"] == "retract_source"
                and (
                    not isinstance(action.get("source_ref"), str)
                    or not action["source_ref"].strip()
                )
            )
        ):
            raise ValueError("unsupported deletion action; target remains blocked")
        scope = MemoryScope(*action["scope"])
        actions.append((scope, action))
    with write_transaction(store.db.backend):
        for scope, action in actions:
            if action["operation"] == "retract_source":
                retract_source(store, action["source_ref"], scope=scope)
            else:
                forget_scope(store, scope=scope)
        store.db.execute("DELETE FROM meta WHERE key = ?", (WORKER_HEARTBEAT_KEY,))
        store.db.meta_set("operations.restore_state", "reconciled", commit=False)
    return {"actions_applied": len(actions), "state": "reconciled"}
