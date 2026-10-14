"""GOP compaction: fold older P-frames into their base I-frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gluomem.config.app_config import AppConfig
from gluomem.frames.content_hash import content_hash, suppression_key
from gluomem.frames.eligibility import can_merge_into
from gluomem.storage.scope import scope_sql

if TYPE_CHECKING:
    from gluomem.frames.store import FrameStore
    from gluomem.storage.scope import MemoryScope


def compact(
    store: FrameStore,
    gop: str,
    cfg: AppConfig,
    *,
    scope: MemoryScope | None = None,
) -> None:
    """Compact a GOP's P-frames into their base I-frame when count exceeds trigger."""
    eff_scope = store._eff_scope(scope)
    comp = cfg.write.compaction
    trigger = int(comp.p_frame_trigger)
    keep = int(comp.keep_recent)
    sep = comp.separator
    if keep < 0:
        raise ValueError(f"write.compaction.keep_recent must be non-negative, got {keep}")

    backend = store.db.backend
    rows = backend.fetchall(
        "SELECT id, base_frame_id FROM memory_frames "
        f"WHERE {scope_sql()} "
        "AND gop_id = ? AND frame_type = 'P' AND base_frame_id IS NOT NULL "
        "ORDER BY id ASC",
        (
            *eff_scope.as_tuple(),
            gop,
        ),
    )
    by_base: dict[int, list[int]] = {}
    for pid, bid in rows:
        if bid is not None:
            by_base.setdefault(int(bid), []).append(int(pid))

    for base_id, p_ids in by_base.items():
        if len(p_ids) <= trigger:
            continue
        try:
            base = store.get_frame(base_id, scope=eff_scope)
        except KeyError:
            continue

        compatible = [
            p
            for p in (store.get_frame(pid, scope=eff_scope) for pid in p_ids)
            if can_merge_into(base, p) and not store.frames_based_on(p.id, scope=eff_scope)
        ]
        if len(compatible) <= trigger:
            continue

        older_frames = compatible[: len(compatible) - keep] if keep > 0 else compatible
        if not older_frames:
            continue

        older_bodies = [p.content for p in older_frames]
        merged_content = base.content + sep + sep.join(older_bodies)
        prepared = store._prepare_index(merged_content)
        new_hash = content_hash(merged_content)
        old_hash = base.content_hash

        with store.db.transaction():
            store._update(
                base_id,
                eff_scope,
                content=merged_content,
                content_hash=new_hash,
                suppression_key=suppression_key(merged_content),
            )
            store._write_index(base_id, prepared)
            store._record_lineage(base_id, base, old_hash, "compaction_base")
            for p in older_frames:
                store._record_lineage(base_id, p, content_hash(p.content), "compaction_merge")
                for table, column in (
                    ("kg_entity_frames", "entity_id"),
                    ("kg_relation_frames", "relation_id"),
                ):
                    backend.execute(
                        f"INSERT INTO {table}({column}, frame_id) SELECT {column}, ? "
                        f"FROM {table} WHERE frame_id = ? "
                        f"ON CONFLICT ({column}, frame_id) DO NOTHING",
                        (base_id, p.id),
                    )
                backend.execute(
                    "UPDATE memory_provenance SET frame_id = ? "
                    f"WHERE frame_id = ? AND {scope_sql()}",
                    (base_id, p.id, *eff_scope.as_tuple()),
                )
                store._delete_index_rows(p.id, commit=False)
                backend.execute("DELETE FROM memory_frames WHERE id = ?", (p.id,))
