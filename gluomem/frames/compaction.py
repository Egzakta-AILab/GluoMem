"""GOP compaction: fold older P-frames into their base I-frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gluomem.config.app_config import AppConfig
from gluomem.frames.content_hash import content_hash
from gluomem.frames.eligibility import can_merge_into

if TYPE_CHECKING:
    from gluomem.frames.store import FrameStore


def compact(store: FrameStore, gop: str, cfg: AppConfig) -> None:
    """Compact a GOP's P-frames into their base I-frame when count exceeds trigger."""
    comp = cfg.write.compaction
    trigger = int(comp.p_frame_trigger)
    keep = int(comp.keep_recent)
    sep = comp.separator
    if keep < 0:
        raise ValueError(
            f"write.compaction.keep_recent must be non-negative, got {keep}"
        )

    backend = store.db.backend
    rows = backend.fetchall(
        "SELECT id, base_frame_id FROM memory_frames "
        "WHERE gop_id = ? AND frame_type = 'P' AND base_frame_id IS NOT NULL "
        "ORDER BY id ASC",
        (gop,),
    )
    by_base: dict[int, list[int]] = {}
    for pid, bid in rows:
        if bid is not None:
            by_base.setdefault(int(bid), []).append(int(pid))

    for base_id, p_ids in by_base.items():
        if len(p_ids) <= trigger:
            continue
        try:
            base = store.get_frame(base_id)
        except KeyError:
            continue

        compatible = [
            p
            for p in (store.get_frame(pid) for pid in p_ids)
            if can_merge_into(base, p) and not store.frames_based_on(p.id)
        ]
        if len(compatible) <= trigger:
            continue

        older_frames = (
            compatible[: len(compatible) - keep] if keep > 0 else compatible
        )
        if not older_frames:
            continue

        older_bodies = [p.content for p in older_frames]
        merged_content = base.content + sep + sep.join(older_bodies)
        prepared = store._prepare_index(merged_content)
        new_hash = content_hash(merged_content)
        old_hash = store._content_hash_of(base_id)

        with store.db.transaction():
            backend.execute(
                "UPDATE memory_frames SET content = ?, content_hash = ? WHERE id = ?",
                (merged_content, new_hash, base_id),
            )
            store._write_index(base_id, prepared)
            store._record_lineage(base_id, base, old_hash, "compaction_base")
            for p in older_frames:
                store._record_lineage(
                    base_id, p, content_hash(p.content), "compaction_merge"
                )
                if backend.dialect == "postgres":
                    backend.execute(
                        "INSERT INTO kg_entity_frames(entity_id, frame_id) "
                        "SELECT entity_id, ? FROM kg_entity_frames WHERE frame_id = ? "
                        "ON CONFLICT (entity_id, frame_id) DO NOTHING",
                        (base_id, p.id),
                    )
                else:
                    backend.execute(
                        "INSERT OR IGNORE INTO kg_entity_frames(entity_id, frame_id) "
                        "SELECT entity_id, ? FROM kg_entity_frames WHERE frame_id = ?",
                        (base_id, p.id),
                    )
                store._delete_index_rows(p.id, commit=False)
                backend.execute("DELETE FROM memory_frames WHERE id = ?", (p.id,))
