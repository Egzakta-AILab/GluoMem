"""Benchmark decay-anchor backdating — write.backdate_created_at.

Without backdating, ingesting a 2023 benchmark today makes every memory
identically fresh and the temporal scoring signal pure noise.
"""

from __future__ import annotations

from pathlib import Path

from gluomem.llm import FakeExtractors
from gluomem.write.gate import InjectionScanner, admit
from gluomem.write.ingest import ingest_conversation, load_locomo

FIXTURE = Path(__file__).parent.parent / "fixtures" / "locomo_tiny.json"


def test_store_backdates_created_at_when_given(env) -> None:
    frame = env.store.create_i_frame(
        "g1", "old fact", created_at="2023-05-08T13:56:00"
    )
    assert frame.created_at == "2023-05-08T13:56:00"
    # Default stays schema-now.
    fresh = env.store.create_i_frame("g1", "new fact")
    assert fresh.created_at.startswith("20")
    assert fresh.created_at != "2023-05-08T13:56:00"


def test_gate_passes_created_at_for_all_frame_types(env) -> None:
    scanner = InjectionScanner(env.cfg.safety)
    kwargs = dict(
        gop="g1", scanner=scanner, store=env.store, kg=env.kg, cfg=env.cfg,
        created_at="2023-05-08T13:56:00",
    )
    base = admit("base", **kwargs)
    assert base.frame is not None
    base_id = base.frame.id
    p = admit("delta", frame_type="P", base_id=base_id, **kwargs)
    b = admit(
        "corr", frame_type="B", base_id=base_id, references=[base_id], **kwargs
    )
    for result in (base, p, b):
        assert result.frame is not None
        assert result.frame.created_at == "2023-05-08T13:56:00"


def _ingest(env, cfg):
    conv = load_locomo(FIXTURE)[0]
    return ingest_conversation(
        conv,
        facade=FakeExtractors(banner=False),
        scanner=InjectionScanner(cfg.safety),
        store=env.store,
        kg=env.kg,
        cfg=cfg,
    )


def test_backdated_ingest_anchors_decay_on_session_time(env) -> None:
    cfg = env.cfg.with_overrides({"write.backdate_created_at": True})
    _ingest(env, cfg)
    rows = env.db.conn.execute(
        "SELECT DISTINCT created_at FROM memory_frames "
        "WHERE gop_id LIKE '%:session:%'"
    ).fetchall()
    assert {r[0] for r in rows} == {
        "2023-05-08T13:56:00",
        "2023-06-21T10:15:00",
        "2023-08-02T19:30:00",
    }
    # Lane frames carry their session's datetime too.
    fact_rows = env.db.conn.execute(
        "SELECT DISTINCT created_at FROM memory_frames WHERE gop_id LIKE '%:facts'"
    ).fetchall()
    assert {r[0] for r in fact_rows} <= {
        "2023-05-08T13:56:00",
        "2023-06-21T10:15:00",
        "2023-08-02T19:30:00",
    }


def test_default_ingest_keeps_ingest_time(env) -> None:
    _ingest(env, env.cfg)
    rows = env.db.conn.execute(
        "SELECT DISTINCT created_at FROM memory_frames"
    ).fetchall()
    assert all(not r[0].startswith("2023-") for r in rows)
