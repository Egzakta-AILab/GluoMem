"""Backend-portable typed-writer contracts with explicit authored proposals.

These exercise real storage/commit paths; they are not LLM or native-wire scores.
"""

from gluomem.config.app_config import AppConfig
from gluomem.frames.lifecycle import forget_memory, retract_source
from gluomem.frames.semantics import SemanticStore
from gluomem.llm.semantic import AssertionKey, ClaimMeaning, SemanticClaimSpec
from gluomem.search.evidence import evidence_frames
from gluomem.write.events import SourceEvent, SourceEventMessage, ingest_source_event
from tests._source_helpers import AuthoredProjection, SafeScanner, choice

CONTRACT = "semantic-state-writer-v4"


class SemanticProjection(AuthoredProjection):
    writer_capability = CONTRACT


def assertion(text, predicate, value, *, anchor=None):
    key = AssertionKey(subject="speaker", predicate=predicate, perspective="user")
    return SemanticClaimSpec(
        text=text,
        message_id="m",
        verbatim_span=text,
        meaning=ClaimMeaning(
            key=key, value_key=value, independently_versioned=anchor is None, refines=anchor
        ),
    )


def publish(store, event_id, specs, decisions):
    cfg = AppConfig(dspy={"model": "mock"}, _env_file=None, _secrets_dir=None)
    cfg.dspy.cache.disk = False
    cfg.write.writer_contract = CONTRACT
    cfg.write.cognify.enabled = False
    event = SourceEvent(
        event_id=event_id,
        source_ref="authored:" + event_id,
        capture_policy="attributed-interaction",
        created_at="2026-01-02T08:00:00Z",
        messages=[
            SourceEventMessage(
                message_id="m", role="user", content=" ".join(spec.text for spec in specs)
            )
        ],
    )
    result = ingest_source_event(
        event,
        store=store,
        cfg=cfg,
        scanner=SafeScanner(),
        extractors=SemanticProjection(specs, decisions),
    )
    return result


def test_semantic_receipts_publish_revoke_and_preserve_independent_replacement(frame_store):
    code = assertion("My locker code is LIME-427.", "locker-code", "LIME-427")
    name = assertion("My name is Iona.", "name", "Iona")
    original = publish(
        frame_store, "semantic-backend-original", [code, name], [choice("ADD"), choice("ADD")]
    )
    old, sibling = original.frames
    bindings = SemanticStore(frame_store)
    old_binding = bindings.snapshot(old.id, scope=frame_store.scope)
    sibling_binding = bindings.snapshot(sibling.id, scope=frame_store.scope)
    assert old_binding is not None and old_binding.meaning == code.meaning
    assert sibling_binding is not None and sibling_binding.meaning == name.meaning
    replacement_spec = assertion("My locker code is AMBER-693.", "locker-code", "AMBER-693")
    changed = publish(
        frame_store,
        "semantic-backend-correction",
        [replacement_spec],
        [choice("SUPERSEDE", frame_id=old.id)],
    )
    replacement = changed.frames[0]
    sibling_proof = evidence_frames(
        frame_store,
        [frame_store.get_frame(sibling.id)],
        frame_store.scope,
    )[0].provenance[0]
    assert sibling_proof.span_verified and sibling_proof.verification_basis == "retained-fragment"
    assert bindings.snapshot(sibling.id, scope=frame_store.scope) is None
    replacement_binding = bindings.snapshot(replacement.id, scope=frame_store.scope)
    assert (
        replacement_binding is not None and replacement_binding.meaning == replacement_spec.meaning
    )
    note = assertion(
        "The current locker has a copper door.",
        "locker-note",
        "copper",
        anchor=replacement_spec.meaning.key,
    )
    detail = publish(
        frame_store, "semantic-backend-detail", [note], [choice("REFINE", frame_id=replacement.id)]
    ).frames[0]
    assert detail.frame_type == "P" and detail.base_frame_id == replacement.id
    detail_binding = bindings.snapshot(detail.id, scope=frame_store.scope)
    assert detail_binding is not None and detail_binding.meaning == note.meaning
    result = retract_source(frame_store, "authored:semantic-backend-original")
    assert replacement.id not in result.forgotten_frame_ids
    assert frame_store.get_frame(replacement.id).status == "active"
    assert frame_store.get_frame(detail.id).status == "active"
    assert bindings.snapshot(replacement.id, scope=frame_store.scope) is not None
    forget_memory(frame_store, replacement.id)
    assert bindings.snapshot(replacement.id, scope=frame_store.scope) is None
    assert bindings.snapshot(detail.id, scope=frame_store.scope) is None
    assert frame_store.get_frame(detail.id).status == "forgotten"


def test_semantic_noop_retains_multiple_independent_receipts(frame_store):
    spec = assertion("My name is Iona.", "name", "Iona")
    first = publish(frame_store, "semantic-backend-first", [spec], [choice("ADD")]).frames[0]
    second = publish(
        frame_store, "semantic-backend-second", [spec], [choice("NOOP", frame_id=first.id)]
    ).frames[0]
    assert first.id == second.id
    bindings = SemanticStore(frame_store)
    receipts = bindings.snapshot(first.id, scope=frame_store.scope)
    assert receipts is not None and len(receipts.receipts) == 2
    result = retract_source(frame_store, "authored:semantic-backend-first")
    assert first.id in result.retained_frame_ids
    remaining = bindings.snapshot(first.id, scope=frame_store.scope)
    assert remaining is not None and len(remaining.receipts) == 1
    assert remaining.receipts[0].witness.source_ref == "authored:semantic-backend-second"
    retract_source(frame_store, "authored:semantic-backend-second")
    assert bindings.snapshot(first.id, scope=frame_store.scope) is None
    assert frame_store.get_frame(first.id).status == "forgotten"


def test_semantic_correction_accepts_nullable_legacy_provenance(frame_store):
    old = assertion("My locker code is LIME-427.", "locker-code", "LIME-427")
    target = publish(frame_store, "semantic-legacy-original", [old], [choice("ADD")]).frames[0]
    # Supported legacy/importer provenance lacks an envelope; it neither grants
    # semantic authority nor invalidates the target's real typed source receipt.
    frame_store.db.execute(
        "INSERT INTO memory_provenance "
        "(tenant_id,principal_id,namespace,frame_id,source_type,source_ref,verbatim_span) "
        "VALUES (?,?,?,?,?,?,?)",
        (*frame_store.scope.as_tuple(), target.id, "document", "authored:legacy", old.text),
    )
    frame_store.db.commit()
    assert (
        frame_store.db.scalar(
            "SELECT COUNT(*) FROM memory_provenance WHERE frame_id=? AND source_event_id IS NULL",
            (target.id,),
        )
        == 1
    )
    old_binding = SemanticStore(frame_store).snapshot(target.id)
    assert old_binding is not None and old_binding.meaning == old.meaning
    replacement_spec = assertion("My locker code is AMBER-693.", "locker-code", "AMBER-693")
    try:
        changed = publish(
            frame_store,
            "semantic-legacy-correction",
            [replacement_spec],
            [choice("SUPERSEDE", frame_id=target.id)],
        )
    except TypeError as exc:
        raise AssertionError(
            "Nullable legacy provenance broke source-revocation projection"
        ) from exc
    replacement = changed.frames[0]
    assert frame_store.get_frame(target.id).status == "superseded"
    assert frame_store.get_frame(target.id).content == "[forgotten]"
    assert replacement.status == "active" and replacement.supersedes == target.id
    assert SemanticStore(frame_store).snapshot(target.id) is None
    replacement_binding = SemanticStore(frame_store).snapshot(replacement.id)
    assert (
        replacement_binding is not None and replacement_binding.meaning == replacement_spec.meaning
    )
