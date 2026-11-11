"""Bind extracted claims to the identities supplied by a trusted source envelope."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from unicodedata import category

from gluomem.storage.scope import FrameSource

if TYPE_CHECKING:
    from gluomem.write.events import SourceEvent

CapturePolicy = Literal["trusted-only", "attributed-interaction"]
ACTIVE_SOURCE_ENVELOPE: ContextVar[dict | None] = ContextVar("source_aligned_writer", default=None)

_ROLE_SOURCE: dict[str, FrameSource] = {
    "user": "user_stated",
    "system": "system",
    "tool": "tool_verified",
    "assistant": "agent_inferred",
}


@dataclass(frozen=True, slots=True)
class SourceBinding:
    """Trusted attribution; no model-supplied role or learned-date authority."""

    message_id: str | None
    role: str
    frame_source: FrameSource
    observed_at: str | None
    verbatim_span: str


def validate_quote_alignment(source: str, quote: str) -> None:
    """Require one exact occurrence not cutting a lexical unit; NOT entailment.

    Unicode letters/numbers/marks/underscore form units, with interior code/name
    hyphens and apostrophes. No minimum length, trimming, expansion or fallback.
    """

    def word(index):
        return 0 <= index < len(source) and (
            source[index] == "_" or category(source[index])[0] in "LNM"
        )

    def unit(index):
        return word(index) or (
            0 <= index < len(source)
            and source[index] in "-'’"
            and word(index - 1)
            and word(index + 1)
        )

    start = source.find(quote)
    while start >= 0 and quote.strip():
        end = start + len(quote)
        if not (unit(start - 1) and unit(start)) and not (unit(end - 1) and unit(end)):
            return
        start = source.find(quote, start + 1)
    raise ValueError("claim supporting span must align with source lexical boundaries")


def bind_source_claim(
    event: SourceEvent,
    *,
    message_id: str | None,
    quote: str,
    capture_policy: CapturePolicy = "trusted-only",
    writer_contract: str | None = None,
) -> SourceBinding:
    """Resolve a source identity without silently replacing an unsupported ID."""
    from gluomem.llm.outputs import SUPPORTED_WRITER_CONTRACTS

    if writer_contract is not None and writer_contract not in SUPPORTED_WRITER_CONTRACTS:
        raise ValueError("unknown writer contract")
    if capture_policy not in ("trusted-only", "attributed-interaction"):
        raise ValueError("unknown source capture policy")
    if not event.messages:
        from gluomem.write.events import _SOURCE_TYPE_TO_FRAME_SOURCE

        if message_id is not None:
            raise ValueError("document claim must not invent a source message")
        if not quote.strip() or event.content is None or quote not in event.content:
            raise ValueError("claim supporting span must occur literally in its source document")
        if writer_contract:
            validate_quote_alignment(event.content, quote)
        return SourceBinding(
            None,
            event.source_type,
            _SOURCE_TYPE_TO_FRAME_SOURCE[event.source_type],
            event.created_at,
            quote,
        )
    messages = {message.message_id: message for message in event.messages}
    if len(messages) != len(event.messages) or any(not mid.strip() for mid in messages):
        raise ValueError("source message identities must be unique and nonempty")
    if message_id not in messages:
        raise ValueError("claim must reference a supplied source message")
    message = messages[message_id]
    if message.role not in _ROLE_SOURCE or (
        message.role == "assistant" and capture_policy != "attributed-interaction"
    ):
        raise ValueError("source role is not permitted by the capture policy")
    if not quote.strip() or quote not in message.content:
        raise ValueError("claim supporting span must occur literally in its source message")
    if writer_contract:
        validate_quote_alignment(message.content, quote)
    return SourceBinding(
        message.message_id,
        message.role,
        _ROLE_SOURCE[message.role],
        message.timestamp or event.created_at,
        quote,
    )
