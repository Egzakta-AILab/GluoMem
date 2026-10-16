"""Bounded request models and conversion to canonical source events."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gluomem.client.models import MemoryKind, SourceType
from gluomem.storage.scope import MemoryScope
from gluomem.timeutil import parse_utc
from gluomem.write.events import SourceEvent, SourceEventMessage


class _Req(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IngestEventRequest(_Req):
    event_id: str = Field(min_length=1, max_length=256)
    revision: int = Field(default=1, ge=1)
    source_type: Annotated[
        SourceType,
        Field(
            description=(
                "Host-authenticated provenance. explicit_memory is reserved for authentic user "
                "confirmation of the exact content; never pass an agent-selected source type."
            ),
        ),
    ] = "chat_turn"
    source_ref: str = Field(min_length=1, max_length=1024)
    messages: list[SourceEventMessage] = Field(default_factory=list, max_length=100)
    content: str | None = Field(default=None, max_length=100_000)
    kind: MemoryKind | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    require_review: bool = False
    pinned: bool = False
    event_time: str | None = Field(default=None, max_length=64)
    created_at: str | None = Field(default=None, max_length=64)
    force_reassert: bool = False
    async_mode: bool | None = None

    @field_validator("event_time", "created_at")
    @classmethod
    def valid_time(cls, value: str | None) -> str | None:
        if value is not None:
            parse_utc(value)
        return value

    @field_validator("messages")
    @classmethod
    def bounded_messages(cls, values: list[SourceEventMessage]) -> list[SourceEventMessage]:
        for message in values:
            if len(message.content) > 100_000 or len(message.message_id) > 256:
                raise ValueError("message content or id exceeds limit")
            if len(message.role) > 64:
                raise ValueError("message role exceeds limit")
            if message.timestamp is not None:
                if len(message.timestamp) > 64:
                    raise ValueError("message timestamp exceeds limit")
                parse_utc(message.timestamp)
        return values

    @model_validator(mode="after")
    def bounded_aggregate_content(self) -> IngestEventRequest:
        """Limit combined content plus all message bodies to 100,000 UTF-8 bytes.

        This bounds the extraction input, not the raw HTTP body; deployments must
        separately limit request bytes at their trusted reverse proxy.
        """
        total = len((self.content or "").encode("utf-8"))
        total += sum(len(message.content.encode("utf-8")) for message in self.messages)
        if total > 100_000:
            raise ValueError("aggregate content exceeds 100000 UTF-8 bytes")
        return self

    def to_source_event(self, scope: MemoryScope) -> SourceEvent:
        return SourceEvent.model_validate(
            {**self.model_dump(exclude={"async_mode"}), "scope": scope}
        )


class RecallRequest(_Req):
    query: str = Field(min_length=1, max_length=4096)
    k: int = Field(default=10, ge=1, le=100)
    profile: str = Field(default="balanced", max_length=128)
    since: str | None = Field(default=None, max_length=64)
    until: str | None = Field(default=None, max_length=64)
    now_iso: str | None = Field(default=None, max_length=64)

    @field_validator("since", "until", "now_iso")
    @classmethod
    def valid_time(cls, value: str | None) -> str | None:
        if value is not None:
            parse_utc(value)
        return value


class ReviewRequest(_Req):
    action: Literal["accept", "edit", "dismiss"]
    edited_content: str | None = Field(default=None, max_length=100_000)


class CorrectRequest(_Req):
    new_content: str = Field(min_length=1, max_length=100_000)


class RetractSourceRequest(_Req):
    source_ref: str = Field(min_length=1, max_length=1024)


class SweepExpiredRequest(_Req):
    now_iso: str | None = Field(default=None, max_length=64)
    dedup_key: str | None = Field(default=None, min_length=1, max_length=256)
    async_mode: bool = False

    @field_validator("now_iso")
    @classmethod
    def valid_time(cls, value: str | None) -> str | None:
        if value is not None:
            parse_utc(value)
        return value
