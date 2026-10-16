"""Thin provider with a host-independent directory installer."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .provider import GluoMemProvider


def __getattr__(name: str):
    if name != "GluoMemProvider":
        raise AttributeError(name)
    from .provider import GluoMemProvider

    return GluoMemProvider


def register(ctx):
    from .provider import GluoMemProvider

    ctx.register_memory_provider(GluoMemProvider())


__all__ = ["GluoMemProvider", "register"]
