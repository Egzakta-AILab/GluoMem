"""Configured episode expiration."""

from datetime import timedelta

from gluomem.config.app_config import LifecycleConfig
from gluomem.timeutil import parse_utc, to_storage


def episode_expiry(anchor: str, cfg: LifecycleConfig) -> str:
    """Return the episode expiry anchored at ``anchor`` in UTC."""
    expiry = to_storage(parse_utc(anchor) + timedelta(days=cfg.episode_ttl_days))
    assert expiry is not None
    return expiry
