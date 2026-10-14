"""Service authentication and header-derived ``MemoryScope`` resolution."""

from __future__ import annotations

import hmac
import os
import re

from fastapi import HTTPException

from gluomem.client.models import HEADER_NAMESPACE as HEADER_NAMESPACE
from gluomem.client.models import HEADER_PRINCIPAL as HEADER_PRINCIPAL
from gluomem.client.models import HEADER_TENANT as HEADER_TENANT
from gluomem.extras import require
from gluomem.storage.scope import MemoryScope

require("service", "GluoMem HTTP service")
_SCOPE_HEADER = re.compile(r"[A-Za-z0-9._:@-]{1,128}\Z")


def resolve_service_key(explicit_key: str | None = None) -> str:
    """Resolve the shared service bearer secret from explicit value or ``GLUOMEM_SERVICE_KEY``."""
    key = explicit_key if explicit_key is not None else os.environ.get("GLUOMEM_SERVICE_KEY", "")
    cleaned = key.strip()
    if not cleaned:
        raise ValueError(
            "GluoMem service requires a non-empty service key "
            "(pass service_key=... or set GLUOMEM_SERVICE_KEY)"
        )
    return cleaned


def verify_bearer_header(authorization: str | None, expected_key: str) -> None:
    """Verify ``Authorization: Bearer <key>`` using constant-time comparison."""
    if not authorization or not authorization.strip():
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(
            status_code=401,
            detail="Authorization header must be 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = parts[1].strip()
    if not hmac.compare_digest(token.encode("utf-8"), expected_key.encode("utf-8")):
        raise HTTPException(
            status_code=401,
            detail="Invalid service bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def scope_from_headers(
    tenant: str | None,
    principal: str | None,
    namespace: str | None = None,
) -> MemoryScope:
    """Construct a validated ``MemoryScope`` from ``X-GluoMem-*`` request headers."""
    if not tenant or not tenant.strip():
        raise HTTPException(
            status_code=422,
            detail=f"Missing required header {HEADER_TENANT}",
        )
    if not principal or not principal.strip():
        raise HTTPException(
            status_code=422,
            detail=f"Missing required header {HEADER_PRINCIPAL}",
        )
    eff_ns = namespace if namespace is not None else "personal"
    if any(not _SCOPE_HEADER.fullmatch(value) for value in (tenant, principal, eff_ns)):
        raise HTTPException(status_code=422, detail="Invalid scope header")
    try:
        return MemoryScope(
            tenant_id=tenant.strip(),
            principal_id=principal.strip(),
            namespace=eff_ns,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
