"""Service authentication and header-derived ``MemoryScope`` resolution."""

from __future__ import annotations

import hmac
import os

from gluomem.extras import require
from gluomem.storage.scope import MemoryScope

HEADER_TENANT = "X-GluoMem-Tenant"
HEADER_PRINCIPAL = "X-GluoMem-Principal"
HEADER_NAMESPACE = "X-GluoMem-Namespace"


def resolve_service_key(explicit_key: str | None = None) -> str:
    """Resolve the shared service bearer secret from explicit value or ``GLUOMEM_SERVICE_KEY``."""
    key = (
        explicit_key
        if explicit_key is not None
        else os.environ.get("GLUOMEM_SERVICE_KEY", "")
    )
    cleaned = key.strip()
    if not cleaned:
        raise ValueError(
            "GluoMem service requires a non-empty service key "
            "(pass service_key=... or set GLUOMEM_SERVICE_KEY)"
        )
    return cleaned


def verify_bearer_header(authorization: str | None, expected_key: str) -> None:
    """Verify ``Authorization: Bearer <key>`` using constant-time comparison."""
    require("service", "GluoMem HTTP service")
    from fastapi import HTTPException

    if not authorization or not authorization.strip():
        raise HTTPException(
            status_code=401, detail="Missing Authorization header"
        )
    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(
            status_code=401, detail="Authorization header must be 'Bearer <token>'"
        )
    token = parts[1].strip()
    if not hmac.compare_digest(
        token.encode("utf-8"), expected_key.encode("utf-8")
    ):
        raise HTTPException(status_code=403, detail="Invalid service bearer token")


def scope_from_headers(
    tenant: str | None,
    principal: str | None,
    namespace: str | None = None,
) -> MemoryScope:
    """Construct a validated ``MemoryScope`` from ``X-GluoMem-*`` request headers."""
    require("service", "GluoMem HTTP service")
    from fastapi import HTTPException

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
    eff_ns = namespace.strip() if namespace and namespace.strip() else "personal"
    try:
        return MemoryScope(
            tenant_id=tenant.strip(),
            principal_id=principal.strip(),
            namespace=eff_ns,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
