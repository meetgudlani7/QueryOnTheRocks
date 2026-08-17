"""
API Key Authentication (roadmap Phase 23)

Opt-in: API_KEYS empty (the default) disables auth entirely — nothing
about existing local dev, tests, or scripts changes unless keys are
actually configured. When configured, requests to protected routes must
carry a valid key in the X-API-Key header.
"""

from typing import Optional, Set

from fastapi import Header, HTTPException

from config import settings


def _configured_keys() -> Set[str]:
    return {k.strip() for k in settings.API_KEYS.split(",") if k.strip()}


async def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    FastAPI dependency — raises 401 if API_KEYS is configured and the
    request's X-API-Key header doesn't match one of them. A no-op when
    API_KEYS is unset (today's behavior, unchanged).
    """
    keys = _configured_keys()
    if not keys:
        return
    if not x_api_key or x_api_key not in keys:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
