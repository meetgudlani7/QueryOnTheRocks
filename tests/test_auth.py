"""
API Key Auth Tests (roadmap Phase 23)

require_api_key is a plain async function taking the header value FastAPI
would have already extracted — tested directly, no HTTP/TestClient needed.
"""

import pytest
from fastapi import HTTPException

from apps.api.dependencies import require_api_key
from config import settings


@pytest.fixture(autouse=True)
def _reset_api_keys(monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", "")


class TestRequireApiKey:
    @pytest.mark.asyncio
    async def test_disabled_by_default_allows_any_request(self):
        """Empty API_KEYS (the default) must be a true no-op — no header required at all."""
        await require_api_key(x_api_key=None)  # must not raise

    @pytest.mark.asyncio
    async def test_missing_key_rejected_when_configured(self, monkeypatch):
        monkeypatch.setattr(settings, "API_KEYS", "secret123")
        with pytest.raises(HTTPException) as exc_info:
            await require_api_key(x_api_key=None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_key_rejected(self, monkeypatch):
        monkeypatch.setattr(settings, "API_KEYS", "secret123")
        with pytest.raises(HTTPException) as exc_info:
            await require_api_key(x_api_key="wrongkey")
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_correct_key_allowed(self, monkeypatch):
        monkeypatch.setattr(settings, "API_KEYS", "secret123")
        await require_api_key(x_api_key="secret123")  # must not raise

    @pytest.mark.asyncio
    async def test_multiple_keys_any_one_valid(self, monkeypatch):
        """Comma-separated list — each issued key must work independently."""
        monkeypatch.setattr(settings, "API_KEYS", "key-a, key-b ,key-c")
        await require_api_key(x_api_key="key-a")
        await require_api_key(x_api_key="key-b")
        await require_api_key(x_api_key="key-c")
        with pytest.raises(HTTPException):
            await require_api_key(x_api_key="key-d")
