"""
Rate Limiting Middleware Tests (roadmap Phase 23)

Uses httpx's ASGI transport for real request/response dispatch against a
minimal in-process app — same approach as tests/test_concurrency_middleware.py
and for the same reason: this is fundamentally about HTTP mechanics (CORS
headers surviving a 429, distinct clients getting independent buckets),
not pure Python logic.
"""

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import apps.api.middleware.rate_limit as rate_limit_module
from apps.api.middleware.rate_limit import rate_limit_middleware
from config import settings


def _build_test_app() -> FastAPI:
    """Mirrors apps/api/main.py's registration order: rate limiter innermost, CORS wraps it."""
    app = FastAPI()
    app.middleware("http")(rate_limit_middleware)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_credentials=True)

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.post("/api/query")
    async def query():
        return {"ok": True}

    return app


@pytest.fixture(autouse=True)
def _reset_buckets(monkeypatch):
    monkeypatch.setattr(rate_limit_module, "_buckets", {})


class TestRateLimitMiddleware:
    @pytest.mark.asyncio
    async def test_requests_within_limit_are_allowed(self, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 5)
        app = _build_test_app()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for _ in range(5):
                response = await client.post("/api/query")
                assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_requests_past_limit_get_429(self, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 3)
        app = _build_test_app()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            statuses = [(await client.post("/api/query")).status_code for _ in range(5)]

        assert statuses == [200, 200, 200, 429, 429]

    @pytest.mark.asyncio
    async def test_429_still_carries_cors_headers(self, monkeypatch):
        """Same real bug class as the concurrency ceiling — registration order must keep CORS outermost."""
        monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 1)
        app = _build_test_app()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/api/query", headers={"Origin": "http://localhost:3000"})
            second = await client.post("/api/query", headers={"Origin": "http://localhost:3000"})

        assert second.status_code == 429
        assert second.headers.get("access-control-allow-origin") == "http://localhost:3000"

    @pytest.mark.asyncio
    async def test_distinct_api_keys_get_independent_buckets(self, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 1)
        app = _build_test_app()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r1 = await client.post("/api/query", headers={"X-API-Key": "client-a"})
            r2 = await client.post("/api/query", headers={"X-API-Key": "client-b"})
            r3 = await client.post("/api/query", headers={"X-API-Key": "client-a"})  # client-a's second request

        assert r1.status_code == 200
        assert r2.status_code == 200  # different key, independent bucket
        assert r3.status_code == 429  # client-a already used its one token

    @pytest.mark.asyncio
    async def test_non_api_paths_bypass_rate_limiting(self, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 1)
        app = _build_test_app()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/api/query")  # consume the one token
            responses = [await client.get("/health") for _ in range(5)]

        assert all(r.status_code == 200 for r in responses)

    @pytest.mark.asyncio
    async def test_disabled_when_zero(self, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 0)
        app = _build_test_app()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            responses = [await client.post("/api/query") for _ in range(20)]

        assert all(r.status_code == 200 for r in responses)

    @pytest.mark.asyncio
    async def test_bucket_refills_over_time(self, monkeypatch):
        monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 60)  # 1 token/second
        app = _build_test_app()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = await client.post("/api/query")
            await asyncio.sleep(1.1)  # should refill roughly one more token
            second = await client.post("/api/query")

        assert first.status_code == 200
        assert second.status_code == 200
