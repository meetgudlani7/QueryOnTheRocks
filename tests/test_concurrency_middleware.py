"""
Concurrency Ceiling Middleware Tests

Uses httpx's ASGI transport to drive real concurrent requests directly
against a minimal in-process test app — genuine async dispatch (unlike a
thread-based TestClient), no actual server/socket needed. This is the one
Phase 22 test file that reaches for an HTTP-level test rather than pure
Python logic, because the thing being verified — CORS headers correctly
propagating across middleware layers, including on a 503 this middleware
generates itself — is fundamentally about HTTP mechanics that a
unit-level test of the function alone can't meaningfully exercise.
"""

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import apps.api.middleware.concurrency as concurrency_module
from apps.api.middleware.concurrency import concurrency_limit_middleware
from config import settings


def _build_test_app(slow_delay_s: float = 0.2) -> FastAPI:
    """
    Mirrors apps/api/main.py's registration order exactly: concurrency
    middleware registered first (innermost), CORS registered after
    (outermost, wraps it) — see that file's comment for why this order
    isn't arbitrary.
    """
    app = FastAPI()
    app.middleware("http")(concurrency_limit_middleware)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_credentials=True)

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.post("/api/slow")
    async def slow():
        await asyncio.sleep(slow_delay_s)
        return {"ok": True}

    return app


@pytest.fixture(autouse=True)
def _reset_semaphore(monkeypatch):
    """Each test gets a fresh semaphore, sized from whatever it sets MAX_CONCURRENT_REQUESTS to."""
    monkeypatch.setattr(concurrency_module, "_semaphore", None)


class TestConcurrencyCeiling:
    @pytest.mark.asyncio
    async def test_admits_requests_within_capacity(self, monkeypatch):
        monkeypatch.setattr(settings, "MAX_CONCURRENT_REQUESTS", 5)
        monkeypatch.setattr(settings, "CONCURRENCY_QUEUE_TIMEOUT_MS", 1000)
        app = _build_test_app(slow_delay_s=0.05)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            responses = await asyncio.gather(*(client.post("/api/slow") for _ in range(3)))
        assert all(r.status_code == 200 for r in responses)

    @pytest.mark.asyncio
    async def test_rejects_with_503_past_capacity(self, monkeypatch):
        monkeypatch.setattr(settings, "MAX_CONCURRENT_REQUESTS", 1)
        monkeypatch.setattr(settings, "CONCURRENCY_QUEUE_TIMEOUT_MS", 100)
        app = _build_test_app(slow_delay_s=0.3)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            responses = await asyncio.gather(*(client.post("/api/slow") for _ in range(4)))

        statuses = sorted(r.status_code for r in responses)
        assert statuses.count(200) == 1
        assert statuses.count(503) == 3

    @pytest.mark.asyncio
    async def test_503_still_carries_cors_headers(self, monkeypatch):
        """
        Regression test for the exact bug found in live testing:
        registering this middleware *after* CORS (making it the outermost
        layer) would let it short-circuit with a 503 before CORS ever
        runs, leaving a cross-origin browser client with an opaque
        network error instead of a readable 503. Registration order in
        _build_test_app mirrors apps/api/main.py exactly, so this proves
        the shipped ordering, not just the middleware in isolation.
        """
        monkeypatch.setattr(settings, "MAX_CONCURRENT_REQUESTS", 1)
        monkeypatch.setattr(settings, "CONCURRENCY_QUEUE_TIMEOUT_MS", 100)
        app = _build_test_app(slow_delay_s=0.3)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            responses = await asyncio.gather(
                *(client.post("/api/slow", headers={"Origin": "http://localhost:3000"}) for _ in range(3))
            )

        rejected = [r for r in responses if r.status_code == 503]
        assert rejected, "test setup should have produced at least one 503"
        for r in rejected:
            assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"

    @pytest.mark.asyncio
    async def test_non_api_paths_bypass_the_ceiling(self, monkeypatch):
        """/health must stay responsive even while /api/* is saturated — see the middleware's own docstring."""
        monkeypatch.setattr(settings, "MAX_CONCURRENT_REQUESTS", 1)
        monkeypatch.setattr(settings, "CONCURRENCY_QUEUE_TIMEOUT_MS", 200)
        app = _build_test_app(slow_delay_s=0.3)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            slow_task = asyncio.create_task(client.post("/api/slow"))
            await asyncio.sleep(0.02)  # let the slow request actually claim the one slot first
            health_response = await client.get("/health")
            await slow_task

        assert health_response.status_code == 200

    @pytest.mark.asyncio
    async def test_ceiling_disabled_when_zero(self, monkeypatch):
        monkeypatch.setattr(settings, "MAX_CONCURRENT_REQUESTS", 0)
        app = _build_test_app(slow_delay_s=0.05)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            responses = await asyncio.gather(*(client.post("/api/slow") for _ in range(10)))
        assert all(r.status_code == 200 for r in responses)

    @pytest.mark.asyncio
    async def test_semaphore_releases_after_success_and_after_rejection(self, monkeypatch):
        """No leaked slots — capacity must fully recover once in-flight requests finish."""
        monkeypatch.setattr(settings, "MAX_CONCURRENT_REQUESTS", 1)
        monkeypatch.setattr(settings, "CONCURRENCY_QUEUE_TIMEOUT_MS", 100)
        app = _build_test_app(slow_delay_s=0.15)

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await asyncio.gather(*(client.post("/api/slow") for _ in range(3)))  # some of these 503
            # A fresh round after everything settles must succeed cleanly.
            second_round = await asyncio.gather(*(client.post("/api/slow") for _ in range(1)))

        assert second_round[0].status_code == 200
