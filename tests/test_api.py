"""Tests for FastAPI endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport

from main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_get_character(client):
    resp = await client.get("/api/character/")
    assert resp.status_code == 200
    data = resp.json()
    assert "name" in data
    assert "niche" in data


@pytest.mark.asyncio
async def test_get_current_plan(client):
    resp = await client.get("/api/content/plan")
    assert resp.status_code == 200
    data = resp.json()
    assert "plan" in data
    assert "count" in data


@pytest.mark.asyncio
async def test_get_publish_queue(client):
    resp = await client.get("/api/publishing/queue")
    assert resp.status_code == 200
    data = resp.json()
    assert "queue" in data
