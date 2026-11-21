from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from main import app, lifespan


@pytest.mark.anyio
async def test_html_404_exception_handler(client: AsyncClient):
    response = await client.get("/this-page-does-not-exist")
    assert response.status_code == 404
    assert "text/html" in response.headers.get("content-type", "")
    assert "application/json" not in response.headers.get("content-type", "")


@pytest.mark.anyio
async def test_api_404_exception_handler_bypasses_html(client: AsyncClient):
    response = await client.get("/api/users/99999")
    assert response.status_code == 404
    assert "text/html" not in response.headers.get("content-type", "")
    assert "application/json" in response.headers.get("content-type", "")


@pytest.mark.anyio
async def test_html_validation_exception_handler(client: AsyncClient):
    response = await client.get("/posts/abc")
    assert response.status_code == 422
    assert "text/html" in response.headers.get("content-type", "")
    assert "application/json" not in response.headers.get("content-type", "")
    assert "Invalid request" in response.text


@pytest.mark.anyio
async def test_api_validation_exception_handler(client: AsyncClient):
    response = await client.get("/api/posts/abc")
    assert response.status_code == 422
    assert "text/html" not in response.headers.get("content-type", "")
    assert "application/json" in response.headers.get("content-type", "")


@pytest.mark.anyio
async def test_lifespan_teardown():
    with patch("main.engine") as mock_engine:
        mock_engine.dispose = AsyncMock()
        async with lifespan(app):
            pass
        mock_engine.dispose.assert_awaited_once()


@pytest.mark.anyio
async def test_health_check_success(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "exc",
    [
        SQLAlchemyError("connection refused"),
        OSError("network unreachable"),
        RuntimeError("pool exhausted"),
    ],
    ids=["sqlalchemy", "oserror", "runtime"],
)
async def test_health_check_db_failure_returns_503(client: AsyncClient, exc):
    failing_db = AsyncMock()
    failing_db.execute.side_effect = exc

    async def override_get_db():
        yield failing_db

    app.dependency_overrides[get_db] = override_get_db
    try:
        response = await client.get("/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "text/html" in response.headers["content-type"]
    assert "Database unavailable" in response.text
