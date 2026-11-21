import pytest
from httpx import AsyncClient

from .conftest import auth_header, create_test_user, login_user

pytestmark = pytest.mark.anyio
BASE = "/api/posts"


async def make_user_and_token(
    client: AsyncClient,
    username: str = "testuser",
    email: str = "test@example.com",
) -> tuple[dict, str]:
    user = await create_test_user(client, username=username, email=email)
    token = await login_user(client, email=email)
    return user, token


async def make_post(
    client: AsyncClient, token: str, title: str = "Title", content: str = "Content"
) -> dict:
    response = await client.post(
        BASE, json={"title": title, "content": content}, headers=auth_header(token)
    )
    assert response.status_code == 201, response.text
    return response.json()


async def test_get_posts_empty(client: AsyncClient):
    response = await client.get(BASE)
    assert response.status_code == 200
    data = response.json()
    assert data["posts"] == []
    assert data["total"] == 0
    assert data["skip"] == 0
    assert not data["has_more"]


async def test_get_posts_newest_first(client: AsyncClient):
    _, token = await make_user_and_token(client)
    await make_post(client, token, title="first")
    await make_post(client, token, title="second")
    response = await client.get(BASE)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert [p["title"] for p in data["posts"]] == ["second", "first"]
    assert data["posts"][0]["author"]["username"] == "testuser"


async def test_get_posts_pagination_has_more(client: AsyncClient):
    _, token = await make_user_and_token(client)
    for i in range(3):
        await make_post(client, token, title=f"post {i}")
    page1 = (await client.get(BASE, params={"skip": 0, "limit": 2})).json()
    assert len(page1["posts"]) == 2
    assert page1["total"] == 3
    assert page1["limit"] == 2
    assert page1["has_more"] is True

    page2 = (await client.get(BASE, params={"skip": 2, "limit": 2})).json()
    assert len(page2["posts"]) == 1
    assert page2["skip"] == 2
    assert page2["has_more"] is False


@pytest.mark.parametrize("params", [{"skip": -1}, {"limit": 0}, {"limit": 101}])
async def test_get_posts_invalid_query_params(client: AsyncClient, params):
    response = await client.get(BASE, params=params)
    assert response.status_code == 422


async def test_create_post_success(client: AsyncClient):
    user, token = await make_user_and_token(client)
    response = await client.post(
        BASE, json={"title": "Hello", "content": "World"}, headers=auth_header(token)
    )
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Hello"
    assert data["content"] == "World"
    assert data["user_id"] == user["id"]
    assert data["author"]["id"] == user["id"]


async def test_create_post_requires_auth(client: AsyncClient):
    response = await client.post(BASE, json={"title": "t", "content": "c"})
    assert response.status_code == 401


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "", "content": "c"},
        {"title": "t", "content": ""},
        {"title": "x" * 101, "content": "c"},
        {"content": "c"},
    ],
)
async def test_create_post_validation_errors(client: AsyncClient, payload):
    _, token = await make_user_and_token(client)
    response = await client.post(BASE, json=payload, headers=auth_header(token))
    assert response.status_code == 422


async def test_get_post_success(client: AsyncClient):
    user, token = await make_user_and_token(client)
    created = await make_post(client, token, content="find me")
    assert created["user_id"] == user["id"]
    data = (await client.get(f"{BASE}/{created['id']}")).json()
    assert data["id"] == created["id"]
    assert data["content"] == "find me"


async def test_get_post_not_found(client: AsyncClient):
    response = await client.get(f"{BASE}/99999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


async def test_update_post_full_success(client: AsyncClient):
    _, token = await make_user_and_token(client)
    created = await make_post(client, token)
    response = await client.put(
        f"{BASE}/{created['id']}",
        json={"title": "new", "content": "new"},
        headers=auth_header(token),
    )
    updated = response.json()
    assert updated["title"] == "new"
    assert updated["content"] == "new"
    assert updated["author"]["username"] == "testuser"


async def test_update_post_full_not_found(client: AsyncClient):
    _, token = await make_user_and_token(client)
    response = await client.put(
        f"{BASE}/99999",
        json={"title": "new", "content": "new"},
        headers=auth_header(token),
    )
    assert response.status_code == 404


async def test_update_post_full_forbidden(client: AsyncClient):
    _, owner_token = await make_user_and_token(client)
    _, another_token = await make_user_and_token(
        client, username="another", email="another@example.com"
    )
    owner_created = await make_post(client, token=owner_token)
    response = await client.put(
        f"{BASE}/{owner_created['id']}",
        json={"title": "t", "content": "c"},
        headers=auth_header(another_token),
    )
    assert response.status_code == 403
    assert "not allowed" in response.json()["detail"]


async def test_update_post_full_requires_auth(client: AsyncClient):
    response = await client.put(f"{BASE}/1", json={"title": "t", "content": "c"})
    assert response.status_code == 401


async def test_update_post_partial_only_title(client: AsyncClient):
    _, token = await make_user_and_token(client)
    created = await make_post(client, token, title="old title", content="old content")
    response = await client.patch(
        f"{BASE}/{created['id']}",
        json={"title": "new title"},
        headers=auth_header(token),
    )
    data = response.json()
    assert response.status_code == 200
    assert data["title"] == "new title"
    assert data["content"] == "old content"


async def test_update_post_partial_not_found(client: AsyncClient):
    _, token = await make_user_and_token(client)
    response = await client.patch(
        f"{BASE}/999999", json={"title": "x"}, headers=auth_header(token)
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Post not found"


async def test_update_post_partial_forbidden(client: AsyncClient):
    _, owner_token = await make_user_and_token(client)
    _, other_token = await make_user_and_token(
        client, username="other", email="other@example.com"
    )
    created = await make_post(client, owner_token)

    response = await client.patch(
        f"{BASE}/{created['id']}",
        json={"title": "hack"},
        headers=auth_header(other_token),
    )
    assert response.status_code == 403


async def test_update_post_partial_invalid_data(client: AsyncClient):
    _, token = await make_user_and_token(client)
    created = await make_post(client, token)
    response = await client.patch(
        f"{BASE}/{created['id']}", json={"title": ""}, headers=auth_header(token)
    )
    assert response.status_code == 422


async def test_delete_post_success(client: AsyncClient):
    _, token = await make_user_and_token(client)
    created = await make_post(client, token)

    response = await client.delete(
        f"{BASE}/{created['id']}", headers=auth_header(token)
    )
    assert response.status_code == 204

    follow_up = await client.get(f"{BASE}/{created['id']}")
    assert follow_up.status_code == 404


async def test_delete_post_not_found(client: AsyncClient):
    _, token = await make_user_and_token(client)
    response = await client.delete(f"{BASE}/999999", headers=auth_header(token))
    assert response.status_code == 404
    assert response.json()["detail"] == "Post not found"


async def test_delete_post_forbidden(client: AsyncClient):
    _, owner_token = await make_user_and_token(client)
    _, other_token = await make_user_and_token(
        client, username="other", email="other@example.com"
    )
    created = await make_post(client, owner_token)

    response = await client.delete(
        f"{BASE}/{created['id']}", headers=auth_header(other_token)
    )
    assert response.status_code == 403

    assert (await client.get(f"{BASE}/{created['id']}")).status_code == 200


async def test_delete_post_requires_auth(client: AsyncClient):
    response = await client.delete(f"{BASE}/1")
    assert response.status_code == 401
