import pprint
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import get_args
from unittest.mock import AsyncMock, patch

import pytest
from botocore.exceptions import ClientError
from httpx import AsyncClient
from PIL import Image
from sqlalchemy import func, select

import models
from auth import CurrentUser, hash_reset_token
from config import settings
from main import app
from tests.conftest import auth_header, create_test_user, login_user

OTHER_EMAIL = "other@example.com"
BASE = "/api/users"
pytestmark = pytest.mark.anyio


async def create_other_user(client: AsyncClient) -> tuple[dict, str]:
    user = await create_test_user(client, username="other", email=OTHER_EMAIL)
    token = await login_user(client, email=OTHER_EMAIL)
    return user, token


def make_image_bytes(fmt: str = "PNG") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "red").save(buffer, format=fmt)
    return buffer.getvalue()


async def upload_picture(
    client: AsyncClient, user_id: int, token: str, data: bytes | None = None
):
    return await client.patch(
        f"{BASE}/{user_id}/picture",
        files={
            "file": ("p.png", BytesIO(data or make_image_bytes()), "image/png"),
        },
        headers=auth_header(token),
    )


async def count_reset_tokens(db_session, user_id: int) -> int:
    result = await db_session.execute(
        select(func.count())
        .select_from(models.PasswordResetToken)
        .where(models.PasswordResetToken.user_id == user_id)
    )
    return result.scalar()


async def test_create_user_validation_error(client: AsyncClient):
    test_data = {"username": "testuser"}
    response = await client.post(BASE, json=test_data)
    assert response.status_code == 422
    assert "email" in response.text
    assert "password" in response.text


async def test_create_user_duplicate_email(client: AsyncClient):
    await create_test_user(client)
    test_data = {
        "username": "different_user",
        "email": "test@example.com",
        "password": "password123",
    }
    response = await client.post(
        BASE,
        json=test_data,
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Email or user name already used"


async def test_login_success(client: AsyncClient):
    await create_test_user(client)
    response = await client.post(
        f"{BASE}/token",
        data={"username": "test@example.com", "password": "testpassword123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["token_type"] == "bearer"
    assert data["access_token"]


async def test_login_wrong_password(client: AsyncClient):
    await create_test_user(client)
    response = await client.post(
        f"{BASE}/token",
        data={"username": "test@example.com", "password": "wrongpassword"},
    )
    assert response.status_code == 401
    assert "Incorrect" in response.json()["detail"]
    assert response.headers["www-authenticate"] == "Bearer"


async def test_login_no_user(client: AsyncClient):
    await create_test_user(client)
    response = await client.post(
        f"{BASE}/token",
        data={"username": "nouser@example.com", "password": "testpassword123"},
    )
    assert response.status_code == 401
    assert "Incorrect" in response.json()["detail"]
    assert response.headers["www-authenticate"] == "Bearer"


async def test_get_current_user_success(client: AsyncClient):
    await create_test_user(client)
    token = await login_user(client)
    response = await client.get(f"{BASE}/me", headers=auth_header(token))
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"
    assert response.json()["email"] == "test@example.com"


async def test_get_current_user_requires_auth(client: AsyncClient):
    response = await client.get(f"{BASE}/me")
    assert response.status_code == 401


async def test_forgot_password_unknown_email_same_response(client: AsyncClient):
    with patch(
        "routers.users.send_password_reset_email", new_callable=AsyncMock
    ) as mock_send:
        response = await client.post(
            f"{BASE}/forgot-password", json={"email": "ghost@example.com"}
        )
        assert response.status_code == 202
        assert "If an account exists" in response.json()["message"]
        mock_send.assert_not_awaited()


async def test_forgot_password_replaces_previous_token(client: AsyncClient, db_session):
    with patch(
        "routers.users.send_password_reset_email", new_callable=AsyncMock
    ) as mock_send:
        user = await create_test_user(client)
        response = await client.post(
            f"{BASE}/forgot-password", json={"email": "test@example.com"}
        )
        assert response.status_code == 202
        mock_send.assert_awaited_once()
        assert await count_reset_tokens(db_session, user_id=user["id"]) == 1
        response = await client.post(
            f"{BASE}/forgot-password", json={"email": "test@example.com"}
        )
        assert response.status_code == 202
        assert await count_reset_tokens(db_session, user_id=user["id"]) == 1


async def test_reset_password_invalid_token(client: AsyncClient):
    response = await client.post(
        f"{BASE}/reset-password",
        json={
            "token": "does-not-exist",
            "new_password": "newpassword123",
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or expired reset token"


async def test_reset_password_expired_token(client: AsyncClient, db_session):
    user = await create_test_user(client)
    db_session.add(
        models.PasswordResetToken(
            user_id=user["id"],
            token_hash=hash_reset_token("expired-raw-token"),
            expires_at=datetime.now(UTC) - timedelta(hours=1),
        )
    )
    await db_session.commit()
    response = await client.post(
        f"{BASE}/reset-password",
        json={
            "token": "expired-raw-token",
            "new_password": "newpassword123",
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid or expired reset token"


async def test_reset_password_success_full_flow(client: AsyncClient, db_session):
    user = await create_test_user(client)
    with patch(
        "routers.users.send_password_reset_email", new_callable=AsyncMock
    ) as mock_send:
        response = await client.post(
            f"{BASE}/forgot-password", json={"email": "test@example.com"}
        )
        assert response.status_code == 202
        mock_send.assert_awaited_once()
    token = mock_send.call_args.kwargs["token"]
    response = await client.post(
        f"{BASE}/reset-password",
        json={"token": token, "new_password": "newpassword123"},
    )
    assert response.status_code == 200
    assert "successfully" in response.json()["message"]
    assert await count_reset_tokens(db_session, user["id"]) == 0
    response = await client.post(
        f"{BASE}/token",
        data={
            "username": "test@example.com",
            "password": "newpassword123",
        },
    )
    assert response.status_code == 200
    response = await client.post(
        f"{BASE}/token",
        data={
            "username": "test@example.com",
            "password": "testpassword123",
        },
    )
    assert response.status_code == 401
    assert "Incorrect" in response.json()["detail"]


async def test_reset_password_user_missing(client: AsyncClient, db_session):
    user = await create_test_user(client)
    db_session.add(
        models.PasswordResetToken(
            user_id=user["id"],
            token_hash=hash_reset_token("orphan-token"),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
    )
    await db_session.commit()
    real_execute = db_session.execute
    calls = {"n": 0}

    class _Empty:
        def scalars(self):
            return self

        def first(self):
            return None

    async def flaky_execute(stmt, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            return _Empty()
        return await real_execute(stmt, *args, **kwargs)

    with patch.object(db_session, "execute", flaky_execute):
        response = await client.post(
            f"{BASE}/reset-password",
            json={"token": "orphan-token", "new_password": "newpassword123"},
        )
        assert response.status_code == 400
        assert "Invalid or expired" in response.json()["detail"]


async def test_change_password_wrong_current_password(client: AsyncClient):
    await create_test_user(client)
    token = await login_user(client)
    response = await client.patch(
        f"{BASE}/me/password",
        json={"current_password": "not-my-password", "new_password": "newpassword123"},
        headers=auth_header(token),
    )
    assert response.status_code == 400
    assert "Current password is incorrect" in response.json()["detail"]


async def test_change_password_success(client: AsyncClient, db_session):
    user = await create_test_user(client)
    token = await login_user(client)
    response = await client.patch(
        f"{BASE}/me/password",
        json={"current_password": "testpassword123", "new_password": "newpassword123"},
        headers=auth_header(token),
    )
    assert response.status_code == 200
    assert "changed successfully" in response.json()["message"]
    assert await count_reset_tokens(db_session, user["id"]) == 0


async def test_get_user_success(client):
    user = await create_test_user(client)
    response = await client.get(f"{BASE}/{user['id']}")
    assert response.status_code == 200
    assert response.json()["id"] == user["id"]


async def test_get_user_posts_user_not_found(client: AsyncClient):
    response = await client.get(f"{BASE}/99999/posts")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


async def test_get_user_posts_empty(client: AsyncClient):
    user = await create_test_user(client)
    response = await client.get(f"{BASE}/{user['id']}/posts")
    assert response.status_code == 200
    data = response.json()
    assert data["posts"] == []
    assert data["total"] == 0
    assert data["has_more"] is False


async def test_get_user_posts_pagination_and_isolation(client: AsyncClient):
    user = await create_test_user(client)
    token = await login_user(client)
    _, other_token = await create_other_user(client)
    for i in range(3):
        await client.post(
            "/api/posts",
            json={
                "title": f"mine {i}",
                "content": "c",
            },
            headers=auth_header(token),
        )
    await client.post(
        "/api/posts",
        json={
            "title": "not mine",
            "content": "c",
        },
        headers=auth_header(other_token),
    )
    page1 = (
        await client.get(f"{BASE}/{user['id']}/posts", params={"skip": 0, "limit": 2})
    ).json()
    assert page1["total"] == 3
    assert page1["skip"] == 0
    assert page1["limit"] == 2
    assert page1["has_more"] is True
    assert [p["title"] for p in page1["posts"]] == ["mine 2", "mine 1"]

    page2 = (
        await client.get(f"{BASE}/{user['id']}/posts", params={"skip": 2, "limit": 2})
    ).json()

    assert page2["skip"] == 2
    assert page2["limit"] == 2
    assert page2["has_more"] is False
    assert [p["title"] for p in page2["posts"]] == ["mine 0"]


async def test_update_user_forbidden(client: AsyncClient):
    await create_test_user(client)
    token = await login_user(client)
    other, _ = await create_other_user(client)
    response = await client.patch(
        f"{BASE}/{other['id']}", json={"username": "hacked"}, headers=auth_header(token)
    )
    assert response.status_code == 403
    assert "not allowed" in response.json()["detail"]


async def test_update_user_not_found(client: AsyncClient):
    dependency = get_args(CurrentUser)[1].dependency
    app.dependency_overrides[dependency] = lambda: SimpleNamespace(id=99999)
    try:
        response = await client.patch(f"{BASE}/99999", json={"username": "ghost"})
    finally:
        app.dependency_overrides.pop(dependency, None)
    assert response.status_code == 404
    assert "User not found" in response.json()["detail"]


async def test_update_user_username_success(client: AsyncClient):
    user = await create_test_user(client)
    token = await login_user(client)
    response = await client.patch(
        f"{BASE}/{user['id']}",
        json={
            "username": "newusername",
        },
        headers=auth_header(token),
    )
    assert response.status_code == 200
    assert response.json()["username"] == "newusername"
    assert response.json()["email"] == "test@example.com"


async def test_update_user_email_success_and_lowercased(client: AsyncClient):
    user = await create_test_user(client)
    token = await login_user(client)
    response = await client.patch(
        f"/api/users/{user['id']}",
        json={"email": "NEW.Address@Example.com"},
        headers=auth_header(token),
    )
    assert response.status_code == 200
    assert response.json()["email"] == "new.address@example.com"


async def test_update_user_username_taken(client: AsyncClient):
    user = await create_test_user(client)
    token = await login_user(client)
    await create_other_user(client)
    response = await client.patch(
        f"/api/users/{user['id']}",
        json={"username": "OTHER"},
        headers=auth_header(token),
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Username already exists"


async def test_update_user_email_taken(client: AsyncClient):
    user = await create_test_user(client)
    token = await login_user(client)
    await create_other_user(client)
    response = await client.patch(
        f"/api/users/{user['id']}",
        json={"email": "OTHER@example.com"},
        headers=auth_header(token),
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Email already registered"


async def test_update_user_empty_body_changes_nothing(client: AsyncClient):
    user = await create_test_user(client)
    token = await login_user(client)
    response = await client.patch(
        f"/api/users/{user['id']}",
        json={},
        headers=auth_header(token),
    )
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"
    assert response.json()["email"] == "test@example.com"


async def test_delete_user_forbidden(client: AsyncClient):
    await create_test_user(client)
    token = await login_user(client)
    other, _ = await create_other_user(client)
    response = await client.delete(
        f"/api/users/{other['id']}",
        headers=auth_header(token),
    )
    assert response.status_code == 403


async def test_delete_user_not_found(client: AsyncClient):
    dependency = get_args(CurrentUser)[1].dependency
    app.dependency_overrides[dependency] = lambda: SimpleNamespace(id=424242)
    try:
        response = await client.delete("/api/users/424242")
    finally:
        app.dependency_overrides.pop(dependency, None)
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"


async def test_delete_user_success_without_picture(client: AsyncClient):
    user = await create_test_user(client)
    token = await login_user(client)
    response = await client.delete(
        f"/api/users/{user['id']}",
        headers=auth_header(token),
    )
    assert response.status_code == 204
    assert (await client.get(f"/api/users/{user['id']}")).status_code == 404


async def test_delete_user_success_removes_picture_from_s3(
    client: AsyncClient, mocked_aws
):
    user = await create_test_user(client)
    token = await login_user(client)
    await upload_picture(client, user["id"], token)
    assert "Contents" in mocked_aws.list_objects_v2(Bucket="test-bucket")
    response = await client.delete(
        f"/api/users/{user['id']}",
        headers=auth_header(token),
    )
    assert response.status_code == 204
    assert "Contents" not in mocked_aws.list_objects_v2(Bucket="test-bucket")


async def test_upload_picture_forbidden(client: AsyncClient):
    await create_test_user(client)
    token = await login_user(client)
    other, _ = await create_other_user(client)
    response = await client.patch(
        f"{BASE}/{other['id']}/picture",
        files={"file": ("p.png", BytesIO(make_image_bytes()), "image/png")},
        headers=auth_header(token),
    )
    assert response.status_code == 403
    assert "Not authorized" in response.json()["detail"]


async def test_upload_picture_too_large(client: AsyncClient):
    user = await create_test_user(client)
    token = await login_user(client)
    too_big = b"0" * (settings.max_upload_size_bytes + 1)
    response = await upload_picture(client, user["id"], token, data=too_big)
    assert response.status_code == 400
    assert "File too large" in response.json()["detail"]


async def test_upload_picture_invalid_image(client: AsyncClient):
    user = await create_test_user(client)
    token = await login_user(client)
    response = await upload_picture(
        client, user["id"], token, data=b"not an image file"
    )
    assert response.status_code == 400
    assert "Invalid image" in response.json()["detail"]


async def test_upload_picture_s3_failure(client: AsyncClient):
    user = await create_test_user(client)
    token = await login_user(client)
    boom = ClientError({"Error": {"Code": "500", "Message": "boom"}}, "PutObject")
    with patch(
        "routers.users.upload_profile_image", new_callable=AsyncMock, side_effect=boom
    ):
        response = await upload_picture(client, user["id"], token)
    assert response.status_code == 500
    assert "Failed to upload image" in response.json()["detail"]


async def test_upload_picture_replaces_old_picture(client: AsyncClient, mocked_aws):
    user = await create_test_user(client)
    token = await login_user(client)
    first_image_file = (await upload_picture(client, user["id"], token)).json()[
        "image_file"
    ]
    response = await upload_picture(client, user["id"], token)
    assert response.status_code == 200
    assert response.json()["image_file"] != first_image_file
    assert len(mocked_aws.list_objects_v2(Bucket="test-bucket")["Contents"]) == 1


async def test_delete_picture_forbidden(client: AsyncClient):
    await create_test_user(client)
    token = await login_user(client)
    other, _ = await create_other_user(client)
    response = await client.delete(
        f"/api/users/{other['id']}/picture",
        headers=auth_header(token),
    )
    assert response.status_code == 403


async def test_delete_picture_when_none_set(client: AsyncClient):
    user = await create_test_user(client)
    token = await login_user(client)
    response = await client.delete(
        f"/api/users/{user['id']}/picture",
        headers=auth_header(token),
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "No profile picture to delete"


@pytest.mark.anyio
async def test_delete_picture_success(client: AsyncClient, mocked_aws):
    user = await create_test_user(client)
    token = await login_user(client)
    await upload_picture(client, user["id"], token)

    response = await client.delete(
        f"/api/users/{user['id']}/picture",
        headers=auth_header(token),
    )
    assert response.status_code == 200
    assert response.json()["image_file"] is None
    assert "Contents" not in mocked_aws.list_objects_v2(Bucket="test-bucket")
