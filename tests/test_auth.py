from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import HTTPException, status

from auth import (
    create_access_token,
    generate_reset_token,
    get_current_user,
    hash_password,
    hash_reset_token,
    verify_access_token,
    verify_password,
)
from config import settings
from models import User


def test_hash_and_verify_password():
    plain = "super_secret_password"
    hashed = hash_password(plain)
    assert hashed != plain
    assert verify_password(plain, hashed) is True
    assert verify_password("wrong_password", hashed) is False


def test_generate_and_hash_reset_token():
    token = generate_reset_token()
    assert len(token) > 30
    token_hash = hash_reset_token(token)
    assert len(token_hash) == 64
    assert hash_reset_token("diff_token") != token_hash


def test_create_access_token_with_default_expiry():
    token = create_access_token({"sub": "123"})
    payload = jwt.decode(
        token,
        settings.secret_key.get_secret_value(),
        algorithms=[settings.algorithm],
    )
    assert payload["sub"] == "123"
    assert "exp" in payload


def test_create_access_token_with_custom_expiry():
    token = create_access_token({"sub": "123"}, expires_delta=timedelta(hours=1))
    payload = jwt.decode(
        token,
        settings.secret_key.get_secret_value(),
        algorithms=[settings.algorithm],
    )
    assert payload["sub"] == "123"
    assert "exp" in payload


def test_verify_access_token_valid():
    token = create_access_token({"sub": "123"})
    assert verify_access_token(token) == "123"


def test_verify_access_token_invalid():
    assert verify_access_token("invalid_token") is None


@pytest.mark.anyio
async def test_get_current_user_success(db_session):
    new_user = User(
        username="testuser",
        email="test@example.com",
        password_hash=hash_password("password123"),
    )
    db_session.add(new_user)
    await db_session.commit()
    await db_session.refresh(new_user)
    token = create_access_token(data={"sub": str(new_user.id)})
    user = await get_current_user(token=token, db=db_session)
    assert user.id == new_user.id
    assert user.username == "testuser"


@pytest.mark.anyio
async def test_get_current_user_invalid_token(db_session):
    with pytest.raises(HTTPException) as e:
        await get_current_user(token="invalid_token", db=db_session)
    assert e.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert e.value.detail == "Invalid or expired token"


@pytest.mark.anyio
async def test_get_current_user_non_integer_sub(db_session):
    with pytest.raises(HTTPException) as e:
        await get_current_user(
            token=create_access_token({"sub": "not_an_int"}), db=db_session
        )
    assert e.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert e.value.detail == "Invalid or expired token"


@pytest.mark.anyio
async def test_get_current_user_not_found_in_db(db_session):
    with pytest.raises(HTTPException) as e:
        await get_current_user(
            token=create_access_token({"sub": "999999"}), db=db_session
        )
    assert e.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert e.value.detail == "User not found"
