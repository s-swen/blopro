from unittest.mock import AsyncMock, patch

import pytest

from config import settings
from email_utils import send_email, send_password_reset_email


@pytest.mark.anyio
@patch("email_utils.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_email_plain_text(mock_send: AsyncMock):
    await send_email(
        to_email="test@example.com",
        subject="Test Subject",
        plain_text="Test body content",
    )
    mock_send.assert_awaited_once()
    message = mock_send.call_args.args[0]
    kwargs = mock_send.call_args.kwargs
    assert message["From"] == settings.mail_from
    assert message["To"] == "test@example.com"
    assert message["Subject"] == "Test Subject"
    assert kwargs["hostname"] == settings.mail_server
    assert kwargs["port"] == settings.mail_port


@pytest.mark.anyio
@patch("email_utils.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_email_with_html(mock_send: AsyncMock):
    await send_email(
        to_email="test@example.com",
        subject="Test Subject",
        plain_text="Test body content",
        html_content="<h1>Test body content</h1>",
    )
    mock_send.assert_awaited_once()
    message = mock_send.call_args.args[0]
    assert message.is_multipart() is True


@pytest.mark.anyio
@patch("email_utils.aiosmtplib.send", new_callable=AsyncMock)
async def test_send_password_reset_email(mock_send: AsyncMock):
    await send_password_reset_email(
        to_email="user@example.com", username="johndoe", token="secure_reset_token_123"
    )

    mock_send.assert_awaited_once()
    message = mock_send.call_args.args[0]
    assert message.is_multipart() is True
    assert message["To"] == "user@example.com"
    assert "Reset Your Password" in message["Subject"]

    msg_str = str(message)
    assert "johndoe" in msg_str
    assert "secure_reset_token_123" in msg_str
    assert "reset-password" in msg_str
