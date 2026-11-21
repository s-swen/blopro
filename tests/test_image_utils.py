from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_utils import (
    delete_profile_image,
    process_profile_image,
)


def create_image_bytes(mode: str = "RGB", size: tuple = (500, 500)) -> bytes:
    img = Image.new(mode, size, color="red")
    buffer = BytesIO()
    img.save(buffer, format="PNG" if mode in ("RGBA", "LA", "P") else "JPEG")
    return buffer.getvalue()


def test_process_profile_image_rgb():
    content = create_image_bytes()
    processed_bytes, filename = process_profile_image(content)
    assert filename.endswith(".jpg")
    assert len(processed_bytes) > 0
    with Image.open(BytesIO(processed_bytes)) as img:
        assert img.format == "JPEG"
        assert img.size == (300, 300)
        assert img.mode == "RGB"


def test_process_profile_image_rgba():
    content = create_image_bytes(mode="RGBA")
    processed_bytes, filename = process_profile_image(content)
    assert filename.endswith(".jpg")
    assert len(processed_bytes) > 0
    with Image.open(BytesIO(processed_bytes)) as img:
        assert img.format == "JPEG"
        assert img.size == (300, 300)
        assert img.mode == "RGB"


@pytest.mark.anyio
async def test_delete_profile_image_none():
    with patch("image_utils._get_s3_client") as mock_s3_client:
        await delete_profile_image(None)
        mock_s3_client.assert_not_called()


@pytest.mark.anyio
async def test_delete_profile_image_success():
    mock_s3 = MagicMock()

    with patch("image_utils._get_s3_client", return_value=mock_s3):
        await delete_profile_image("test_image.jpg")
        mock_s3.delete_object.assert_called_once()
        assert (
            mock_s3.delete_object.call_args.kwargs["Key"]
            == "profile_pics/test_image.jpg"
        )
