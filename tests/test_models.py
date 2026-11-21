from config import settings
from models import User


def test_image_path_returns_s3_url_when_image_exists():
    user = User(image_file="avatar.png")
    expected_url = f"https://{settings.s3_bucket_name}.s3.{settings.s3_region}.amazonaws.com/profile_pics/avatar.png"
    assert user.image_path == expected_url


def test_image_path_returns_default_when_image_is_missing():
    user = User(image_file=None)
    assert user.image_path == "/static/profile_pics/default.jpg"
