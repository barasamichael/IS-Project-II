"""Module 11: User Model — SB-TECH-2026-001 §5.3"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_FRONTEND_DIR = Path(__file__).parent.parent
if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))


@pytest.fixture()
def app():
    """
    Create a Flask app with in-memory SQLite for each test.
    :return: Flask app instance configured for testing.
    """
    from app import create_app, db as _db

    flask_app = create_app("testing")
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    flask_app.config["WTF_CSRF_ENABLED"] = False

    with flask_app.app_context():
        _db.create_all()
        yield flask_app
        _db.session.remove()
        _db.drop_all()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_create_user_persists(app) -> None:
    """User.create() persists a new user to the database and returns a User instance."""
    from app.models import User

    with app.app_context():
        with patch("app.models.user.send_email"):
            user = User.create(
                {
                    "fullName": "Alice Wanjiru",
                    "username": "alicew",
                    "emailAddress": "alice@example.com",
                    "password": "Pass1234!",
                }
            )

        assert user is not None
        assert user.userId is not None
        fetched = User.query.filter_by(emailAddress="alice@example.com").first()
        assert fetched is not None
        assert fetched.fullName == "Alice Wanjiru"


def test_password_not_readable(app) -> None:
    """Accessing user.password raises AttributeError."""
    from app.models import User

    with app.app_context():
        with patch("app.models.user.send_email"):
            user = User.create(
                {
                    "fullName": "Bob Kamau",
                    "username": "bobk",
                    "emailAddress": "bob@example.com",
                    "password": "SecurePass1",
                }
            )

        with pytest.raises(AttributeError):
            _ = user.password  # type: ignore[assignment]


def test_verify_password_correct(app) -> None:
    """user.verifyPassword(correct_password) returns True."""
    from app.models import User

    with app.app_context():
        with patch("app.models.user.send_email"):
            user = User.create(
                {
                    "fullName": "Carol Ochieng",
                    "username": "carolo",
                    "emailAddress": "carol@example.com",
                    "password": "CorrectPass9",
                }
            )

        assert user.verifyPassword("CorrectPass9") is True


def test_verify_password_incorrect(app) -> None:
    """user.verifyPassword(wrong_password) returns False."""
    from app.models import User

    with app.app_context():
        with patch("app.models.user.send_email"):
            user = User.create(
                {
                    "fullName": "David Mwangi",
                    "username": "davidm",
                    "emailAddress": "david@example.com",
                    "password": "CorrectPass9",
                }
            )

        assert user.verifyPassword("WrongPassword") is False


def test_update_returns_success_tuple(app) -> None:
    """user.update(valid_details) returns (True, message_string)."""
    from app.models import User

    with app.app_context():
        with patch("app.models.user.send_email"):
            user = User.create(
                {
                    "fullName": "Eve Njoroge",
                    "username": "even",
                    "emailAddress": "eve@example.com",
                    "password": "SecurePass1",
                }
            )

        result = user.update({"fullName": "Eve Njoroge Updated"})

        assert isinstance(result, tuple)
        assert result[0] is True
        assert isinstance(result[1], str)
        assert user.fullName == "Eve Njoroge Updated"


def test_delete_removes_record(app) -> None:
    """user.delete() removes the record from the database."""
    from app.models import User

    with app.app_context():
        with patch("app.models.user.send_email"):
            user = User.create(
                {
                    "fullName": "Frank Otieno",
                    "username": "franko",
                    "emailAddress": "frank@example.com",
                    "password": "SecurePass1",
                }
            )
        user_id = user.userId

        user.delete()

        fetched = User.query.get(user_id)
        assert fetched is None, "User record should be deleted from the database"
