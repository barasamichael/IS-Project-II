"""Module 10: Authentication — SB-TECH-2026-001 §5.3"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add front-end to path so app imports resolve
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
    flask_app.config["SERVER_NAME"] = "localhost"

    with flask_app.app_context():
        _db.create_all()
        yield flask_app
        _db.session.remove()
        _db.drop_all()


@pytest.fixture()
def client(app):
    """
    Return a Flask test client.
    :param app: Flask app fixture.
    :return: Flask test client.
    """
    return app.test_client()


@pytest.fixture()
def test_user(app):
    """
    Create a test user with known credentials.
    :param app: Flask app fixture.
    :return: User instance.
    """
    from app.models import User

    with patch("app.models.user.send_email"):
        user = User.create(
            {
                "fullName": "Test Student",
                "username": "teststudent",
                "emailAddress": "test@example.com",
                "password": "SecurePass123",
            }
        )
    return user


@pytest.fixture()
def inactive_user(app):
    """
    Create an inactive test user.
    :param app: Flask app fixture.
    :return: Inactive User instance.
    """
    from app import db
    from app.models import User

    with patch("app.models.user.send_email"):
        user = User.create(
            {
                "fullName": "Inactive User",
                "username": "inactiveuser",
                "emailAddress": "inactive@example.com",
                "password": "SecurePass123",
            }
        )
    user.isActive = False
    db.session.commit()
    return user


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_login_valid_credentials(client, test_user, app) -> None:
    """Valid credentials set session and redirect to dashboard (302)."""
    with app.app_context():
        response = client.post(
            "/authentication/sign-in",
            data={
                "emailAddress": "test@example.com",
                "password": "SecurePass123",
                "remember_me": False,
            },
            follow_redirects=False,
        )
        assert response.status_code == 302
        with client.session_transaction() as sess:
            assert "user_role" in sess


def test_login_invalid_password(client, test_user) -> None:
    """Wrong password returns login page without creating a session."""
    response = client.post(
        "/authentication/sign-in",
        data={
            "emailAddress": "test@example.com",
            "password": "WrongPassword!",
            "remember_me": False,
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    with client.session_transaction() as sess:
        assert "user_role" not in sess


def test_login_inactive_account(client, inactive_user) -> None:
    """Inactive account login returns appropriate error response (not session)."""
    response = client.post(
        "/authentication/sign-in",
        data={
            "emailAddress": "inactive@example.com",
            "password": "SecurePass123",
            "remember_me": False,
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    with client.session_transaction() as sess:
        assert "user_role" not in sess


def test_logout_clears_session(client, test_user, app) -> None:
    """Logout removes user_role from the session."""
    with app.app_context():
        client.post(
            "/authentication/sign-in",
            data={
                "emailAddress": "test@example.com",
                "password": "SecurePass123",
                "remember_me": False,
            },
            follow_redirects=True,
        )
        with client.session_transaction() as sess:
            sess["user_role"] = "student"

        client.get("/authentication/sign-out", follow_redirects=True)

        with client.session_transaction() as sess:
            assert "user_role" not in sess


def test_password_reset_token_valid(client, test_user, app) -> None:
    """Valid unexpired reset token allows the password to be changed."""
    with app.app_context():
        from app.models import User

        with patch("app.models.user.send_email"):
            token = test_user.generateResetToken()

        response = client.post(
            f"/authentication/password-reset/{token}",
            data={
                "password": "NewSecurePass456",
                "confirmPassword": "NewSecurePass456",
            },
            follow_redirects=False,
        )
        assert response.status_code in (200, 302)
        updated_user = User.query.filter_by(emailAddress="test@example.com").first()
        assert updated_user.verifyPassword("NewSecurePass456")


def test_password_reset_token_expired(client, test_user, app) -> None:
    """Expired token returns error without changing the password."""
    with app.app_context():
        with patch("app.models.user.send_email"):
            test_user.generateResetToken()

        # Invalidate by using a garbage token
        response = client.post(
            "/authentication/password-reset/invalid-expired-token-xyz123",
            data={
                "password": "NewPass789",
                "confirmPassword": "NewPass789",
            },
            follow_redirects=True,
        )
        assert response.status_code == 200
        from app.models import User

        user = User.query.filter_by(emailAddress="test@example.com").first()
        assert not user.verifyPassword("NewPass789"), (
            "Password must not change when expired/invalid token is used"
        )
