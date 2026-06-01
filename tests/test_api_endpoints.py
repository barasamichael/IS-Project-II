"""Module 9: API Endpoints — SB-TECH-2026-001 §5.2"""

import io
import os


os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
# Use the same key as conftest.py so it matches api/main.py's module-level API_KEY
os.environ.setdefault("SETTLEBOT_API_KEY", "test-conftest-api-key")
os.environ.setdefault("SETTLEBOT_LOCALE", "nairobi")

from fastapi.testclient import TestClient
from api.main import app

_API_KEY = os.environ["SETTLEBOT_API_KEY"]
_AUTH = {"Authorization": f"Bearer {_API_KEY}"}
_client = TestClient(app, headers=_AUTH)
_unauth_client = TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_query_endpoint_requires_api_key() -> None:
    """POST /query without Authorization header returns HTTP 401."""
    response = _unauth_client.post(
        "/query", json={"query": "Where can I find housing?"}
    )
    assert response.status_code == 401


def test_query_endpoint_rejects_long_query() -> None:
    """POST /query with a query longer than 2000 characters returns HTTP 422."""
    long_query = "a" * 2001
    response = _client.post("/query", json={"query": long_query})
    assert response.status_code == 422


def test_query_endpoint_rejects_empty_query() -> None:
    """POST /query with query shorter than 3 characters returns HTTP 422."""
    response = _client.post("/query", json={"query": "ab"})
    assert response.status_code == 422


def test_health_endpoint_no_auth_required() -> None:
    """GET /health returns HTTP 200 without any Authorization header."""
    response = _unauth_client.get("/health")
    assert response.status_code == 200


def test_upload_rejects_path_traversal_filename() -> None:
    """
    POST /documents/upload with a path-traversal filename is handled safely —
    either rejected (400) or the stored file uses only the basename component.
    Regression guard for Milestone 1 path-traversal fix.
    """
    file_content = b"safe text content for upload test"
    response = _client.post(
        "/documents/upload",
        files={"file": ("../../etc/passwd", io.BytesIO(file_content), "text/plain")},
    )
    if response.status_code == 200:
        # If accepted, the stored filename must be basename only
        data = response.json()
        filename = data.get("file_name", "")
        assert "/" not in filename and "\\" not in filename, (
            "Path traversal components must be stripped from uploaded filename"
        )
    else:
        # Any error response (400, 422) is also acceptable
        assert response.status_code in (400, 422)


def test_upload_rejects_forbidden_extension() -> None:
    """POST /documents/upload with a .exe file returns HTTP 400."""
    response = _client.post(
        "/documents/upload",
        files={"file": ("malware.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
    )
    assert response.status_code == 400


def test_search_patterns_returns_501() -> None:
    """
    GET /analytics/search-patterns returns HTTP 501 Not Implemented.
    Regression guard for Milestone 2 stub-endpoint fix.
    """
    response = _client.get("/analytics/search-patterns")
    assert response.status_code == 501
