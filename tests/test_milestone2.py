"""
Tests for Milestone 2 — Production Hardening and Observability.

Covers: correlation ID middleware, rate limiting, input validation,
tenacity retry wiring, stub endpoint corrections, and dependency cleanup.
"""

import os
import pytest

os.environ.setdefault("SETTLEBOT_API_KEY", "test-secure-key-for-milestone2-tests")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")

from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from config.constants import ERROR_CODE_RATE_LIMIT_EXCEEDED
from config.constants import LLM_CALL_TIMEOUT_SECONDS
from config.constants import LLM_RETRY_ATTEMPTS
from config.constants import LLM_RETRY_WAIT_MAX
from config.constants import LLM_RETRY_WAIT_MIN
from config.constants import LLM_TAVILY_TIMEOUT_SECONDS


@pytest.fixture()
def client():
    """
    Returns a FastAPI TestClient with the Authorization header pre-set.
    :return: TestClient - Authenticated test client for the FastAPI app.
    """
    return TestClient(
        app, headers={"Authorization": "Bearer test-secure-key-for-milestone2-tests"}
    )


# ── Constants ────────────────────────────────────────────────────────────────


def test_llm_retry_constants_defined():
    """
    LLM_RETRY_ATTEMPTS, LLM_RETRY_WAIT_MIN, LLM_RETRY_WAIT_MAX, and
    LLM_CALL_TIMEOUT_SECONDS must all be defined in config/constants.py.
    """
    assert LLM_RETRY_ATTEMPTS == 3
    assert LLM_RETRY_WAIT_MIN == 1
    assert LLM_RETRY_WAIT_MAX == 8
    assert LLM_CALL_TIMEOUT_SECONDS == 15


def test_tavily_timeout_constant_defined():
    """LLM_TAVILY_TIMEOUT_SECONDS must be defined in config/constants.py."""
    assert LLM_TAVILY_TIMEOUT_SECONDS == 4


def test_rate_limit_error_code_defined():
    """ERROR_CODE_RATE_LIMIT_EXCEEDED must be defined in config/constants.py."""
    assert ERROR_CODE_RATE_LIMIT_EXCEEDED == "RATE_LIMIT_EXCEEDED"


# ── Correlation ID ───────────────────────────────────────────────────────────


def test_response_includes_correlation_id(client):
    """
    Every response must include an X-Request-ID header.
    """
    response = client.get("/health")
    assert "x-request-id" in response.headers


def test_request_id_echoed_when_provided(client):
    """
    When the client supplies X-Request-ID, the same value must appear in the response.
    """
    response = client.get("/health", headers={"X-Request-ID": "test-id-abc-123"})
    assert response.headers.get("x-request-id") == "test-id-abc-123"


def test_unique_ids_generated_per_request(client):
    """
    Two requests without X-Request-ID must receive different correlation IDs.
    """
    r1 = client.get("/health")
    r2 = client.get("/health")
    assert r1.headers.get("x-request-id") != r2.headers.get("x-request-id")


# ── Input validation ─────────────────────────────────────────────────────────


def test_query_too_long_returns_422(client):
    """
    A POST /query request with a query longer than 2000 characters must return HTTP 422.
    """
    payload = {"query": "a" * 2001}
    response = client.post("/query", json=payload)
    assert response.status_code == 422


def test_query_too_short_returns_422(client):
    """
    A POST /query request with a query shorter than 3 characters must return HTTP 422.
    """
    payload = {"query": "ab"}
    response = client.post("/query", json=payload)
    assert response.status_code == 422


def test_query_exactly_min_length_accepted(client):
    """
    A POST /query request with a 3-character query must pass validation (not 422).
    The test mocks service calls to avoid real LLM/DB calls.
    """
    mock_intent = {
        "intent_type": MagicMock(value="off_topic"),
        "topic": MagicMock(value="general"),
        "confidence": 0.1,
        "settlement_relevance": 0.1,
        "classification_method": "cosine",
        "is_off_topic": True,
        "semantic_scores": {},
        "off_topic_indicators": [],
    }
    mock_response = {
        "response": "I can only help with settlement queries.",
        "original_response": None,
        "language_detected": "english",
        "translation_needed": False,
        "token_usage": None,
        "current_time": "2026-01-01",
        "empathy_applied": False,
        "safety_protocols_added": False,
        "crisis_level": "none",
        "emotional_state": None,
        "web_search_used": False,
    }
    with patch("api.main.intent_recognizer.get_intent_info", return_value=mock_intent):
        with patch(
            "api.main.response_generator.generate_response", return_value=mock_response
        ):
            response = client.post("/query", json={"query": "abc"})
    assert response.status_code != 422


def test_query_exactly_max_length_accepted(client):
    """
    A POST /query request with exactly 2000 characters must pass validation.
    """
    mock_intent = {
        "intent_type": MagicMock(value="off_topic"),
        "topic": MagicMock(value="general"),
        "confidence": 0.1,
        "settlement_relevance": 0.1,
        "classification_method": "cosine",
        "is_off_topic": True,
        "semantic_scores": {},
        "off_topic_indicators": [],
    }
    mock_response = {
        "response": "I can only help with settlement queries.",
        "original_response": None,
        "language_detected": "english",
        "translation_needed": False,
        "token_usage": None,
        "current_time": "2026-01-01",
        "empathy_applied": False,
        "safety_protocols_added": False,
        "crisis_level": "none",
        "emotional_state": None,
        "web_search_used": False,
    }
    with patch("api.main.intent_recognizer.get_intent_info", return_value=mock_intent):
        with patch(
            "api.main.response_generator.generate_response", return_value=mock_response
        ):
            response = client.post("/query", json={"query": "a" * 2000})
    assert response.status_code != 422


# ── Stub endpoint corrections ─────────────────────────────────────────────────


def test_analytics_search_patterns_returns_501(client):
    """
    GET /analytics/search-patterns must return HTTP 501 with status=not_implemented.
    """
    response = client.get("/analytics/search-patterns")
    assert response.status_code == 501
    body = response.json()
    assert body.get("status") == "not_implemented"


def test_vector_db_optimize_returns_no_op(client):
    """
    POST /vector-db/optimize must return a body with status=no_op.
    """
    with patch(
        "api.main.vector_db_service.get_collection_stats", return_value={"count": 0}
    ):
        response = client.post("/vector-db/optimize")
    assert response.status_code == 200
    body = response.json()
    assert body.get("status") == "no_op"
    assert "optimizations_available" in body


def test_webhook_with_callback_returns_501(client):
    """
    POST /webhooks/document-processed with a callback_url must return HTTP 501.
    """
    mock_doc = {
        "file_name": "test.pdf",
        "num_chunks": 5,
        "avg_settlement_score": 0.7,
    }
    with patch("api.main.document_processor.get_document_info", return_value=mock_doc):
        response = client.post(
            "/webhooks/document-processed",
            params={
                "doc_id": "test-doc",
                "callback_url": "https://example.com/callback",
            },
        )
    assert response.status_code == 501


# ── Tenacity retry wiring ─────────────────────────────────────────────────────


def test_language_processor_has_retry_helper():
    """
    LanguageProcessor must expose _call_language_detection_llm decorated with @retry.
    """
    from services.language_processor import LanguageProcessor

    lp = LanguageProcessor()
    assert hasattr(lp, "_call_language_detection_llm")
    assert callable(lp._call_language_detection_llm)


def test_response_generator_has_retry_helper():
    """
    ResponseGenerator must expose _call_generation_llm decorated with @retry.
    """
    from services.response_generator import ResponseGenerator

    rg = ResponseGenerator()
    assert hasattr(rg, "_call_generation_llm")
    assert callable(rg._call_generation_llm)


def test_language_detector_retries_on_rate_limit_error():
    """
    _call_language_detection_llm must retry on openai.RateLimitError and eventually raise
    after LLM_RETRY_ATTEMPTS attempts.
    """
    import openai
    from services.language_processor import LanguageProcessor

    lp = LanguageProcessor()
    call_count = 0

    def fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        raise openai.RateLimitError(
            "rate limit", response=MagicMock(status_code=429), body={}
        )

    with patch.object(
        lp.openai_client.chat.completions, "create", side_effect=fake_create
    ):
        with pytest.raises(openai.RateLimitError):
            lp._call_language_detection_llm([{"role": "user", "content": "test"}])

    assert call_count == LLM_RETRY_ATTEMPTS


def test_response_generator_retries_on_rate_limit_error():
    """
    _call_generation_llm must retry on openai.RateLimitError and eventually raise
    after LLM_RETRY_ATTEMPTS attempts.
    """
    import openai
    from services.response_generator import ResponseGenerator

    rg = ResponseGenerator()
    call_count = 0

    def fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        raise openai.RateLimitError(
            "rate limit", response=MagicMock(status_code=429), body={}
        )

    with patch.object(rg.client.chat.completions, "create", side_effect=fake_create):
        with pytest.raises(openai.RateLimitError):
            rg._call_generation_llm(
                [{"role": "user", "content": "test"}], max_tokens=100
            )

    assert call_count == LLM_RETRY_ATTEMPTS


# ── Auth regression ───────────────────────────────────────────────────────────


def test_query_endpoint_requires_api_key():
    """
    POST /query without Authorization header must return HTTP 401.
    """
    unauthenticated_client = TestClient(app)
    response = unauthenticated_client.post("/query", json={"query": "housing query"})
    assert response.status_code == 401


def test_health_endpoint_no_auth_required():
    """
    GET /health must return HTTP 200 without any Authorization header.
    """
    unauthenticated_client = TestClient(app)
    response = unauthenticated_client.get("/health")
    assert response.status_code == 200
