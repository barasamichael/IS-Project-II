"""
Shared pytest fixtures for all SettleBot RAG backend test modules.
All fixtures mock external services (OpenAI, Tavily, ChromaDB) so tests
run without real API calls.
"""

import json
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

# ── Environment ──────────────────────────────────────────────────────────────
os.environ.setdefault("OPENAI_API_KEY", "sk-test-conftest-placeholder")
os.environ.setdefault("SETTLEBOT_API_KEY", "test-conftest-api-key")
os.environ.setdefault("SETTLEBOT_LOCALE", "nairobi")

# ── Paths ─────────────────────────────────────────────────────────────────────
_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_TAVILY_FIXTURE = _FIXTURES_DIR / "tavily_responses.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

_EMBED_DIM = 1536


def _fixed_embedding(query: str) -> np.ndarray:
    """
    Return a deterministic unit-normalised embedding seeded by the query string.
    :param query: str - The query to hash into a vector.
    :return: np.ndarray - Float32 array of shape (_EMBED_DIM,).
    """
    seed = hash(query) % (2**31)
    rng = np.random.default_rng(seed)
    vec = rng.random(_EMBED_DIM).astype(np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else vec


def _make_chat_completion(content: str) -> MagicMock:
    """
    Build a minimal MagicMock that looks like an OpenAI ChatCompletion.
    :param content: str - The assistant message content to return.
    :return: MagicMock - Mock ChatCompletion object.
    """
    mock = MagicMock()
    mock.choices[0].message.content = content
    return mock


def _make_embedding_response(n: int = 1) -> MagicMock:
    """
    Build a minimal MagicMock that looks like an OpenAI Embeddings response.
    :param n: int - Number of embedding objects to include.
    :return: MagicMock - Mock embeddings response.
    """
    mock = MagicMock()
    mock.data = [MagicMock(embedding=np.zeros(_EMBED_DIM).tolist()) for _ in range(n)]
    return mock


# ── Default LLM response bodies ──────────────────────────────────────────────

_LANG_DETECT_JSON = json.dumps(
    {
        "detected_language": "english",
        "language_code": "en",
        "english_query": "Where can I find affordable student housing?",
        "needs_translation": False,
        "confidence": 0.97,
        "preserved_terms": [],
    }
)

_RESPONSE_CONTENT = (
    "## DIRECT ANSWER\n"
    "You can find student housing in several residential areas near campus. "
    "Budget around the local currency equivalent per month.\n\n"
    "## ADDITIONAL INFORMATION\n"
    "Consider proximity to university and transport links when choosing accommodation. "
    "Always inspect the property before signing any agreement.\n\n"
    "## NEXT STEPS\n"
    "1. Contact the university housing office for on-campus options.\n"
    "2. Check the university noticeboard for vetted off-campus listings.\n"
    "3. Visit properties in person before committing.\n"
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def in_memory_chroma():
    """
    Return a chromadb.EphemeralClient pre-seeded with 5 synthetic settlement
    chunks for use in vector database tests.
    :return: Tuple[chromadb.EphemeralClient, chromadb.Collection] -
             (client, collection) pair.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.EphemeralClient(
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = client.create_collection(
        name="settlebot_test",
        metadata={"hnsw:space": "cosine"},
    )

    chunks: List[Dict[str, Any]] = [
        {
            "id": f"chunk_{i:04d}",
            "text": f"International student housing accommodation visa university chunk {i}",
            "metadata": {
                "doc_id": "doc_fixture",
                "chunk_id": f"chunk_{i:04d}",
                "chunk_index": i,
                "settlement_score": 0.8 - i * 0.1,
                "topic_tags": json.dumps(["housing"]),
                "location_entities": json.dumps(["Westlands"] if i == 0 else []),
                "source_url": "https://uonbi.ac.ke/housing",
            },
        }
        for i in range(5)
    ]

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=[_fixed_embedding(c["text"]).tolist() for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )

    return client, collection


@pytest.fixture()
def mock_openai() -> Generator:
    """
    Patch openai.OpenAI globally and configure its return values for language
    detection and response generation calls.
    :return: Generator - Yields the mocked OpenAI class.
    """
    with patch("openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        def _chat_side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            return _make_chat_completion(_RESPONSE_CONTENT)

        mock_client.chat.completions.create.side_effect = _chat_side_effect
        mock_client.embeddings.create.return_value = _make_embedding_response(1)
        yield mock_client


@pytest.fixture()
def mock_embed_service():
    """
    Return a MagicMock EmbeddingService whose embed_query() yields a
    deterministic fixed-dimension vector seeded from the query string.
    :return: MagicMock - Mock EmbeddingService.
    """
    svc = MagicMock()
    svc.dimension = _EMBED_DIM
    svc.model_name = "text-embedding-3-small"
    svc.embed_query.side_effect = lambda q: _fixed_embedding(q) if q else None
    svc.embed_batch_optimized.side_effect = lambda texts, **kw: np.stack(
        [_fixed_embedding(t) for t in texts]
    )
    return svc


@pytest.fixture()
def mock_tavily():
    """
    Patch TavilyClient.search to return fixture results from
    tests/fixtures/tavily_responses.json.
    :return: Generator - Yields the mock TavilyClient instance.
    """
    fixture_data = json.loads(_TAVILY_FIXTURE.read_text(encoding="utf-8"))
    with patch("services.response_generator.TavilyClient", create=True) as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.search.return_value = fixture_data
        yield mock_client


@pytest.fixture()
def api_client():
    """
    Return a FastAPI TestClient with the Authorization header pre-set to the
    test API key. All service calls go through the actual app middleware stack.
    :return: TestClient - Authenticated test client.
    """
    from fastapi.testclient import TestClient
    from api.main import app

    return TestClient(app, headers={"Authorization": "Bearer test-conftest-api-key"})
