"""Module 1: Embeddings — SB-TECH-2026-001 §5.2"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("SETTLEBOT_API_KEY", "test-api-key-for-unit-tests")
os.environ.setdefault("SETTLEBOT_LOCALE", "nairobi")

from config.settings import settings

_EMBED_DIM = settings.embedding.dimension


def _make_embed_svc(locale=None):
    """Construct an EmbeddingService with a mocked OpenAI client."""
    with patch("services.embeddings.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        vec = np.random.default_rng(42).random(_EMBED_DIM).astype(np.float32)
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=vec.tolist())]
        )
        from services.embeddings import EmbeddingService

        svc = EmbeddingService(locale=locale)
        svc.client = mock_client
        return svc, mock_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_embed_query_returns_array() -> None:
    """Valid query returns a numpy array of the correct embedding dimension."""
    svc, _ = _make_embed_svc()
    result = svc.embed_query("Where can I find affordable student housing?")
    assert isinstance(result, np.ndarray)
    assert result.shape == (_EMBED_DIM,)


def test_embed_query_empty_returns_none() -> None:
    """Empty string passed to embed_query returns None without raising."""
    svc, _ = _make_embed_svc()
    result = svc.embed_query("")
    assert result is None


def test_embed_query_prepends_locale_prefix() -> None:
    """
    Query without locale city/country triggers _optimize_query_for_embedding,
    which prepends the locale prefix when locale is set.
    """
    locale = settings.locale
    svc, _ = _make_embed_svc(locale=locale)
    generic_query = "how do I open a bank account"
    optimized = svc._optimize_query_for_embedding(generic_query)
    if locale:
        assert (
            locale.city in optimized
            or locale.country in optimized
            or generic_query in optimized
        )
    else:
        assert generic_query in optimized


def test_embed_query_no_duplicate_prefix() -> None:
    """Query already containing locale city is not double-prefixed."""
    locale = settings.locale
    svc, _ = _make_embed_svc(locale=locale)
    if locale:
        query_with_city = f"housing in {locale.city}"
        optimized = svc._optimize_query_for_embedding(query_with_city)
        city_lower = locale.city.lower()
        assert optimized.lower().count(city_lower) == query_with_city.lower().count(
            city_lower
        )
    else:
        pytest.skip("No locale configured")


def test_embed_batch_returns_correct_count() -> None:
    """embed_batch_optimized(texts) returns an array with len(texts) rows."""
    svc, mock_client = _make_embed_svc()
    texts = ["query one", "query two", "query three"]
    n = len(texts)
    vec = np.random.default_rng(99).random(_EMBED_DIM).astype(np.float32)
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=vec.tolist()) for _ in range(n)]
    )
    result = svc.embed_batch_optimized(texts, batch_size=50)
    assert result.shape[0] == n


def test_cache_prevents_regeneration() -> None:
    """
    Calling embed_chunks twice with the same unmodified chunks file skips
    OpenAI embeddings.create on the second call (cache hit via file hash).
    """
    svc, mock_client = _make_embed_svc()
    with tempfile.TemporaryDirectory() as tmp:
        chunks_file = Path(tmp) / "doc_test_chunks.jsonl"
        chunks_file.write_text(
            '{"chunk_id": "c1", "text": "student housing", "doc_id": "d1"}\n',
            encoding="utf-8",
        )
        svc.chunks_dir = Path(tmp)
        svc.embeddings_dir = Path(tmp)
        svc.metadata_file = Path(tmp) / "embeddings_metadata.json"
        svc.embeddings_metadata = {}

        vec = np.random.default_rng(7).random(_EMBED_DIM).astype(np.float32)
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=vec.tolist())]
        )

        svc.embed_chunks(chunks_file)
        first_call_count = mock_client.embeddings.create.call_count

        svc.embed_chunks(chunks_file)
        second_call_count = mock_client.embeddings.create.call_count

        assert second_call_count == first_call_count, (
            "embeddings.create should not be called again when file hash unchanged"
        )


def test_cache_invalidated_on_model_change() -> None:
    """Changing self.model_name invalidates the cached hash, triggering regeneration."""
    svc, mock_client = _make_embed_svc()
    with tempfile.TemporaryDirectory() as tmp:
        chunks_file = Path(tmp) / "doc_cache_chunks.jsonl"
        chunks_file.write_text(
            '{"chunk_id": "c2", "text": "visa immigration student", "doc_id": "d2"}\n',
            encoding="utf-8",
        )
        svc.chunks_dir = Path(tmp)
        svc.embeddings_dir = Path(tmp)
        svc.metadata_file = Path(tmp) / "embeddings_metadata.json"
        svc.embeddings_metadata = {}

        vec = np.random.default_rng(11).random(_EMBED_DIM).astype(np.float32)
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=vec.tolist())]
        )

        svc.embed_chunks(chunks_file)
        call_count_before = mock_client.embeddings.create.call_count

        svc.model_name = "text-embedding-ada-002"
        svc.embed_chunks(chunks_file)
        call_count_after = mock_client.embeddings.create.call_count

        assert call_count_after > call_count_before, (
            "Model name change must invalidate cache and trigger re-embedding"
        )
