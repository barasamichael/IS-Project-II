"""
Tests for Milestone 3 — Request Pipeline Latency Reduction.

Covers: embedding model unification, LRU query cache, shared embedding,
parallel retrieval, intent-aware max_tokens, and Tavily TTL cache.
"""

import os

os.environ.setdefault("SETTLEBOT_API_KEY", "test-secure-key-for-milestone3-tests")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_embedding(dim: int = 1536) -> np.ndarray:
    """Return a deterministic unit vector of the given dimension."""
    v = np.ones(dim, dtype=np.float32)
    return v / np.linalg.norm(v)


def _mock_openai_embedding_response(embedding: np.ndarray):
    """Build a minimal mock that looks like openai.embeddings.create() response."""
    item = MagicMock()
    item.embedding = embedding.tolist()
    response = MagicMock()
    response.data = [item]
    return response


# ---------------------------------------------------------------------------
# Part A — Embedding model unification in IntentRecognizer
# ---------------------------------------------------------------------------


class TestNoAda002InIntentRecognizer:
    def test_no_ada002_string_in_source(self):
        """Verify no hardcoded ada-002 model string remains in intent_recognizer.py."""
        import pathlib

        source = pathlib.Path("services/intent_recognizer.py").read_text()
        assert "ada-002" not in source, (
            "Hardcoded 'ada-002' found in services/intent_recognizer.py"
        )

    def test_intent_recognizer_uses_settings_model(self):
        """
        IntentRecognizer must use settings.embedding.model for prototype embeddings,
        not a hardcoded string.
        """
        fake_embedding = _make_fake_embedding()

        with (
            patch("services.intent_recognizer.OpenAI") as mock_openai_cls,
            patch("services.intent_recognizer.settings") as mock_settings,
        ):
            mock_settings.embedding.model = "text-embedding-3-small"
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.embeddings.create.return_value = (
                _mock_openai_embedding_response(fake_embedding)
            )

            from services.intent_recognizer import IntentRecognizer

            recognizer = IntentRecognizer.__new__(IntentRecognizer)
            recognizer.openai_client = mock_client
            recognizer.cache_dir = pytest.importorskip("pathlib").Path(
                "/tmp/test_ir_cache"
            )
            recognizer.cache_dir.mkdir(exist_ok=True)
            recognizer.cache_file = recognizer.cache_dir / "intent_embeddings.npz"
            recognizer.metadata_file = recognizer.cache_dir / "intent_metadata.json"
            recognizer.intent_patterns = recognizer._initialize_intent_patterns()
            recognizer.off_topic_threshold = 0.40
            recognizer.confidence_threshold = 0.75
            recognizer.settlement_keywords = {
                "high_relevance": [],
                "medium_relevance": [],
                "location_specific": [],
            }

            recognizer._compute_pattern_embeddings()

            # All embedding calls must use settings.embedding.model, not a literal
            for c in mock_client.embeddings.create.call_args_list:
                assert (
                    c.kwargs.get("model") == "text-embedding-3-small"
                    or (c.args and c.args[0] == "text-embedding-3-small")
                    or c[1].get("model") == "text-embedding-3-small"
                ), f"Unexpected model in embeddings.create call: {c}"


# ---------------------------------------------------------------------------
# Part B — Cache model-name validation
# ---------------------------------------------------------------------------


class TestIntentCacheModelValidation:
    def test_cache_invalidated_on_model_change(self, tmp_path):
        """
        _cache_is_valid() must return False when the stored model name differs
        from settings.embedding.model.
        """
        import json
        from services.intent_recognizer import IntentType

        metadata = {
            "intent_types": [i.value for i in IntentType if i.value != "off_topic"],
            "embedding_model": "text-embedding-ada-002",
            "embedding_dim": 1536,
        }
        meta_file = tmp_path / "intent_metadata.json"
        meta_file.write_text(json.dumps(metadata))
        cache_file = tmp_path / "intent_embeddings.npz"
        # Create a minimal npz so the file-exists check passes
        np.savez(str(cache_file), dummy=np.array([1.0]))

        with (
            patch("services.intent_recognizer.OpenAI"),
            patch("services.intent_recognizer.settings") as mock_settings,
        ):
            mock_settings.embedding.model = "text-embedding-3-small"

            from services.intent_recognizer import IntentRecognizer

            recognizer = IntentRecognizer.__new__(IntentRecognizer)
            recognizer.openai_client = MagicMock()
            recognizer.cache_dir = tmp_path
            recognizer.cache_file = cache_file
            recognizer.metadata_file = meta_file
            recognizer.intent_patterns = {}
            recognizer.off_topic_threshold = 0.40
            recognizer.confidence_threshold = 0.75
            recognizer.settlement_keywords = {
                "high_relevance": [],
                "medium_relevance": [],
                "location_specific": [],
            }

            assert recognizer._cache_is_valid() is False

    def test_cache_valid_when_model_matches(self, tmp_path):
        """_cache_is_valid() returns True when stored model matches settings."""
        import json
        from services.intent_recognizer import IntentType

        current_model = "text-embedding-3-small"
        metadata = {
            "intent_types": [i.value for i in IntentType if i.value != "off_topic"],
            "embedding_model": current_model,
            "embedding_dim": 1536,
        }
        meta_file = tmp_path / "intent_metadata.json"
        meta_file.write_text(json.dumps(metadata))
        cache_file = tmp_path / "intent_embeddings.npz"
        np.savez(str(cache_file), dummy=np.array([1.0]))

        with (
            patch("services.intent_recognizer.OpenAI"),
            patch("services.intent_recognizer.settings") as mock_settings,
        ):
            mock_settings.embedding.model = current_model

            from services.intent_recognizer import IntentRecognizer

            recognizer = IntentRecognizer.__new__(IntentRecognizer)
            recognizer.openai_client = MagicMock()
            recognizer.cache_dir = tmp_path
            recognizer.cache_file = cache_file
            recognizer.metadata_file = meta_file
            recognizer.intent_patterns = {}
            recognizer.off_topic_threshold = 0.40
            recognizer.confidence_threshold = 0.75
            recognizer.settlement_keywords = {
                "high_relevance": [],
                "medium_relevance": [],
                "location_specific": [],
            }

            assert recognizer._cache_is_valid() is True


# ---------------------------------------------------------------------------
# Part C — embed_query LRU cache
# ---------------------------------------------------------------------------


class TestEmbedQueryLRUCache:
    def test_cache_prevents_regeneration(self):
        """
        Calling embed_query() twice with the same string results in exactly one
        underlying OpenAI API call.
        """
        fake_embedding = _make_fake_embedding()

        with (
            patch("services.embeddings.OpenAI") as mock_openai_cls,
            patch("services.embeddings.settings") as mock_settings,
            patch(
                "services.embeddings.ROOT_DIR",
                new=pytest.importorskip("pathlib").Path("/tmp"),
            ),
        ):
            mock_settings.embedding.model = "text-embedding-3-small"
            mock_settings.embedding.dimension = 1536
            mock_settings.deduplication.enabled = False
            mock_settings.deduplication.similarity_threshold = 0.92
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.embeddings.create.return_value = (
                _mock_openai_embedding_response(fake_embedding)
            )

            from services.embeddings import EmbeddingService

            svc = EmbeddingService.__new__(EmbeddingService)
            svc.client = mock_client
            svc.model_name = "text-embedding-3-small"
            svc.dimension = 1536

            from cachetools import LRUCache
            from config.constants import EMBEDDING_LRU_CACHE_SIZE

            svc._query_embedding_cache = LRUCache(maxsize=EMBEDDING_LRU_CACHE_SIZE)
            svc.settlement_keywords = []

            query = "Where can I find student housing in Nairobi?"
            svc.embed_query(query)
            svc.embed_query(query)

            assert mock_client.embeddings.create.call_count == 1, (
                f"Expected 1 API call, got {mock_client.embeddings.create.call_count}"
            )

    def test_embed_query_empty_returns_none_no_api_call(self):
        """Empty query returns None without calling the OpenAI API."""
        with (
            patch("services.embeddings.OpenAI") as mock_openai_cls,
            patch("services.embeddings.settings") as mock_settings,
            patch(
                "services.embeddings.ROOT_DIR",
                new=pytest.importorskip("pathlib").Path("/tmp"),
            ),
        ):
            mock_settings.embedding.model = "text-embedding-3-small"
            mock_settings.embedding.dimension = 1536
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client

            from services.embeddings import EmbeddingService
            from cachetools import LRUCache
            from config.constants import EMBEDDING_LRU_CACHE_SIZE

            svc = EmbeddingService.__new__(EmbeddingService)
            svc.client = mock_client
            svc.model_name = "text-embedding-3-small"
            svc.dimension = 1536
            svc._query_embedding_cache = LRUCache(maxsize=EMBEDDING_LRU_CACHE_SIZE)
            svc.settlement_keywords = []

            result = svc.embed_query("")
            assert result is None
            mock_client.embeddings.create.assert_not_called()


# ---------------------------------------------------------------------------
# Part D — VectorDBService accepts pre-computed embedding
# ---------------------------------------------------------------------------


class TestVectorDBPreComputedEmbedding:
    def test_search_uses_provided_embedding(self):
        """
        When an embedding is passed to search(), embed_query() must not be called.
        """
        fake_embedding = _make_fake_embedding()

        with (
            patch("services.vector_db.EmbeddingService") as mock_emb_cls,
            patch("services.vector_db.settings") as mock_settings,
            patch(
                "services.vector_db.ROOT_DIR",
                new=pytest.importorskip("pathlib").Path("/tmp"),
            ),
            patch("services.vector_db.chromadb") as mock_chroma,
        ):
            mock_settings.vector_db.collection_name = "test_col"
            mock_settings.embedding.dimension = 1536
            mock_emb_instance = MagicMock()
            mock_emb_instance.dimension = 1536
            mock_emb_cls.return_value = mock_emb_instance

            mock_collection = MagicMock()
            mock_collection.count.return_value = 1
            mock_collection.query.return_value = {
                "ids": [["chunk_1"]],
                "documents": [["Some housing info"]],
                "distances": [[0.1]],
                "metadatas": [[{"doc_id": "d1", "chunk_id": "c1", "chunk_index": 0}]],
            }
            mock_chroma.PersistentClient.return_value.get_collection.return_value = (
                mock_collection
            )

            from services.vector_db import VectorDBService

            svc = VectorDBService.__new__(VectorDBService)
            svc.embedding_service = mock_emb_instance
            svc.dimension = 1536
            svc.collection_name = "test_col"
            svc.collection = mock_collection
            svc.topic_weights = {}
            svc.location_boost = {}

            svc.search("housing query", top_k=5, embedding=fake_embedding)

            mock_emb_instance.embed_query.assert_not_called()
            mock_collection.query.assert_called_once()


# ---------------------------------------------------------------------------
# Part E — Intent-aware max_tokens
# ---------------------------------------------------------------------------


class TestIntentAwareMaxTokens:
    def _make_generator(self, mock_llm_call):
        """Build a ResponseGenerator with mocked LLM call."""
        with (
            patch("services.response_generator.OpenAI"),
            patch("services.response_generator.LanguageProcessor"),
            patch("services.response_generator.settings") as mock_settings,
        ):
            mock_settings.llm.temperature = 0.2
            mock_settings.llm.max_tokens = 4096
            mock_settings.llm.model = "gpt-4.1-mini"

            from services.response_generator import ResponseGenerator

            gen = ResponseGenerator.__new__(ResponseGenerator)
            gen.client = MagicMock()
            gen.temperature = 0.2
            gen.max_tokens = 4096
            gen.model = "gpt-4.1-mini"
            gen.language_processor = MagicMock()
            gen.language_processor.detect_and_process_query.return_value = {
                "english_query": "test",
                "detected_language": "english",
                "needs_translation": False,
            }
            gen.min_context_relevance = 0.3
            gen.min_chunks_for_response = 1
            gen.essential_info = {
                "emergency_numbers": {},
                "key_hospitals": {},
                "universities": {},
                "immigration_office": {},
            }
            gen.empathy_responses = {}
            gen.safety_protocols = {"general": []}
            gen.off_topic_response = "off-topic"
            gen._call_generation_llm = mock_llm_call
            return gen

    def test_emergency_intent_uses_emergency_max_tokens(self):
        """EMERGENCY_HELP intent must pass LLM_EMERGENCY_MAX_TOKENS to _call_generation_llm."""
        from config.constants import LLM_EMERGENCY_MAX_TOKENS
        from services.intent_recognizer import IntentType, TopicType

        captured = {}

        def fake_llm(messages, max_tokens):
            captured["max_tokens"] = max_tokens
            return "## DIRECT ANSWER\nhelp\n## ADDITIONAL INFORMATION\nok\n## NEXT STEPS\n1. call 999"

        gen = self._make_generator(fake_llm)
        gen.language_processor.detect_and_process_query.return_value = {
            "english_query": "emergency help",
            "detected_language": "english",
            "needs_translation": False,
        }

        intent_info = {
            "intent_type": IntentType.EMERGENCY_HELP,
            "topic": TopicType.EMERGENCY,
            "confidence": 0.9,
            "settlement_relevance": 0.8,
            "semantic_scores": {},
            "off_topic_indicators": [],
            "classification_method": "semantic_embedding",
            "is_off_topic": False,
        }

        gen.generate_response(
            query="I need emergency help",
            retrieved_context=[],
            intent_info=intent_info,
            web_info=None,
        )

        assert captured.get("max_tokens") == LLM_EMERGENCY_MAX_TOKENS, (
            f"Expected {LLM_EMERGENCY_MAX_TOKENS}, got {captured.get('max_tokens')}"
        )

    def test_non_emergency_intent_uses_standard_max_tokens(self):
        """Non-emergency intent must pass LLM_RESPONSE_MAX_TOKENS to _call_generation_llm."""
        from config.constants import LLM_RESPONSE_MAX_TOKENS
        from services.intent_recognizer import IntentType, TopicType

        captured = {}

        def fake_llm(messages, max_tokens):
            captured["max_tokens"] = max_tokens
            return "## DIRECT ANSWER\nhousing\n## ADDITIONAL INFORMATION\nok\n## NEXT STEPS\n1. look"

        gen = self._make_generator(fake_llm)

        intent_info = {
            "intent_type": IntentType.HOUSING_INQUIRY,
            "topic": TopicType.HOUSING,
            "confidence": 0.9,
            "settlement_relevance": 0.8,
            "semantic_scores": {},
            "off_topic_indicators": [],
            "classification_method": "semantic_embedding",
            "is_off_topic": False,
        }

        gen.generate_response(
            query="Where can I find housing?",
            retrieved_context=[],
            intent_info=intent_info,
            web_info=None,
        )

        assert captured.get("max_tokens") == LLM_RESPONSE_MAX_TOKENS, (
            f"Expected {LLM_RESPONSE_MAX_TOKENS}, got {captured.get('max_tokens')}"
        )


# ---------------------------------------------------------------------------
# Part F — Tavily TTL cache
# ---------------------------------------------------------------------------


class TestTavilyCacheBehaviour:
    def _get_fresh_cache(self):
        """Return a fresh TTLCache and patch the module-level one."""
        from cachetools import TTLCache

        return TTLCache(maxsize=200, ttl=3600)

    def test_tavily_cache_hit_skips_api_call(self):
        """
        Two calls with the same (intent_type, query) result in exactly one Tavily
        API call; the second is served from the cache.
        """
        from services.intent_recognizer import IntentType
        import services.response_generator as rg_module

        fresh_cache = self._get_fresh_cache()
        original_cache = rg_module._tavily_cache

        try:
            rg_module._tavily_cache = fresh_cache

            from services.response_generator import ResponseGenerator

            gen = ResponseGenerator.__new__(ResponseGenerator)
            gen.client = MagicMock()
            gen.temperature = 0.2
            gen.max_tokens = 2048
            gen.model = "gpt-4.1-mini"
            gen.language_processor = MagicMock()
            gen.essential_info = {}
            gen.empathy_responses = {}
            gen.safety_protocols = {}

            mock_tavily_client = MagicMock()
            mock_tavily_client.search.return_value = {
                "results": [{"title": "t", "content": "c", "url": "u", "score": 0.9}],
                "answer": "summary",
            }

            with (
                patch(
                    "tavily.TavilyClient", return_value=mock_tavily_client
                ) as mock_cls,
                patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
            ):
                gen.search_web_for_current_info(
                    "How much is rent in Nairobi?", IntentType.COST_INQUIRY
                )
                gen.search_web_for_current_info(
                    "How much is rent in Nairobi?", IntentType.COST_INQUIRY
                )

            # TavilyClient should be instantiated exactly once; the second
            # search_web_for_current_info call is served from the cache.
            assert mock_cls.call_count == 1, (
                f"Expected TavilyClient instantiated once (cache hit on 2nd call), "
                f"got {mock_cls.call_count}"
            )

        finally:
            rg_module._tavily_cache = original_cache

    def test_tavily_cache_bypassed_for_crisis(self):
        """
        Two calls with the same query but crisis_level='high' both reach
        the Tavily API — cache is bypassed.
        """
        from services.intent_recognizer import IntentType
        import services.response_generator as rg_module

        fresh_cache = self._get_fresh_cache()
        original_cache = rg_module._tavily_cache

        try:
            rg_module._tavily_cache = fresh_cache

            from services.response_generator import ResponseGenerator

            gen = ResponseGenerator.__new__(ResponseGenerator)
            gen.client = MagicMock()
            gen.temperature = 0.2
            gen.max_tokens = 4096
            gen.model = "gpt-4.1-mini"
            gen.language_processor = MagicMock()
            gen.essential_info = {}
            gen.empathy_responses = {}
            gen.safety_protocols = {}

            mock_tavily_client = MagicMock()
            mock_tavily_client.search.return_value = {
                "results": [{"title": "t", "content": "c", "url": "u", "score": 0.9}],
                "answer": "summary",
            }

            with (
                patch(
                    "tavily.TavilyClient", return_value=mock_tavily_client
                ) as mock_cls,
                patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
            ):
                gen.search_web_for_current_info(
                    "I need emergency help now",
                    IntentType.EMERGENCY_HELP,
                    crisis_level="high",
                )
                gen.search_web_for_current_info(
                    "I need emergency help now",
                    IntentType.EMERGENCY_HELP,
                    crisis_level="high",
                )

            # crisis_level="high" must bypass the cache — TavilyClient
            # instantiated twice, once per call.
            assert mock_cls.call_count == 2, (
                f"Expected TavilyClient instantiated twice (crisis bypasses cache), "
                f"got {mock_cls.call_count}"
            )

        finally:
            rg_module._tavily_cache = original_cache


# ---------------------------------------------------------------------------
# Part G — Constants exist
# ---------------------------------------------------------------------------


class TestNewConstantsExist:
    def test_constants_defined(self):
        """All five Milestone 3 constants must be present in config.constants."""
        from config.constants import (
            LLM_RESPONSE_MAX_TOKENS,
            LLM_EMERGENCY_MAX_TOKENS,
            EMBEDDING_LRU_CACHE_SIZE,
            TAVILY_CACHE_TTL,
            TAVILY_CACHE_MAXSIZE,
        )

        assert LLM_RESPONSE_MAX_TOKENS == 2048
        assert LLM_EMERGENCY_MAX_TOKENS == 4096
        assert EMBEDDING_LRU_CACHE_SIZE == 512
        assert TAVILY_CACHE_TTL == 3600
        assert TAVILY_CACHE_MAXSIZE == 200
