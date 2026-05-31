"""
Tests for Milestone 5 — Web Search Robustness.

Covers all ten invariant targets:
 1. CULTURAL_ADAPTATION does not trigger search_web_for_current_info()
 2. EMERGENCY_HELP does not trigger search_web_for_current_info()
 3. IMMIGRATION_VISA always triggers search_web_for_current_info()
 4. HOUSING_INQUIRY with high-confidence chunk does not trigger search
 5. HOUSING_INQUIRY with low-confidence chunk triggers search
 6. Trusted-domain results are retained in search output
 7. All-untrusted results cause search_web_for_current_info() to return None
 8. LocaleFactStore has a trusted_domains field
 9. load_fact_store("nairobi") returns a non-empty trusted_domains list
10. WEB_SEARCH_CONFIDENCE_THRESHOLD is 0.5 in config/constants.py
"""

import os

import pytest

from unittest.mock import MagicMock
from unittest.mock import patch

os.environ.setdefault("SETTLEBOT_API_KEY", "test-secure-key-milestone5")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder-milestone5")


# ---------------------------------------------------------------------------
# Invariant 10 — constant value
# ---------------------------------------------------------------------------


def test_web_search_confidence_threshold_value():
    """WEB_SEARCH_CONFIDENCE_THRESHOLD must exist in constants and equal 0.5."""
    from config.constants import WEB_SEARCH_CONFIDENCE_THRESHOLD

    assert WEB_SEARCH_CONFIDENCE_THRESHOLD == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Invariants 8 & 9 — LocaleFactStore.trusted_domains
# ---------------------------------------------------------------------------


def test_locale_fact_store_has_trusted_domains_field():
    """LocaleFactStore must expose a trusted_domains: List[str] field."""
    from config.locale import LocaleFactStore

    fields = LocaleFactStore.model_fields
    assert "trusted_domains" in fields, (
        "LocaleFactStore must declare a trusted_domains field."
    )


def test_load_fact_store_nairobi_has_trusted_domains():
    """load_fact_store('nairobi') must return a non-empty trusted_domains list."""
    from config.locale import load_fact_store

    fs = load_fact_store("nairobi")
    assert isinstance(fs.trusted_domains, list)
    assert len(fs.trusted_domains) > 0, "trusted_domains must not be empty."
    assert "immigration.go.ke" in fs.trusted_domains
    assert "knh.or.ke" in fs.trusted_domains


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_generator():
    """Build a ResponseGenerator with all attributes mocked for unit tests."""
    from config.locale import load_fact_store
    from services.response_generator import ResponseGenerator

    gen = ResponseGenerator.__new__(ResponseGenerator)
    gen.fact_store = load_fact_store("nairobi")
    gen.min_context_relevance = 0.3
    gen.min_chunks_for_response = 1
    gen.language_processor = MagicMock()
    gen.language_processor.detect_and_process_query.return_value = {
        "english_query": "test query",
        "detected_language": "english",
        "needs_translation": False,
    }
    gen.empathy_responses = {
        "neutral": [],
        "stress": ["Understandable."],
        "anxiety": ["Valid concerns."],
        "urgency": ["Immediate steps."],
        "confusion": ["Let me clarify."],
    }
    gen.safety_protocols = {"general": ["Stay aware."]}
    gen.off_topic_response = (
        "## DIRECT ANSWER\nOff topic.\n"
        "## ADDITIONAL INFORMATION\nN/A\n"
        "## NEXT STEPS\n1. Ask a settlement question."
    )
    gen._call_generation_llm = MagicMock(
        return_value=(
            "## DIRECT ANSWER\nHere is the info.\n"
            "## ADDITIONAL INFORMATION\nMore.\n"
            "## NEXT STEPS\n1. Act now."
        )
    )
    return gen


def _intent_info(intent_type, topic_type):
    """Build a minimal intent_info dict."""
    return {
        "intent_type": intent_type,
        "topic": topic_type,
        "confidence": 0.9,
        "settlement_relevance": 0.8,
        "semantic_scores": {},
        "off_topic_indicators": [],
        "classification_method": "semantic_embedding",
        "is_off_topic": False,
    }


def _chunk(score: float) -> dict:
    return {
        "text": "Some context text.",
        "score": score,
        "doc_id": "doc-001",
        "chunk_id": "chunk-001",
    }


# ---------------------------------------------------------------------------
# Invariants 1 & 2 — never-search intents
# ---------------------------------------------------------------------------


def test_cultural_adaptation_skips_web_search():
    """
    generate_response() with CULTURAL_ADAPTATION must not call
    search_web_for_current_info().
    """
    from services.intent_recognizer import IntentType
    from services.intent_recognizer import TopicType

    gen = _make_generator()
    with patch.object(gen, "search_web_for_current_info") as mock_search:
        gen.generate_response(
            query="how do I adapt to Kenyan culture",
            retrieved_context=[],
            intent_info=_intent_info(IntentType.CULTURAL_ADAPTATION, TopicType.CULTURE),
            web_info=None,
        )
    mock_search.assert_not_called()


def test_emergency_help_skips_web_search():
    """
    generate_response() with EMERGENCY_HELP must not call
    search_web_for_current_info().
    """
    from services.intent_recognizer import IntentType
    from services.intent_recognizer import TopicType

    gen = _make_generator()
    with patch.object(gen, "search_web_for_current_info") as mock_search:
        gen.generate_response(
            query="I need emergency help now",
            retrieved_context=[],
            intent_info=_intent_info(IntentType.EMERGENCY_HELP, TopicType.EMERGENCY),
            web_info=None,
        )
    mock_search.assert_not_called()


# ---------------------------------------------------------------------------
# Invariant 3 — always-search intent
# ---------------------------------------------------------------------------


def test_immigration_intent_triggers_web_search():
    """
    generate_response() with IMMIGRATION_VISA must call
    search_web_for_current_info() exactly once regardless of chunk scores.
    """
    from services.intent_recognizer import IntentType
    from services.intent_recognizer import TopicType

    gen = _make_generator()
    with patch.object(
        gen, "search_web_for_current_info", return_value=None
    ) as mock_search:
        gen.generate_response(
            query="how do I renew my student visa",
            retrieved_context=[_chunk(0.9)],
            intent_info=_intent_info(IntentType.IMMIGRATION_VISA, TopicType.LEGAL),
            web_info=None,
        )
    mock_search.assert_called_once()


def test_cost_inquiry_triggers_web_search_even_with_high_confidence():
    """
    COST_INQUIRY must always trigger web search, even when RAG confidence is high.
    """
    from services.intent_recognizer import IntentType
    from services.intent_recognizer import TopicType

    gen = _make_generator()
    with patch.object(
        gen, "search_web_for_current_info", return_value=None
    ) as mock_search:
        gen.generate_response(
            query="what is the cost of living in Nairobi",
            retrieved_context=[_chunk(0.95)],
            intent_info=_intent_info(IntentType.COST_INQUIRY, TopicType.FINANCE),
            web_info=None,
        )
    mock_search.assert_called_once()


# ---------------------------------------------------------------------------
# Invariants 4 & 5 — conditional intents
# ---------------------------------------------------------------------------


def test_housing_high_confidence_skips_search():
    """
    HOUSING_INQUIRY with top chunk score >= 0.5 must not trigger web search.
    """
    from services.intent_recognizer import IntentType
    from services.intent_recognizer import TopicType

    gen = _make_generator()
    with patch.object(gen, "search_web_for_current_info") as mock_search:
        gen.generate_response(
            query="where can I find student housing",
            retrieved_context=[_chunk(0.8)],
            intent_info=_intent_info(IntentType.HOUSING_INQUIRY, TopicType.HOUSING),
            web_info=None,
        )
    mock_search.assert_not_called()


def test_housing_low_confidence_triggers_search():
    """
    HOUSING_INQUIRY with top chunk score < 0.5 must trigger web search.
    """
    from services.intent_recognizer import IntentType
    from services.intent_recognizer import TopicType

    gen = _make_generator()
    with patch.object(
        gen, "search_web_for_current_info", return_value=None
    ) as mock_search:
        gen.generate_response(
            query="where can I find student housing",
            retrieved_context=[_chunk(0.3)],
            intent_info=_intent_info(IntentType.HOUSING_INQUIRY, TopicType.HOUSING),
            web_info=None,
        )
    mock_search.assert_called_once()


def test_housing_empty_context_triggers_search():
    """
    HOUSING_INQUIRY with no retrieved context (score defaults to 0) triggers search.
    """
    from services.intent_recognizer import IntentType
    from services.intent_recognizer import TopicType

    gen = _make_generator()
    with patch.object(
        gen, "search_web_for_current_info", return_value=None
    ) as mock_search:
        gen.generate_response(
            query="where can I find student housing",
            retrieved_context=[],
            intent_info=_intent_info(IntentType.HOUSING_INQUIRY, TopicType.HOUSING),
            web_info=None,
        )
    mock_search.assert_called_once()


# ---------------------------------------------------------------------------
# Invariants 6 & 7 — domain filter inside search_web_for_current_info()
# ---------------------------------------------------------------------------


def _build_search_gen():
    """Build a generator wired for search_web_for_current_info() unit tests."""
    from config.locale import load_fact_store
    from services.response_generator import ResponseGenerator

    gen = ResponseGenerator.__new__(ResponseGenerator)
    gen.fact_store = load_fact_store("nairobi")
    return gen


def _make_tavily_result(url: str, content: str = "useful info") -> dict:
    return {"title": "Title", "content": content, "url": url, "score": 0.8}


def test_trusted_domain_result_retained():
    """
    search_web_for_current_info() must include results from trusted domains
    in the returned content.
    """
    from services.intent_recognizer import IntentType

    gen = _build_search_gen()
    trusted_result = _make_tavily_result(
        "https://immigration.go.ke/visa-info", "Visa renewal takes 14 days."
    )
    mock_search_result = {
        "results": [trusted_result],
        "answer": "Visa takes 14 days.",
    }

    with (
        patch("tavily.TavilyClient") as MockClient,
        patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
    ):
        MockClient.return_value.search.return_value = mock_search_result
        result = gen.search_web_for_current_info(
            "how to renew visa", IntentType.IMMIGRATION_VISA
        )

    assert result is not None
    assert result["search_successful"] is True
    combined = " ".join(r["content"] for r in result["results"])
    assert "Visa renewal" in combined or "14 days" in combined


def test_untrusted_domain_result_filtered():
    """
    search_web_for_current_info() with only untrusted-domain results must
    return a dict with search_successful=False, equivalent to no results.
    """
    from services.intent_recognizer import IntentType
    import services.response_generator as rg_module
    from cachetools import TTLCache

    gen = _build_search_gen()
    untrusted_result = _make_tavily_result(
        "https://random-blog.example.com/nairobi-visa", "Some blog post."
    )
    mock_search_result = {
        "results": [untrusted_result],
        "answer": "",
    }

    fresh_cache = TTLCache(maxsize=200, ttl=3600)
    original_cache = rg_module._tavily_cache
    try:
        rg_module._tavily_cache = fresh_cache
        with (
            patch("tavily.TavilyClient") as MockClient,
            patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
        ):
            MockClient.return_value.search.return_value = mock_search_result
            result = gen.search_web_for_current_info(
                "nairobi visa query unique xyz123", IntentType.IMMIGRATION_VISA
            )
    finally:
        rg_module._tavily_cache = original_cache

    # All results filtered → web_results empty → search_successful False
    assert result is not None
    assert result["search_successful"] is False


def test_mixed_domains_only_trusted_retained():
    """
    With one trusted and one untrusted result, only the trusted result's
    content must appear in the assembled output.
    """
    from services.intent_recognizer import IntentType
    import services.response_generator as rg_module
    from cachetools import TTLCache

    gen = _build_search_gen()
    trusted_result = _make_tavily_result(
        "https://knh.or.ke/services", "KNH specialist services available."
    )
    untrusted_result = _make_tavily_result(
        "https://spamsite.bad/nairobi", "Spam content here."
    )
    mock_search_result = {
        "results": [trusted_result, untrusted_result],
        "answer": "",
    }

    fresh_cache = TTLCache(maxsize=200, ttl=3600)
    original_cache = rg_module._tavily_cache
    try:
        rg_module._tavily_cache = fresh_cache
        with (
            patch("tavily.TavilyClient") as MockClient,
            patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
        ):
            MockClient.return_value.search.return_value = mock_search_result
            result = gen.search_web_for_current_info(
                "knh hospital query unique abc987", IntentType.HEALTHCARE
            )
    finally:
        rg_module._tavily_cache = original_cache

    assert result is not None
    assert result["search_successful"] is True
    combined = " ".join(r["content"] for r in result["results"])
    assert "KNH specialist" in combined
    assert "Spam content" not in combined


# ---------------------------------------------------------------------------
# _should_search_web() unit tests
# ---------------------------------------------------------------------------


def test_should_search_web_never_intents():
    """_should_search_web returns False for all never-search intents."""
    from services.intent_recognizer import IntentType

    gen = _build_search_gen()
    for intent in (IntentType.CULTURAL_ADAPTATION, IntentType.EMERGENCY_HELP):
        assert gen._should_search_web(intent, []) is False, (
            f"_should_search_web must return False for {intent}"
        )


def test_should_search_web_always_intents():
    """_should_search_web returns True for always-search intents regardless of chunks."""
    from services.intent_recognizer import IntentType

    gen = _build_search_gen()
    for intent in (IntentType.IMMIGRATION_VISA, IntentType.COST_INQUIRY):
        assert gen._should_search_web(intent, [_chunk(0.99)]) is True, (
            f"_should_search_web must return True for {intent} even with high chunk score"
        )


def test_should_search_web_conditional_above_threshold():
    """_should_search_web returns False for conditional intents when score >= threshold."""
    from services.intent_recognizer import IntentType

    gen = _build_search_gen()
    assert gen._should_search_web(IntentType.HOUSING_INQUIRY, [_chunk(0.7)]) is False


def test_should_search_web_conditional_below_threshold():
    """_should_search_web returns True for conditional intents when score < threshold."""
    from services.intent_recognizer import IntentType

    gen = _build_search_gen()
    assert gen._should_search_web(IntentType.HOUSING_INQUIRY, [_chunk(0.2)]) is True


def test_should_search_web_deterministic():
    """Calling _should_search_web twice with identical inputs yields the same result."""
    from services.intent_recognizer import IntentType

    gen = _build_search_gen()
    r1 = gen._should_search_web(IntentType.HOUSING_INQUIRY, [_chunk(0.3)])
    r2 = gen._should_search_web(IntentType.HOUSING_INQUIRY, [_chunk(0.3)])
    assert r1 == r2
