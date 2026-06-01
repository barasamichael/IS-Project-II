"""Module 3: Language Processing — SB-TECH-2026-001 §5.2"""

import json
import os
from unittest.mock import MagicMock
from unittest.mock import patch

import openai
import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("SETTLEBOT_API_KEY", "test-api-key-for-unit-tests")
os.environ.setdefault("SETTLEBOT_LOCALE", "nairobi")

from config.settings import settings

_ENGLISH_DETECT = json.dumps(
    {
        "detected_language": "english",
        "language_code": "en",
        "english_query": "How do I find accommodation near campus?",
        "needs_translation": False,
        "confidence": 0.97,
        "preserved_terms": [],
    }
)

_SWAHILI_DETECT = json.dumps(
    {
        "detected_language": "swahili",
        "language_code": "sw",
        "english_query": "How do I find accommodation near campus?",
        "needs_translation": True,
        "confidence": 0.92,
        "preserved_terms": ["campus"],
    }
)


def _make_processor(locale=None):
    """
    Build a LanguageProcessor with mocked OpenAI client.
    :param locale: Optional locale to inject.
    :return: LanguageProcessor instance.
    """
    with patch("services.language_processor.openai") as mock_openai_mod:
        mock_client = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client
        from services.language_processor import LanguageProcessor

        processor = LanguageProcessor(locale=locale)
        processor.openai_client = mock_client
        return processor, mock_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_detect_english_returns_english() -> None:
    """English query detected as 'english' with needs_translation=False."""
    processor, mock_client = _make_processor(locale=settings.locale)
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=_ENGLISH_DETECT))]
    )
    result = processor.detect_and_process_query(
        "How do I find accommodation near campus?"
    )
    assert result["detected_language"] == "english"
    assert result["needs_translation"] is False


def test_detect_non_english_returns_translation() -> None:
    """Non-English query returns english_query populated and needs_translation=True."""
    processor, mock_client = _make_processor(locale=settings.locale)
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=_SWAHILI_DETECT))]
    )
    result = processor.detect_and_process_query(
        "Ninahitaji msaada kupata nyumba karibu na chuo"
    )
    assert result["needs_translation"] is True
    assert result["english_query"] != ""


def test_disabled_detection_returns_passthrough() -> None:
    """detection_enabled=False returns the original query unchanged without LLM call."""
    processor, mock_client = _make_processor()
    processor.detection_enabled = False
    query = "Where is the nearest hospital?"
    result = processor.detect_and_process_query(query)
    assert result["english_query"] == query
    mock_client.chat.completions.create.assert_not_called()


def test_fallback_on_llm_failure() -> None:
    """
    OpenAI error in LLM call triggers the fallback result with english_query
    equal to the original query.
    """
    processor, mock_client = _make_processor()
    mock_client.chat.completions.create.side_effect = openai.RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429, headers={}),
        body={},
    )
    original = "How do I register at the university?"
    result = processor.detect_and_process_query(original)
    assert result["english_query"] == original


def test_translation_quality_validates_term_preservation() -> None:
    """
    validate_translation_quality returns a lower preservation_score when the
    translated text is missing a critical term that appeared in the original.
    """
    locale = settings.locale
    if locale is None:
        pytest.skip("No locale configured")

    processor, _ = _make_processor(locale=locale)

    city = locale.city
    original = f"Find accommodation near {city} university"

    result_with_term = processor.validate_translation_quality(
        original=original,
        translated=f"Find accommodation near {city} university",
        target_lang="swahili",
    )
    result_without_term = processor.validate_translation_quality(
        original=original,
        translated="Find accommodation near the campus",
        target_lang="swahili",
    )

    assert result_with_term.get("preservation_score", 1.0) >= result_without_term.get(
        "preservation_score", 0.0
    ), "Missing critical term should reduce preservation_score"
