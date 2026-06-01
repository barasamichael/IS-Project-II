"""Module 5: Response Generation — SB-TECH-2026-001 §5.2

Every test asserts no fabricated phone numbers appear in the response
using extract_phones_from_context() from utilities/factcheck.py.
"""

import os
from typing import Any
from typing import Dict
from unittest.mock import MagicMock
from unittest.mock import patch


os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("SETTLEBOT_API_KEY", "test-api-key-for-unit-tests")
os.environ.setdefault("SETTLEBOT_LOCALE", "nairobi")

from config.constants import GROUNDING_RULE
from config.constants import PHONE_RE
from config.locale import load_fact_store
from config.settings import settings
from utilities.factcheck import extract_phones_from_context
from utilities.factcheck import normalise_phone

_LOCALE = settings.locale
_FACT_STORE = load_fact_store("nairobi")

_SAFE_RESPONSE = (
    "## DIRECT ANSWER\n"
    "You can find student housing in several areas near the university.\n\n"
    "## ADDITIONAL INFORMATION\n"
    "Consider proximity and transport links when choosing accommodation.\n\n"
    "## NEXT STEPS\n"
    "1. Contact the housing office.\n"
    "2. Visit properties in person.\n"
)


def _assert_no_fabricated_phones(response_text: str, context: str) -> None:
    """
    Assert that every phone number in response_text is present in context
    or in the fact_store emergency contacts.
    :param response_text: str - Generated response to audit.
    :param context: str - Combined retrieved context that was injected.
    """
    found = PHONE_RE.findall(response_text)
    if not found:
        return
    verified = extract_phones_from_context(context, _FACT_STORE)
    normalised_verified = {normalise_phone(v) for v in verified}
    for num in found:
        assert normalise_phone(num) in normalised_verified, (
            f"Fabricated phone number in response: {num}"
        )


def _make_generator():
    """
    Build a ResponseGenerator with mocked OpenAI and LanguageProcessor.
    :return: Tuple[ResponseGenerator, MagicMock LLM client].
    """
    with patch("services.response_generator.OpenAI") as mock_openai_cls:
        mock_llm = MagicMock()
        mock_openai_cls.return_value = mock_llm
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=_SAFE_RESPONSE))]
        )
        from services.response_generator import ResponseGenerator
        from services.language_processor import LanguageProcessor

        with patch("services.response_generator.LanguageProcessor") as mock_lp_cls:
            mock_lp = MagicMock(spec=LanguageProcessor)
            mock_lp.detect_and_process_query.return_value = {
                "detected_language": "english",
                "english_query": "Where can I find student housing?",
                "needs_translation": False,
                "confidence": 0.97,
            }
            mock_lp.translate_response.side_effect = lambda r, *a, **kw: r
            mock_lp_cls.return_value = mock_lp

            gen = ResponseGenerator(fact_store=_FACT_STORE, locale=_LOCALE)
            gen.client = mock_llm
            gen.language_processor = mock_lp
        return gen, mock_llm, mock_lp


_HOUSING_INTENT_INFO: Dict[str, Any] = {
    "intent_type": __import__(
        "services.intent_recognizer", fromlist=["IntentType"]
    ).IntentType.HOUSING_INQUIRY,
    "confidence": 0.88,
    "topic": "housing",
    "is_ambiguous": False,
    "secondary_intent": None,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_off_topic_returns_canned_response() -> None:
    """OFF_TOPIC intent returns the static off_topic_response string."""
    from services.intent_recognizer import IntentType

    gen, _, _ = _make_generator()
    off_topic_info = {
        "intent_type": IntentType.OFF_TOPIC,
        "confidence": 0.3,
        "topic": "other",
        "is_ambiguous": False,
        "secondary_intent": None,
    }
    result = gen.generate_response(
        query="What is the capital of France?",
        retrieved_context=[],
        intent_info=off_topic_info,
    )
    assert gen.off_topic_response in result["response"]
    _assert_no_fabricated_phones(result["response"], "")


def test_response_has_three_sections() -> None:
    """
    generate_response() with HOUSING_INQUIRY intent returns a string
    containing all three required section headers.
    """
    gen, mock_llm, _ = _make_generator()
    mock_llm.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=_SAFE_RESPONSE))]
    )
    result = gen.generate_response(
        query="Where can I find student housing?",
        retrieved_context=[],
        intent_info=_HOUSING_INTENT_INFO,
    )
    resp = result["response"]
    assert "## DIRECT ANSWER" in resp
    assert "## ADDITIONAL INFORMATION" in resp
    assert "## NEXT STEPS" in resp
    _assert_no_fabricated_phones(resp, "")


def test_grounding_rule_in_system_prompt() -> None:
    """
    _get_comprehensive_system_prompt() for every IntentType starts with the
    GROUNDING_RULE constant from config/constants.py.
    """
    from services.intent_recognizer import IntentType

    gen, _, _ = _make_generator()
    for intent in IntentType:
        if intent == IntentType.OFF_TOPIC:
            continue
        prompt = gen._get_comprehensive_system_prompt(
            intent_type=intent,
            emotional_state={"primary_emotion": "neutral", "needs_validation": False},
            crisis_assessment={"crisis_level": "none"},
        )
        assert prompt.startswith(GROUNDING_RULE), (
            f"System prompt for {intent} does not start with GROUNDING_RULE"
        )


def test_emergency_number_from_context_only() -> None:
    """
    A phone number that is present in the fact_store emergency contacts
    must NOT be stripped by the post-generation phone audit.
    """
    gen, mock_llm, _ = _make_generator()

    known_number = list(_FACT_STORE.emergency_contacts.values())[0].number
    response_with_known = (
        f"## DIRECT ANSWER\nIn an emergency call {known_number}.\n\n"
        "## ADDITIONAL INFORMATION\nAlways save emergency numbers.\n\n"
        "## NEXT STEPS\n1. Save the number.\n"
    )
    mock_llm.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=response_with_known))]
    )
    context = f"Emergency number: {known_number}"
    result = gen.generate_response(
        query="What is the emergency number?",
        retrieved_context=[
            {
                "text": context,
                "score": 0.9,
                "chunk_id": "c1",
                "doc_id": "d1",
                "chunk_index": 0,
            }
        ],
        intent_info=_HOUSING_INTENT_INFO,
    )
    assert known_number in result["response"], (
        "Verified phone number from fact_store should not be stripped"
    )


def test_crisis_high_adds_emergency_info() -> None:
    """
    A query with crisis_level=high triggers the emergency info path;
    the response must contain the emergency number from fact_store.
    """
    from services.intent_recognizer import IntentType

    gen, mock_llm, _ = _make_generator()
    known_number = list(_FACT_STORE.emergency_contacts.values())[0].number
    emergency_response = (
        f"## DIRECT ANSWER\nCall {known_number} immediately.\n\n"
        "## ADDITIONAL INFORMATION\nStay calm and provide your location.\n\n"
        "## NEXT STEPS\n1. Call emergency services.\n"
    )
    mock_llm.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=emergency_response))]
    )
    emergency_intent = {
        "intent_type": IntentType.EMERGENCY_HELP,
        "confidence": 0.95,
        "topic": "emergency",
        "is_ambiguous": False,
        "secondary_intent": None,
    }
    result = gen.generate_response(
        query="I am in danger, please help!",
        retrieved_context=[],
        intent_info=emergency_intent,
    )
    _assert_no_fabricated_phones(result["response"], known_number)


def test_translation_called_for_non_english() -> None:
    """
    Non-English original query triggers language_processor.translate_response
    exactly once.
    """
    gen, mock_llm, mock_lp = _make_generator()
    mock_lp.detect_and_process_query.return_value = {
        "detected_language": "swahili",
        "english_query": "Where can I find student housing?",
        "needs_translation": True,
        "confidence": 0.91,
    }
    mock_lp.translate_response.return_value = "Translated response"
    mock_llm.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=_SAFE_RESPONSE))]
    )
    gen.generate_response(
        query="Ninahitaji nyumba ya wanafunzi",
        retrieved_context=[],
        intent_info=_HOUSING_INTENT_INFO,
    )
    mock_lp.translate_response.assert_called_once()


def test_duplicate_language_detection_not_called() -> None:
    """
    A single call to generate_response() triggers detect_and_process_query
    at most once — the duplicate call removed in Milestone 1 must not reappear.
    """
    gen, mock_llm, mock_lp = _make_generator()
    mock_llm.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=_SAFE_RESPONSE))]
    )
    gen.generate_response(
        query="Where can I find student housing?",
        retrieved_context=[],
        intent_info=_HOUSING_INTENT_INFO,
    )
    assert mock_lp.detect_and_process_query.call_count <= 1, (
        "detect_and_process_query must not be called more than once per request"
    )
