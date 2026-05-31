"""
Tests for Milestone 4 — Factual Grounding and Hallucination Prevention.

Covers all ten invariant targets:
 1. essential_info removed from response_generator.py
 2. nairobi.json exists and contains all required categories
 3. extract_phones_from_context() and normalise_phone() correctness
 4. Unverified phone numbers are replaced with GROUNDING_FALLBACK_CONTACT
 5. Verified phone numbers are preserved
 6. ## SOURCES section appended when chunks are retrieved
 7. _format_comprehensive_context() chunk label format
 8. POST /admin/update-facts with valid payload returns 200 and reloads
 9. POST /admin/update-facts without auth returns 401
10. Evaluator phone_hallucination_rate field is computed when phone cases exist
"""

import os
import subprocess

import pytest

from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("SETTLEBOT_API_KEY", "test-secure-key-milestone4")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder-milestone4")

ROOT_DIR = Path(__file__).parent.parent.absolute()

# ---------------------------------------------------------------------------
# Invariant 1 — essential_info removed from response_generator.py
# ---------------------------------------------------------------------------


def test_essential_info_removed():
    """
    services/response_generator.py must contain no reference to essential_info.
    """
    result = subprocess.run(
        ["grep", "-n", "essential_info", "services/response_generator.py"],
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
    )
    assert result.returncode == 1, (
        f"essential_info still present in response_generator.py:\n{result.stdout}"
    )


# ---------------------------------------------------------------------------
# Invariant 2 — nairobi.json exists and contains all required categories
# ---------------------------------------------------------------------------


def test_nairobi_json_loads():
    """
    load_fact_store('nairobi') must succeed and return non-empty categories.
    """
    from config.locale import load_fact_store

    fs = load_fact_store("nairobi")
    assert fs.emergency_contacts, "emergency_contacts must be non-empty"
    assert fs.hospitals, "hospitals must be non-empty"
    assert fs.universities, "universities must be non-empty"
    assert fs.government_offices, "government_offices must be non-empty"


def test_nairobi_json_contains_required_counts():
    """
    nairobi.json must contain 6 emergency contacts, hospitals, 16 universities,
    and at least 1 government office entry.
    """
    from config.locale import load_fact_store

    fs = load_fact_store("nairobi")
    assert len(fs.emergency_contacts) == 6, (
        f"Expected 6 emergency contacts, got {len(fs.emergency_contacts)}"
    )
    assert len(fs.universities) == 16, (
        f"Expected 16 universities, got {len(fs.universities)}"
    )
    assert len(fs.government_offices) >= 1


def test_nairobi_json_all_entries_have_verified_date():
    """Every entry in nairobi.json must carry a verified_date field."""
    from config.locale import load_fact_store

    fs = load_fact_store("nairobi")
    for name, contact in fs.emergency_contacts.items():
        assert contact.verified_date, f"emergency_contact {name} missing verified_date"
    for hosp in fs.hospitals:
        assert hosp.verified_date, f"hospital {hosp.name} missing verified_date"
    for uni in fs.universities:
        assert uni.verified_date, f"university {uni.name} missing verified_date"
    for gov in fs.government_offices:
        assert gov.verified_date, f"government_office {gov.name} missing verified_date"


def test_load_fact_store_missing_file():
    """load_fact_store('nonexistent') must raise ValueError."""
    from config.locale import load_fact_store

    with pytest.raises(ValueError, match="not found"):
        load_fact_store("nonexistent_locale_xyz")


# ---------------------------------------------------------------------------
# Invariant 3 — normalise_phone and extract_phones_from_context
# ---------------------------------------------------------------------------


def test_normalise_phone_strips_separators():
    """
    normalise_phone must reduce '+254 20-2845000' and '+254202845000' to the
    same canonical string.
    """
    from utilities.factcheck import normalise_phone

    assert normalise_phone("+254 20-2845000") == normalise_phone("+254202845000")


def test_normalise_phone_strips_parentheses():
    from utilities.factcheck import normalise_phone

    assert normalise_phone("+254 (20) 2845000") == normalise_phone("+254202845000")


def test_extract_phones_returns_verified_set():
    """
    extract_phones_from_context must return a set that includes the normalised
    form of a known hospital number when that number appears in the context text.
    """
    from config.locale import load_fact_store
    from utilities.factcheck import extract_phones_from_context
    from utilities.factcheck import normalise_phone

    fs = load_fact_store("nairobi")
    context = "The Nairobi Hospital can be reached at +254 20 2845000 for appointments."
    result = extract_phones_from_context(context, fs)
    assert normalise_phone("+254202845000") in result


def test_extract_phones_includes_emergency_contacts():
    """
    extract_phones_from_context must include numbers from fact_store.emergency_contacts
    even when the number does not appear in context_text.
    """
    from config.locale import load_fact_store
    from utilities.factcheck import extract_phones_from_context
    from utilities.factcheck import normalise_phone

    fs = load_fact_store("nairobi")
    result = extract_phones_from_context("", fs)
    # Red Cross emergency number is in the fact store
    assert normalise_phone("0700 395 395") in result


# ---------------------------------------------------------------------------
# Invariant 4 — unverified phone numbers are replaced
# ---------------------------------------------------------------------------


def _build_generator_with_mock_llm(llm_return_value: str):
    """
    Build a ResponseGenerator using __new__ with all required attributes set,
    and wire _call_generation_llm to return llm_return_value.
    """
    from config.locale import load_fact_store
    from services.response_generator import ResponseGenerator

    gen = ResponseGenerator.__new__(ResponseGenerator)
    gen.client = MagicMock()
    gen.temperature = 0.2
    gen.max_tokens = 2048
    gen.model = "gpt-4.1-mini"
    gen.fact_store = load_fact_store("nairobi")
    gen.min_context_relevance = 0.3
    gen.min_chunks_for_response = 1
    gen.empathy_responses = {
        "stress": ["This is understandable."],
        "neutral": [],
        "anxiety": ["Your concerns are valid."],
        "urgency": ["Here is what to do."],
        "confusion": ["This can be confusing."],
    }
    gen.safety_protocols = {"general": ["Stay aware of surroundings."]}
    gen.off_topic_response = (
        "## DIRECT ANSWER\nOff topic.\n"
        "## ADDITIONAL INFORMATION\nN/A\n"
        "## NEXT STEPS\n1. Ask a settlement question."
    )
    gen.language_processor = MagicMock()
    gen.language_processor.detect_and_process_query.return_value = {
        "english_query": "test query",
        "detected_language": "english",
        "needs_translation": False,
    }
    gen._call_generation_llm = MagicMock(return_value=llm_return_value)
    return gen


def test_unverified_phone_replaced():
    """
    generate_response() must replace '+254 700 000 000' (not in fact store, not
    in context) with GROUNDING_FALLBACK_CONTACT before the response is returned.
    """
    from config.constants import GROUNDING_FALLBACK_CONTACT
    from services.intent_recognizer import IntentType
    from services.intent_recognizer import TopicType

    fabricated = "+254 700 000 000"
    llm_text = (
        f"## DIRECT ANSWER\nCall {fabricated} for help.\n"
        "## ADDITIONAL INFORMATION\nMore info.\n"
        "## NEXT STEPS\n1. Contact them."
    )

    gen = _build_generator_with_mock_llm(llm_text)
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

    result = gen.generate_response(
        query="test query",
        retrieved_context=[],
        intent_info=intent_info,
        web_info=None,
    )
    assert fabricated not in result["response"], (
        "Fabricated phone number must not appear in the returned response."
    )
    assert GROUNDING_FALLBACK_CONTACT in result["response"], (
        "GROUNDING_FALLBACK_CONTACT must replace the fabricated number."
    )


# ---------------------------------------------------------------------------
# Invariant 5 — verified phone numbers are preserved
# ---------------------------------------------------------------------------


def test_verified_phone_preserved():
    """
    generate_response() must leave '+254 20 2845000' (The Nairobi Hospital,
    present in the fact store) unchanged.
    """
    from services.intent_recognizer import IntentType
    from services.intent_recognizer import TopicType

    verified_phone = "+254 20 2845000"
    llm_text = (
        f"## DIRECT ANSWER\nGo to The Nairobi Hospital: {verified_phone}.\n"
        "## ADDITIONAL INFORMATION\nMore info.\n"
        "## NEXT STEPS\n1. Call ahead."
    )

    gen = _build_generator_with_mock_llm(llm_text)
    intent_info = {
        "intent_type": IntentType.HEALTHCARE,
        "topic": TopicType.HEALTH,
        "confidence": 0.9,
        "settlement_relevance": 0.8,
        "semantic_scores": {},
        "off_topic_indicators": [],
        "classification_method": "semantic_embedding",
        "is_off_topic": False,
    }

    result = gen.generate_response(
        query="which hospital for specialist care",
        retrieved_context=[],
        intent_info=intent_info,
        web_info=None,
    )
    assert verified_phone in result["response"], (
        f"Verified phone {verified_phone} must be preserved in the response."
    )


# ---------------------------------------------------------------------------
# Invariant 6 — ## SOURCES section appended when chunks retrieved
# ---------------------------------------------------------------------------


def test_sources_section_appended():
    """
    generate_response() with non-empty retrieved_context must append ## SOURCES.
    """
    from services.intent_recognizer import IntentType
    from services.intent_recognizer import TopicType

    llm_text = (
        "## DIRECT ANSWER\nHere is housing info.\n"
        "## ADDITIONAL INFORMATION\nMore details.\n"
        "## NEXT STEPS\n1. Search online."
    )
    gen = _build_generator_with_mock_llm(llm_text)
    retrieved = [
        {
            "text": "Westlands is a popular student area.",
            "score": 0.85,
            "doc_id": "doc-abc",
            "chunk_id": "chunk-001",
        }
    ]
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

    result = gen.generate_response(
        query="where to live near university",
        retrieved_context=retrieved,
        intent_info=intent_info,
        web_info=None,
    )
    assert "## SOURCES" in result["response"], (
        "Response must contain ## SOURCES when retrieved_context is non-empty."
    )
    assert "doc-abc" in result["response"], (
        "## SOURCES must list the doc_id of retrieved chunks."
    )


def test_sources_section_absent_for_empty_context():
    """
    generate_response() with empty retrieved_context must not append ## SOURCES.
    """
    from services.intent_recognizer import IntentType
    from services.intent_recognizer import TopicType

    llm_text = (
        "## DIRECT ANSWER\nSome generic guidance.\n"
        "## ADDITIONAL INFORMATION\nMore.\n"
        "## NEXT STEPS\n1. Ask your university."
    )
    gen = _build_generator_with_mock_llm(llm_text)
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

    result = gen.generate_response(
        query="where can I live",
        retrieved_context=[],
        intent_info=intent_info,
        web_info=None,
    )
    assert "## SOURCES" not in result["response"], (
        "Response must not contain ## SOURCES when no chunks were retrieved."
    )


# ---------------------------------------------------------------------------
# Invariant 7 — chunk label format in _format_comprehensive_context()
# ---------------------------------------------------------------------------


def test_chunk_label_format():
    """
    _format_comprehensive_context() must label each chunk as:
    Source N (doc_id: X, chunk: Y, relevance: Z.ZZ):
    """
    from config.locale import load_fact_store
    from services.response_generator import ResponseGenerator

    gen = ResponseGenerator.__new__(ResponseGenerator)
    gen.fact_store = load_fact_store("nairobi")
    gen.min_context_relevance = 0.3

    retrieved = [
        {
            "text": "Safe areas for students include Kilimani.",
            "score": 0.75,
            "doc_id": "doc-xyz",
            "chunk_id": "chunk-007",
        }
    ]
    context_str, doc_ids = gen._format_comprehensive_context(
        retrieved_context=retrieved,
        web_info=None,
        intent_info={"intent_type": None},
        context_evaluation={"essential_needed": []},
    )

    assert "doc_id: doc-xyz" in context_str, (
        "Context must include 'doc_id: doc-xyz' label."
    )
    assert "chunk: chunk-007" in context_str, (
        "Context must include 'chunk: chunk-007' label."
    )
    assert "relevance: 0.75" in context_str, (
        "Context must include 'relevance: 0.75' in the chunk label."
    )
    assert "doc-xyz" in doc_ids, (
        "_format_comprehensive_context must return doc_id in used_doc_ids list."
    )


# ---------------------------------------------------------------------------
# Invariant 8 & 9 — POST /admin/update-facts
# ---------------------------------------------------------------------------


def test_admin_update_facts_requires_auth():
    """
    POST /admin/update-facts without Authorization header must return HTTP 401.
    """
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    response = client.post("/admin/update-facts", json={})
    assert response.status_code == 401, f"Expected 401, got {response.status_code}"


def test_admin_update_facts_invalid_schema():
    """
    POST /admin/update-facts with malformed payload (missing required fields)
    must return HTTP 422.
    """
    from fastapi.testclient import TestClient
    from api.main import app

    api_key = os.environ["SETTLEBOT_API_KEY"]
    client = TestClient(app)
    response = client.post(
        "/admin/update-facts",
        json={"not_a_valid_field": True},
        headers={"Authorization": api_key},
    )
    assert response.status_code == 422, (
        f"Expected 422 for invalid schema, got {response.status_code}"
    )


def test_admin_update_facts_success(tmp_path):
    """
    POST /admin/update-facts with a valid payload and correct API key must
    return 200 and reload the fact store in the running process.
    """
    from fastapi.testclient import TestClient
    from config.locale import load_fact_store
    import api.main as main_module
    from api.main import app

    original_fact_store = main_module.fact_store

    valid_payload = load_fact_store("nairobi").model_dump()

    api_key = os.environ["SETTLEBOT_API_KEY"]
    client = TestClient(app)

    locale_file = ROOT_DIR / "config" / "locale" / "nairobi.json"
    original_content = locale_file.read_text(encoding="utf-8")

    try:
        response = client.post(
            "/admin/update-facts",
            json=valid_payload,
            headers={"Authorization": api_key},
        )
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data.get("success") is True

        updated = main_module.fact_store
        assert updated is not None
        assert len(updated.emergency_contacts) == len(
            original_fact_store.emergency_contacts
        )
    finally:
        locale_file.write_text(original_content, encoding="utf-8")
        main_module.fact_store = original_fact_store
        main_module.response_generator.fact_store = original_fact_store


# ---------------------------------------------------------------------------
# Invariant 10 — evaluator phone_hallucination_rate field
# ---------------------------------------------------------------------------


def test_evaluator_phone_hallucination_rate_field_computed():
    """
    _generate_evaluation_report must produce a phone_hallucination_rate field
    when at least one result carries a phone_correct value.
    """
    from services.evaluator import InternationalStudentRAGEvaluator

    evaluator = InternationalStudentRAGEvaluator.__new__(
        InternationalStudentRAGEvaluator
    )

    results = [
        {
            "overall_score": 0.8,
            "intent_match": True,
            "topic_match": True,
            "contains_expected": True,
            "student_relevance_score": 0.7,
            "practical_info_score": 0.6,
            "empathy_score": 0.5,
            "phone_correct": True,
            "priority": "high",
            "bleu_score": 0.3,
            "token_usage": {
                "total_tokens": 100,
                "prompt_tokens": 50,
                "completion_tokens": 50,
            },
            "urgency": "low",
            "chunks_retrieved": 0,
            "avg_chunk_relevance": 0,
            "actual_intent": "healthcare",
        },
        {
            "overall_score": 0.4,
            "intent_match": False,
            "topic_match": False,
            "contains_expected": False,
            "student_relevance_score": 0.3,
            "practical_info_score": 0.2,
            "empathy_score": 0.1,
            "phone_correct": False,
            "priority": "medium",
            "bleu_score": 0.0,
            "token_usage": {
                "total_tokens": 80,
                "prompt_tokens": 40,
                "completion_tokens": 40,
            },
            "urgency": "low",
            "chunks_retrieved": 0,
            "avg_chunk_relevance": 0,
            "actual_intent": "healthcare",
        },
    ]

    report = evaluator._generate_evaluation_report(
        results=results, failed_queries=0, include_bleu=False
    )

    assert "phone_hallucination_rate" in report, (
        "Evaluation report must contain phone_hallucination_rate key."
    )
    assert report["phone_hallucination_rate"] is not None, (
        "phone_hallucination_rate must be a float, not None, when phone cases exist."
    )
    assert report["phone_hallucination_rate"] == pytest.approx(0.5), (
        "With 1 failed and 1 passed phone case, rate must be 0.5."
    )


def test_evaluator_phone_hallucination_rate_none_when_no_cases():
    """
    phone_hallucination_rate must be None when no results have phone_correct set.
    """
    from services.evaluator import InternationalStudentRAGEvaluator

    evaluator = InternationalStudentRAGEvaluator.__new__(
        InternationalStudentRAGEvaluator
    )

    results = [
        {
            "overall_score": 0.8,
            "intent_match": True,
            "topic_match": True,
            "contains_expected": True,
            "student_relevance_score": 0.7,
            "practical_info_score": 0.6,
            "empathy_score": 0.5,
            "phone_correct": None,
            "priority": "high",
            "bleu_score": 0.0,
            "token_usage": {
                "total_tokens": 100,
                "prompt_tokens": 50,
                "completion_tokens": 50,
            },
            "urgency": "low",
            "chunks_retrieved": 0,
            "avg_chunk_relevance": 0,
            "actual_intent": "housing_inquiry",
        }
    ]

    report = evaluator._generate_evaluation_report(
        results=results, failed_queries=0, include_bleu=False
    )

    assert "phone_hallucination_rate" in report
    assert report["phone_hallucination_rate"] is None, (
        "phone_hallucination_rate must be None when no phone_correct cases exist."
    )


def test_audit_skipped_for_off_topic():
    """
    Off-topic responses must be returned before the phone audit block.
    The response must match the off_topic_response attribute unchanged.
    """
    from config.locale import load_fact_store
    from services.response_generator import ResponseGenerator
    from services.intent_recognizer import IntentType
    from services.intent_recognizer import TopicType

    gen = ResponseGenerator.__new__(ResponseGenerator)
    gen.fact_store = load_fact_store("nairobi")
    gen.language_processor = MagicMock()
    gen.language_processor.detect_and_process_query.return_value = {
        "english_query": "what is bread",
        "detected_language": "english",
        "needs_translation": False,
    }
    gen.off_topic_response = (
        "## DIRECT ANSWER\nOff topic.\n"
        "## ADDITIONAL INFORMATION\nN/A\n"
        "## NEXT STEPS\n1. Ask settlement questions."
    )
    intent_info = {
        "intent_type": IntentType.OFF_TOPIC,
        "topic": TopicType.OFF_TOPIC,
        "confidence": 0.95,
        "settlement_relevance": 0.0,
        "semantic_scores": {},
        "off_topic_indicators": [],
        "classification_method": "threshold",
        "is_off_topic": True,
    }

    result = gen.generate_response(
        query="what is bread",
        retrieved_context=[],
        intent_info=intent_info,
        web_info=None,
    )

    assert result["response"] == gen.off_topic_response, (
        "Off-topic response must not be modified by the phone audit."
    )
    assert "## SOURCES" not in result["response"]
