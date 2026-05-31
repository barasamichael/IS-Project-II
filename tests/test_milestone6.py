"""
Tests for Milestone 6 — Intent Recognition Accuracy.

Covers all ten invariant targets:
 1. INTENT_THRESHOLDS has an entry for every non-OFF_TOPIC IntentType
 2. INTENT_THRESHOLDS, INTENT_AMBIGUITY_GAP, and fallback constants exist
 3. IntentResult has is_ambiguous and secondary_intent fields with defaults
 4. get_intent_info() returns is_ambiguous and secondary_intent keys
 5. Top-2 cosine scores within INTENT_AMBIGUITY_GAP → is_ambiguous=True
 6. Top-2 cosine scores more than INTENT_AMBIGUITY_GAP apart → is_ambiguous=False
 7. max_similarity in [0.40, 0.55] triggers _llm_fallback_classify()
 8. max_similarity above 0.55 does not trigger _llm_fallback_classify()
 9. generate_response() with is_ambiguous=True returns clarification without LLM
10. All expected_intent values in evaluator eval_questions are valid IntentType values
"""

import os

import pytest

from unittest.mock import MagicMock

os.environ.setdefault("SETTLEBOT_API_KEY", "test-secure-key-milestone6")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder-milestone6")


# ---------------------------------------------------------------------------
# Invariants 1 & 2 — constants
# ---------------------------------------------------------------------------


def test_intent_thresholds_complete():
    """INTENT_THRESHOLDS must contain one entry per non-OFF_TOPIC IntentType."""
    from config.constants import INTENT_THRESHOLDS
    from services.intent_recognizer import IntentType

    non_off_topic = {i.value for i in IntentType if i != IntentType.OFF_TOPIC}
    missing = non_off_topic - set(INTENT_THRESHOLDS.keys())
    assert not missing, f"INTENT_THRESHOLDS is missing keys: {missing}"


def test_fallback_constants_exist():
    """INTENT_AMBIGUITY_GAP, INTENT_FALLBACK_LLM_LOWER/UPPER/MODEL must be in constants."""
    from config.constants import (
        INTENT_AMBIGUITY_GAP,
        INTENT_FALLBACK_LLM_LOWER,
        INTENT_FALLBACK_LLM_MODEL,
        INTENT_FALLBACK_LLM_UPPER,
    )

    assert INTENT_AMBIGUITY_GAP == pytest.approx(0.05)
    assert INTENT_FALLBACK_LLM_LOWER == pytest.approx(0.40)
    assert INTENT_FALLBACK_LLM_UPPER == pytest.approx(0.55)
    assert INTENT_FALLBACK_LLM_MODEL == "gpt-4.1-nano"


def test_no_nano_literal_in_intent_recognizer():
    """'gpt-4.1-nano' must not appear as a string literal in intent_recognizer.py."""
    from pathlib import Path

    source = (
        Path(__file__).parent.parent / "services" / "intent_recognizer.py"
    ).read_text()
    # The constant name is fine; the literal in the source would only be via import
    # We check that the string does not appear outside of an import or constant usage
    lines_with_literal = [
        line
        for line in source.splitlines()
        if "gpt-4.1-nano" in line
        and not line.strip().startswith("from ")
        and not line.strip().startswith("import ")
    ]
    assert not lines_with_literal, (
        f"Found 'gpt-4.1-nano' literal in intent_recognizer.py: {lines_with_literal}"
    )


# ---------------------------------------------------------------------------
# Invariant 3 — IntentResult dataclass fields with defaults
# ---------------------------------------------------------------------------


def test_intent_result_has_ambiguity_fields():
    """IntentResult must accept construction without is_ambiguous / secondary_intent."""
    from services.intent_recognizer import IntentResult, IntentType, TopicType

    result = IntentResult(
        intent_type=IntentType.HOUSING_INQUIRY,
        topic=TopicType.HOUSING,
        confidence=0.8,
        semantic_scores={},
        off_topic_indicators=[],
        settlement_relevance=0.6,
    )
    assert result.is_ambiguous is False
    assert result.secondary_intent is None


def test_intent_result_ambiguous_fields_settable():
    """IntentResult must accept explicit is_ambiguous and secondary_intent."""
    from services.intent_recognizer import IntentResult, IntentType, TopicType

    result = IntentResult(
        intent_type=IntentType.HOUSING_INQUIRY,
        topic=TopicType.HOUSING,
        confidence=0.8,
        semantic_scores={},
        off_topic_indicators=[],
        settlement_relevance=0.6,
        is_ambiguous=True,
        secondary_intent=IntentType.COST_INQUIRY,
    )
    assert result.is_ambiguous is True
    assert result.secondary_intent == IntentType.COST_INQUIRY


# ---------------------------------------------------------------------------
# Invariant 4 — get_intent_info() propagates flags
# ---------------------------------------------------------------------------


def _build_recognizer_with_mocked_embeddings(sim_map: dict):
    """
    Build an IntentRecognizer using __new__ with _calculate_similarities
    mocked to return sim_map, bypassing all OpenAI calls.
    """
    from services.intent_recognizer import IntentRecognizer, IntentType, TopicType

    rec = IntentRecognizer.__new__(IntentRecognizer)
    rec.intent_patterns = {
        IntentType.HOUSING_INQUIRY: {
            "examples": ["Where can I find housing?"],
            "topic": TopicType.HOUSING,
        },
        IntentType.COST_INQUIRY: {
            "examples": ["How much does rent cost?"],
            "topic": TopicType.FINANCE,
        },
        IntentType.UNIVERSITY_INFO: {
            "examples": ["Tell me about universities."],
            "topic": TopicType.ACADEMICS,
        },
    }
    rec.settlement_keywords = {
        "high_relevance": [],
        "medium_relevance": [],
        "location_specific": [],
    }
    rec._calculate_similarities = MagicMock(return_value=sim_map)
    rec._calculate_settlement_relevance = MagicMock(return_value=0.3)
    rec._get_query_embedding = MagicMock(
        return_value=__import__("numpy").array([0.1] * 1536)
    )
    rec._llm_fallback_classify = MagicMock(return_value=None)
    return rec


def test_get_intent_info_has_ambiguity_keys():
    """get_intent_info() must return is_ambiguous and secondary_intent keys."""
    from services.intent_recognizer import IntentType

    sim_map = {
        IntentType.HOUSING_INQUIRY: 0.70,
        IntentType.COST_INQUIRY: 0.60,
        IntentType.UNIVERSITY_INFO: 0.50,
    }
    rec = _build_recognizer_with_mocked_embeddings(sim_map)
    result = rec.get_intent_info("where can I rent cheaply")
    assert "is_ambiguous" in result
    assert "secondary_intent" in result


# ---------------------------------------------------------------------------
# Invariants 5 & 6 — ambiguity detection
# ---------------------------------------------------------------------------


def test_ambiguous_when_top2_within_gap():
    """Top-2 scores within INTENT_AMBIGUITY_GAP (0.05) → is_ambiguous=True."""
    from services.intent_recognizer import IntentType

    sim_map = {
        IntentType.HOUSING_INQUIRY: 0.70,
        IntentType.COST_INQUIRY: 0.68,  # gap = 0.02 < 0.05
        IntentType.UNIVERSITY_INFO: 0.50,
    }
    rec = _build_recognizer_with_mocked_embeddings(sim_map)
    result = rec.classify_intent("test query")
    assert result.is_ambiguous is True
    assert result.secondary_intent == IntentType.COST_INQUIRY


def test_not_ambiguous_when_top2_far_apart():
    """Top-2 scores more than INTENT_AMBIGUITY_GAP apart → is_ambiguous=False."""
    from services.intent_recognizer import IntentType

    sim_map = {
        IntentType.HOUSING_INQUIRY: 0.80,
        IntentType.COST_INQUIRY: 0.60,  # gap = 0.20 > 0.05
        IntentType.UNIVERSITY_INFO: 0.50,
    }
    rec = _build_recognizer_with_mocked_embeddings(sim_map)
    result = rec.classify_intent("test query")
    assert result.is_ambiguous is False
    assert result.secondary_intent is None


# ---------------------------------------------------------------------------
# Invariants 7 & 8 — LLM fallback trigger
# ---------------------------------------------------------------------------


def test_fallback_triggered_when_max_in_range():
    """max_similarity = 0.47 (within [0.40, 0.55]) must trigger _llm_fallback_classify."""
    from services.intent_recognizer import IntentType

    sim_map = {
        IntentType.HOUSING_INQUIRY: 0.47,
        IntentType.COST_INQUIRY: 0.44,
        IntentType.UNIVERSITY_INFO: 0.41,
    }
    rec = _build_recognizer_with_mocked_embeddings(sim_map)
    rec.classify_intent("uncertain query")
    rec._llm_fallback_classify.assert_called_once()


def test_fallback_not_triggered_above_range():
    """max_similarity = 0.70 (above 0.55) must NOT trigger _llm_fallback_classify."""
    from services.intent_recognizer import IntentType

    sim_map = {
        IntentType.HOUSING_INQUIRY: 0.70,
        IntentType.COST_INQUIRY: 0.55,
        IntentType.UNIVERSITY_INFO: 0.40,
    }
    rec = _build_recognizer_with_mocked_embeddings(sim_map)
    rec.classify_intent("clear housing query")
    rec._llm_fallback_classify.assert_not_called()


def test_fallback_not_triggered_below_off_topic_threshold():
    """max_similarity = 0.30 (below threshold) fires off-topic path, not fallback."""
    from services.intent_recognizer import IntentType

    sim_map = {
        IntentType.HOUSING_INQUIRY: 0.30,
        IntentType.COST_INQUIRY: 0.28,
        IntentType.UNIVERSITY_INFO: 0.20,
    }
    rec = _build_recognizer_with_mocked_embeddings(sim_map)
    result = rec.classify_intent("what is bread")
    assert result.intent_type == IntentType.OFF_TOPIC
    rec._llm_fallback_classify.assert_not_called()


# ---------------------------------------------------------------------------
# Invariant 9 — generate_response() ambiguity guard
# ---------------------------------------------------------------------------


def _build_response_gen():
    """Build a ResponseGenerator with all attributes mocked."""
    from config.locale import load_fact_store
    from services.response_generator import ResponseGenerator

    gen = ResponseGenerator.__new__(ResponseGenerator)
    gen.fact_store = load_fact_store("nairobi")
    gen.min_context_relevance = 0.3
    gen.min_chunks_for_response = 1
    gen.language_processor = MagicMock()
    gen.language_processor.detect_and_process_query.return_value = {
        "english_query": "test",
        "detected_language": "english",
        "needs_translation": False,
    }
    gen.empathy_responses = {
        "neutral": [],
        "stress": ["OK."],
        "anxiety": ["OK."],
        "urgency": ["OK."],
        "confusion": ["OK."],
    }
    gen.safety_protocols = {"general": ["Stay safe."]}
    gen.off_topic_response = (
        "## DIRECT ANSWER\nOff topic.\n"
        "## ADDITIONAL INFORMATION\nN/A\n"
        "## NEXT STEPS\n1. Ask settlement questions."
    )
    gen._call_generation_llm = MagicMock(
        return_value=(
            "## DIRECT ANSWER\nInfo.\n## ADDITIONAL INFORMATION\nMore.\n## NEXT STEPS\n1. Act."
        )
    )
    return gen


def test_clarification_returned_when_ambiguous():
    """generate_response() with is_ambiguous=True must return clarification without calling LLM."""
    from services.intent_recognizer import IntentType, TopicType

    gen = _build_response_gen()
    intent_info = {
        "intent_type": IntentType.HOUSING_INQUIRY,
        "topic": TopicType.HOUSING,
        "confidence": 0.68,
        "settlement_relevance": 0.5,
        "semantic_scores": {},
        "off_topic_indicators": [],
        "classification_method": "semantic_embedding",
        "is_off_topic": False,
        "is_ambiguous": True,
        "secondary_intent": IntentType.COST_INQUIRY,
    }

    result = gen.generate_response(
        query="how much for housing",
        retrieved_context=[],
        intent_info=intent_info,
        web_info=None,
    )

    assert "Are you asking about" in result["response"]
    assert result.get("response_style") == "clarification"
    gen._call_generation_llm.assert_not_called()


def test_non_ambiguous_proceeds_normally():
    """generate_response() with is_ambiguous=False must call _call_generation_llm."""
    from services.intent_recognizer import IntentType, TopicType

    gen = _build_response_gen()
    intent_info = {
        "intent_type": IntentType.HOUSING_INQUIRY,
        "topic": TopicType.HOUSING,
        "confidence": 0.85,
        "settlement_relevance": 0.7,
        "semantic_scores": {},
        "off_topic_indicators": [],
        "classification_method": "semantic_embedding",
        "is_off_topic": False,
        "is_ambiguous": False,
        "secondary_intent": None,
    }

    gen.generate_response(
        query="where can I find housing near campus",
        retrieved_context=[],
        intent_info=intent_info,
        web_info=None,
    )

    gen._call_generation_llm.assert_called_once()


# ---------------------------------------------------------------------------
# Invariant 10 — evaluator expected_intent values all valid
# ---------------------------------------------------------------------------


def test_evaluator_expected_intents_all_valid():
    """All expected_intent strings in the evaluator eval set must be valid IntentType values."""
    from services.intent_recognizer import IntentType
    from services.evaluator import InternationalStudentRAGEvaluator
    from pathlib import Path
    import tempfile

    evaluator = InternationalStudentRAGEvaluator.__new__(
        InternationalStudentRAGEvaluator
    )
    evaluator.eval_dir = Path(tempfile.mkdtemp())

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    eval_path = evaluator.create_international_student_eval_set(output_path=tmp_path)

    import pandas as pd

    df = pd.read_csv(eval_path)
    valid_values = {i.value for i in IntentType}
    invalid = [
        (row["id"], row["expected_intent"])
        for _, row in df.iterrows()
        if row["expected_intent"] not in valid_values
    ]
    assert not invalid, f"Invalid expected_intent values found: {invalid}"


def test_evaluator_expected_topics_all_valid():
    """All expected_topic strings in the evaluator eval set must be valid TopicType values."""
    from services.intent_recognizer import TopicType
    from services.evaluator import InternationalStudentRAGEvaluator
    from pathlib import Path
    import tempfile

    evaluator = InternationalStudentRAGEvaluator.__new__(
        InternationalStudentRAGEvaluator
    )
    evaluator.eval_dir = Path(tempfile.mkdtemp())

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    eval_path = evaluator.create_international_student_eval_set(output_path=tmp_path)

    import pandas as pd

    df = pd.read_csv(eval_path)
    valid_values = {t.value for t in TopicType}
    invalid = [
        (row["id"], row["expected_topic"])
        for _, row in df.iterrows()
        if row["expected_topic"] not in valid_values
    ]
    assert not invalid, f"Invalid expected_topic values found: {invalid}"


# ---------------------------------------------------------------------------
# Regression guards
# ---------------------------------------------------------------------------


def test_get_intent_info_method_exists():
    """get_intent_info() must exist and not raise AttributeError."""
    from services.intent_recognizer import IntentRecognizer

    assert hasattr(IntentRecognizer, "get_intent_info")
    assert callable(getattr(IntentRecognizer, "get_intent_info"))


def test_recognize_intent_method_does_not_exist():
    """recognize_intent() must not exist on IntentRecognizer."""
    from services.intent_recognizer import IntentRecognizer

    assert not hasattr(IntentRecognizer, "recognize_intent"), (
        "IntentRecognizer must not have a recognize_intent() method."
    )
