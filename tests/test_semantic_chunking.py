"""
Tests for services/semantic_chunking.py — Milestone 7.
Covers: no per-chunk LLM calls, SEMANTIC enum alias, fallback chunking,
chunk word count bounds, and settlement relevance scoring.
"""
import os
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-placeholder")

from services.semantic_chunking import ChunkingStrategy
from services.semantic_chunking import SemanticChunk
from services.semantic_chunking import SemanticChunker
from services.semantic_chunking import ChunkType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def chunker(monkeypatch):
    """Return a SemanticChunker with the OpenAI client stubbed out."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-placeholder")
    with patch("services.semantic_chunking.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Stub _analyze_text_with_llm so the document-level call succeeds
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            '{"main_topics": ["housing"], "settlement_relevance": 0.8, '
            '"primary_focus": "housing", "contains_practical_info": true, '
            '"mentions_locations": [], "mentions_costs": false, "complexity": "medium"}'
        )
        mock_client.chat.completions.create.return_value = mock_response

        instance = SemanticChunker(
            strategy=ChunkingStrategy.SETTLEMENT_OPTIMIZED
        )
        yield instance


def _make_fake_chunks(doc_id: str, n: int) -> list:
    """Create n SemanticChunk objects with placeholder text."""
    return [
        SemanticChunk(
            chunk_id=f"{doc_id}_{i:04d}",
            doc_id=doc_id,
            text=f"This is settlement text about housing visa university for chunk {i}.",
            start_pos=i * 50,
            end_pos=i * 50 + 50,
            chunk_type=ChunkType.PARAGRAPH,
            word_count=12,
            char_count=50,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Part A — no per-chunk LLM calls
# ---------------------------------------------------------------------------


def test_no_llm_call_per_chunk(chunker):
    """
    _enrich_chunks_with_llm_analysis must call _analyze_chunk_with_llm
    exactly zero times regardless of how many chunks are passed.
    Addresses SB-TECH-2026-001 §3.2.14.
    """
    fake_chunks = _make_fake_chunks("doc_test", 5)

    with patch.object(
        chunker, "_analyze_chunk_with_llm", wraps=chunker._analyze_chunk_with_llm
    ) as mock_per_chunk:
        result = chunker._enrich_chunks_with_llm_analysis(fake_chunks)

    assert mock_per_chunk.call_count == 0, (
        f"_analyze_chunk_with_llm was called {mock_per_chunk.call_count} times; "
        "expected 0 — per-chunk LLM calls must not occur"
    )
    assert len(result) == 5


def test_enrich_populates_scores(chunker):
    """
    After _enrich_chunks_with_llm_analysis, every chunk has settlement_relevance,
    topic_coherence, and semantic_score populated in [0.0, 1.0].
    """
    fake_chunks = _make_fake_chunks("doc_scores", 3)
    result = chunker._enrich_chunks_with_llm_analysis(fake_chunks)

    for chunk in result:
        assert 0.0 <= chunk.settlement_relevance <= 1.0
        assert 0.0 <= chunk.topic_coherence <= 1.0
        assert 0.0 <= chunk.semantic_score <= 1.0


# ---------------------------------------------------------------------------
# Part B — ChunkingStrategy enum alignment
# ---------------------------------------------------------------------------


def test_semantic_enum_value_exists():
    """ChunkingStrategy.SEMANTIC must exist with value 'semantic'."""
    assert ChunkingStrategy.SEMANTIC == "semantic"
    assert ChunkingStrategy.SEMANTIC.value == "semantic"


def test_semantic_strategy_instantiates(monkeypatch):
    """SemanticChunker(strategy=ChunkingStrategy.SEMANTIC) must not raise."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-placeholder")
    with patch("services.semantic_chunking.OpenAI"):
        chunker = SemanticChunker(strategy=ChunkingStrategy.SEMANTIC)
    assert chunker.strategy == ChunkingStrategy.SEMANTIC


def test_semantic_strategy_resolves_from_string(monkeypatch):
    """ChunkingStrategy('semantic') must resolve without ValueError."""
    strategy = ChunkingStrategy("semantic")
    assert strategy == ChunkingStrategy.SEMANTIC


# ---------------------------------------------------------------------------
# Chunking behaviour
# ---------------------------------------------------------------------------


def test_settlement_optimized_produces_chunks(chunker):
    """Non-empty text must produce at least one chunk (>= min_chunk_size//4 words)."""
    # Use at least 25 words so the chunk passes the min_chunk_size // 4 gate (100//4=25)
    text = (
        "International students arriving in this city need to find accommodation "
        "near their university campus as soon as possible. "
        "Housing options include on-campus dormitories and various off-campus rental apartments. "
        "Visa and immigration requirements must be met well before arrival in the country."
    )
    with patch.object(
        chunker, "_analyze_text_with_llm", return_value={"settlement_relevance": 0.8}
    ):
        result = chunker.create_chunks(text, "doc_prod")

    assert len(result) >= 1


def test_empty_text_returns_empty_list(chunker):
    """Empty string must return an empty list without raising."""
    result = chunker.create_chunks("", "doc_empty")
    assert result == []


def test_chunk_word_count_within_bounds(chunker):
    """No chunk word count may exceed max_chunk_size / 4 when text has splittable
    sentence structure."""
    # Build text with clear sentence boundaries so _split_large_chunk can split.
    # Each sentence is ~10 words; repeat 40× = ~400 words across many sentences.
    sentence = (
        "International students need housing accommodation near the university campus."
    )
    text = " ".join([sentence] * 40)
    with patch.object(
        chunker, "_analyze_text_with_llm", return_value={"settlement_relevance": 0.7}
    ):
        result = chunker.create_chunks(text, "doc_bounds")

    max_allowed = chunker.max_chunk_size // 4
    for chunk in result:
        assert chunk.word_count <= max_allowed, (
            f"Chunk {chunk.chunk_id} has {chunk.word_count} words, "
            f"exceeds max {max_allowed}"
        )


def test_fallback_chunking_on_llm_failure(chunker):
    """OpenAI error in _analyze_text_with_llm must trigger fallback chunking
    with no exception raised to the caller."""
    text = "Student housing information. Visa requirements. Transportation costs."
    with patch.object(
        chunker, "_analyze_text_with_llm", side_effect=Exception("LLM down")
    ):
        result = chunker.create_chunks(text, "doc_fallback")

    # Fallback must produce at least one chunk
    assert len(result) >= 1


# ---------------------------------------------------------------------------
# Settlement relevance scoring
# ---------------------------------------------------------------------------


def test_calculate_settlement_relevance_score_range(chunker):
    """_calculate_settlement_relevance_score must return a value in [0.0, 1.0]."""
    texts = [
        "International student accommodation housing university visa immigration",
        "What is bread?",
        "",
    ]
    for text in texts:
        score = chunker._calculate_settlement_relevance_score(text)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range for text: {text!r}"


def test_settlement_relevance_higher_for_relevant_text(chunker):
    """Settlement-relevant text must score higher than generic text."""
    relevant = "International student accommodation housing visa immigration university"
    generic = "The cat sat on the mat and looked at the window"
    assert chunker._calculate_settlement_relevance_score(relevant) > \
           chunker._calculate_settlement_relevance_score(generic)
