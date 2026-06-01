"""
Tests for services/vector_db.py — Milestone 7.
Covers: BM25 index lifecycle, RRF fusion, deduplication, cross-encoder fallback,
search correctness, and settlement boost ordering.
All ChromaDB interactions use an ephemeral in-memory client.
"""
import json
import os
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-placeholder")
# settings.py validates SETTLEBOT_API_KEY at module load; provide a test value
os.environ.setdefault("SETTLEBOT_API_KEY", "test-api-key-for-unit-tests")

from services.vector_db import VectorDBService
from services.vector_db import VectorDBError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 8  # Small dimension to keep test vectors cheap


def _unit_vec(values: list) -> np.ndarray:
    arr = np.array(values, dtype=float)
    norm = np.linalg.norm(arr)
    return (arr / norm).astype(np.float32)


def _make_embedding_service(query_vec: np.ndarray) -> MagicMock:
    svc = MagicMock()
    svc.dimension = _DIM
    svc.embed_query.return_value = query_vec
    return svc


def _build_service(collection_name: str = "test_col") -> VectorDBService:
    """Create a VectorDBService backed by an in-memory ChromaDB client."""
    import chromadb
    from chromadb.config import Settings

    client = chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=False)
    )
    query_vec = _unit_vec([1, 0, 0, 0, 0, 0, 0, 0])
    embed_svc = _make_embedding_service(query_vec)

    with patch("services.vector_db.chromadb.PersistentClient", return_value=client), \
         patch("services.vector_db.VectorDBService._build_bm25_index"), \
         patch.object(VectorDBService, "_reranker", None):

        svc = VectorDBService.__new__(VectorDBService)
        svc.embedding_service = embed_svc
        svc.dimension = _DIM
        svc.locale = None
        svc.collection_name = collection_name
        svc.client = client
        svc._bm25_index = None
        svc._bm25_texts = []
        svc._bm25_docs = []
        svc.collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        svc._initialize_settlement_filters()

    return svc


def _seed_collection(svc: VectorDBService, n: int = 3) -> list:
    """Add n synthetic chunks to the service collection. Returns chunk dicts."""
    chunks = []
    for i in range(n):
        vec = _unit_vec([float(i + 1), 1, 0, 0, 0, 0, 0, 0])
        chunk = {
            "chunk_id": f"chunk_{i:04d}",
            "doc_id": "doc_abc",
            "chunk_index": i,
            "text": f"International student housing accommodation visa university chunk {i}",
            "score": 0.9 - i * 0.1,
            "base_score": 0.9 - i * 0.1,
            "settlement_score": 0.8 - i * 0.1,
        }
        chunks.append(chunk)
        svc.collection.add(
            ids=[chunk["chunk_id"]],
            embeddings=[vec.tolist()],
            documents=[chunk["text"]],
            metadatas=[
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "chunk_index": chunk["chunk_index"],
                    "settlement_score": chunk["settlement_score"],
                    "topic_tags": json.dumps(["housing"]),
                    "location_entities": json.dumps([]),
                }
            ],
        )
    return chunks


# ---------------------------------------------------------------------------
# Search — basic correctness
# ---------------------------------------------------------------------------


def test_search_empty_collection_returns_empty():
    """Search on an empty collection must return [] without raising."""
    svc = _build_service("empty_col")
    result = svc.search("housing near campus", top_k=5)
    assert result == []


def test_search_returns_top_k_results():
    """search() must return at most top_k results from a seeded collection."""
    svc = _build_service("topk_col")
    _seed_collection(svc, n=5)

    # Re-enable embedding
    svc.embedding_service.embed_query.return_value = _unit_vec(
        [1, 1, 0, 0, 0, 0, 0, 0]
    )

    results = svc.search("international student housing", top_k=2)
    assert len(results) <= 2


def test_search_result_has_required_fields():
    """Every search result must include chunk_id, doc_id, text, score, base_score."""
    svc = _build_service("fields_col")
    _seed_collection(svc, n=2)
    svc.embedding_service.embed_query.return_value = _unit_vec(
        [1, 1, 0, 0, 0, 0, 0, 0]
    )

    results = svc.search("housing", top_k=2)
    for r in results:
        assert "chunk_id" in r
        assert "doc_id" in r
        assert "text" in r
        assert "score" in r
        assert "base_score" in r


# ---------------------------------------------------------------------------
# Settlement boost ordering
# ---------------------------------------------------------------------------


def test_settlement_boost_applied():
    """
    A chunk with a higher settlement_score must rank above a chunk with a lower
    settlement_score when cosine distances are otherwise equal.
    """
    svc = _build_service("boost_col")

    same_vec = _unit_vec([1, 1, 0, 0, 0, 0, 0, 0])

    svc.collection.add(
        ids=["high_score"],
        embeddings=[same_vec.tolist()],
        documents=["High settlement relevance housing visa"],
        metadatas=[
            {
                "doc_id": "d1",
                "chunk_id": "high_score",
                "chunk_index": 0,
                "settlement_score": 0.95,
                "topic_tags": json.dumps(["housing"]),
                "location_entities": json.dumps([]),
            }
        ],
    )
    svc.collection.add(
        ids=["low_score"],
        embeddings=[same_vec.tolist()],
        documents=["Low settlement relevance generic text"],
        metadatas=[
            {
                "doc_id": "d2",
                "chunk_id": "low_score",
                "chunk_index": 0,
                "settlement_score": 0.05,
                "topic_tags": json.dumps([]),
                "location_entities": json.dumps([]),
            }
        ],
    )

    svc.embedding_service.embed_query.return_value = same_vec

    results = svc.search("housing visa settlement", top_k=5)
    chunk_ids = [r["chunk_id"] for r in results]

    assert "high_score" in chunk_ids
    assert "low_score" in chunk_ids
    high_idx = chunk_ids.index("high_score")
    low_idx = chunk_ids.index("low_score")
    assert high_idx < low_idx, (
        f"high_score (idx {high_idx}) should outrank low_score (idx {low_idx})"
    )


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------


def test_bm25_index_none_on_empty_collection():
    """_bm25_index must be None when the collection is empty."""
    svc = _build_service("bm25_empty")
    svc._build_bm25_index()
    assert svc._bm25_index is None


def test_bm25_index_built_after_seeding():
    """After seeding the collection and calling _build_bm25_index, the index
    must be non-None when rank_bm25 is available."""
    from services.vector_db import _BM25_AVAILABLE

    svc = _build_service("bm25_seed")
    _seed_collection(svc, n=3)
    svc._build_bm25_index()

    if _BM25_AVAILABLE:
        assert svc._bm25_index is not None
        assert len(svc._bm25_texts) == 3
    else:
        assert svc._bm25_index is None  # graceful degradation


def test_bm25_index_texts_match_collection():
    """After _build_bm25_index, _bm25_texts must contain the same texts as the
    collection documents."""
    from services.vector_db import _BM25_AVAILABLE

    if not _BM25_AVAILABLE:
        pytest.skip("rank_bm25 not installed")

    svc = _build_service("bm25_match")
    _seed_collection(svc, n=4)
    svc._build_bm25_index()

    assert len(svc._bm25_texts) == 4
    all_texts = svc.collection.get(include=["documents"])["documents"]
    assert sorted(svc._bm25_texts) == sorted(all_texts)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def test_reciprocal_rank_fusion_merges_results():
    """_reciprocal_rank_fusion must return a merged, deduplicated list."""
    svc = _build_service("rrf_col")
    dense = [
        {"chunk_id": "a", "text": "a", "score": 0.9, "base_score": 0.9, "doc_id": "d"},
        {"chunk_id": "b", "text": "b", "score": 0.8, "base_score": 0.8, "doc_id": "d"},
    ]
    sparse = [
        {"chunk_id": "b", "text": "b", "score": 5.0, "base_score": 5.0, "doc_id": "d"},
        {"chunk_id": "c", "text": "c", "score": 4.0, "base_score": 4.0, "doc_id": "d"},
    ]
    merged = svc._reciprocal_rank_fusion(dense, sparse, k=60)

    ids = [r["chunk_id"] for r in merged]
    assert "a" in ids
    assert "b" in ids
    assert "c" in ids
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids in RRF output"


def test_reciprocal_rank_fusion_sorted_descending():
    """RRF result must be sorted by rrf_score descending."""
    svc = _build_service("rrf_sort")
    dense = [
        {"chunk_id": "x", "text": "x", "score": 0.9, "base_score": 0.9, "doc_id": "d"},
    ]
    sparse = [
        {"chunk_id": "x", "text": "x", "score": 5.0, "base_score": 5.0, "doc_id": "d"},
        {"chunk_id": "y", "text": "y", "score": 4.0, "base_score": 4.0, "doc_id": "d"},
    ]
    merged = svc._reciprocal_rank_fusion(dense, sparse, k=60)

    scores = [r["rrf_score"] for r in merged]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_deduplication_skips_near_duplicate():
    """
    When enable_deduplication=True and vector_db_service is injected,
    _create_settlement_chunks must skip chunks whose top neighbour has
    base_score >= similarity_threshold.
    """
    from services.document_processor import DocumentProcessor

    mock_vdb = MagicMock()
    mock_vdb.search.return_value = [
        {"chunk_id": "existing_chunk", "base_score": 0.95}
    ]

    mock_embed_svc = MagicMock()
    mock_embed_svc.embed_query.return_value = _unit_vec([1, 0, 0, 0, 0, 0, 0, 0])

    mock_chunker = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.text = "International student housing visa university accommodation"
    mock_chunk.semantic_score = 0.7
    mock_chunk.topic_coherence = 0.8
    mock_chunk.chunk_type = "paragraph"
    mock_chunker.create_chunks.return_value = [mock_chunk]

    with patch("services.document_processor.SemanticChunker", return_value=mock_chunker), \
         patch("services.document_processor.EmbeddingService", return_value=mock_embed_svc), \
         patch("services.document_processor.nlp", MagicMock(return_value=MagicMock(ents=[]))):

        dp = DocumentProcessor(
            raw_dir="/tmp/dp_raw",
            processed_dir="/tmp/dp_proc",
            chunk_dir="/tmp/dp_chunks",
            dedup_dir="/tmp/dp_dedup",
            embedding_service=mock_embed_svc,
            vector_db_service=mock_vdb,
            enable_deduplication=True,
            similarity_threshold=0.92,
        )

        chunks, skipped = dp._create_settlement_chunks(
            "International student housing visa university accommodation text",
            "doc_dedup",
            "/tmp/fake.txt",
        )

    assert skipped == 1, f"Expected 1 skipped duplicate, got {skipped}"
    assert len(chunks) == 0


def test_deduplication_keeps_unique_chunks():
    """When neighbours have base_score < similarity_threshold, chunk must be kept."""
    from services.document_processor import DocumentProcessor

    mock_vdb = MagicMock()
    mock_vdb.search.return_value = [
        {"chunk_id": "distant_chunk", "base_score": 0.50}
    ]

    mock_embed_svc = MagicMock()
    mock_embed_svc.embed_query.return_value = _unit_vec([1, 0, 0, 0, 0, 0, 0, 0])

    mock_chunker = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.text = "International student housing visa"
    mock_chunk.semantic_score = 0.7
    mock_chunk.topic_coherence = 0.8
    mock_chunk.chunk_type = "paragraph"
    mock_chunker.create_chunks.return_value = [mock_chunk]

    with patch("services.document_processor.SemanticChunker", return_value=mock_chunker), \
         patch("services.document_processor.EmbeddingService", return_value=mock_embed_svc), \
         patch("services.document_processor.nlp", MagicMock(return_value=MagicMock(ents=[]))):

        dp = DocumentProcessor(
            raw_dir="/tmp/dp_raw2",
            processed_dir="/tmp/dp_proc2",
            chunk_dir="/tmp/dp_chunks2",
            dedup_dir="/tmp/dp_dedup2",
            embedding_service=mock_embed_svc,
            vector_db_service=mock_vdb,
            enable_deduplication=True,
            similarity_threshold=0.92,
        )

        chunks, skipped = dp._create_settlement_chunks(
            "International student housing visa text",
            "doc_unique",
            "/tmp/fake2.txt",
        )

    assert skipped == 0
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Locale fix — _generate_settlement_queries
# ---------------------------------------------------------------------------


def test_generate_settlement_queries_no_nairobi_literal():
    """_generate_settlement_queries must contain no hardcoded 'Nairobi' or 'Kenya'."""
    import inspect
    import services.vector_db as vdb_module

    source = inspect.getsource(vdb_module.VectorDBService._generate_settlement_queries)
    assert "Nairobi" not in source, "Hardcoded 'Nairobi' found in _generate_settlement_queries"
    assert "Kenya" not in source, "Hardcoded 'Kenya' found in _generate_settlement_queries"


def test_generate_settlement_queries_uses_locale_city():
    """When locale is passed, alternatives should include locale.city and locale.country."""
    svc = _build_service("locale_col")
    locale = MagicMock()
    locale.city = "Kampala"
    locale.country = "Uganda"

    alts = svc._generate_settlement_queries("where can I live", locale=locale)
    combined = " ".join(alts)
    assert "Kampala" in combined or "Uganda" in combined


def test_generate_settlement_queries_no_locale_does_not_crash():
    """_generate_settlement_queries(query, locale=None) must not raise."""
    svc = _build_service("no_locale_col")
    result = svc._generate_settlement_queries("housing near campus", locale=None)
    assert isinstance(result, list)
    assert len(result) <= 4


# ---------------------------------------------------------------------------
# Cross-encoder fallback
# ---------------------------------------------------------------------------


def test_search_works_when_reranker_is_none():
    """search() must return valid results when _reranker is None."""
    svc = _build_service("no_reranker_col")
    _seed_collection(svc, n=3)
    svc.embedding_service.embed_query.return_value = _unit_vec(
        [1, 1, 0, 0, 0, 0, 0, 0]
    )

    original_reranker = VectorDBService._reranker
    try:
        VectorDBService._reranker = None
        VectorDBService._reranker_warning_logged = False
        results = svc.search("housing", top_k=3)
        assert isinstance(results, list)
    finally:
        VectorDBService._reranker = original_reranker
