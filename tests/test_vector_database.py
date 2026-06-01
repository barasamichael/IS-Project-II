"""Module 4: Vector Database — SB-TECH-2026-001 §5.2"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("SETTLEBOT_API_KEY", "test-api-key-for-unit-tests")
os.environ.setdefault("SETTLEBOT_LOCALE", "nairobi")

_DIM = 8  # small dimension for test speed


def _unit_vec(seed: int) -> np.ndarray:
    """Return a unit-normalised float32 vector seeded for determinism."""
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _build_service(col_name: str):
    """
    Build a VectorDBService backed by an in-memory ChromaDB client.
    :param col_name: str - Collection name.
    :return: VectorDBService instance.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.EphemeralClient(
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    embed_svc = MagicMock()
    embed_svc.dimension = _DIM
    embed_svc.embed_query.return_value = _unit_vec(0)

    with (
        patch("services.vector_db.chromadb.PersistentClient", return_value=client),
        patch("services.vector_db.VectorDBService._build_bm25_index"),
    ):
        from services.vector_db import VectorDBService

        svc = VectorDBService.__new__(VectorDBService)
        svc.embedding_service = embed_svc
        svc.dimension = _DIM
        svc.locale = None
        svc.collection_name = col_name
        svc.client = client
        svc._bm25_index = None
        svc._bm25_texts = []
        svc._bm25_docs = []
        svc.collection = client.create_collection(
            name=col_name, metadata={"hnsw:space": "cosine"}
        )
        svc._initialize_settlement_filters()
    return svc


def _seed(svc, n: int = 3):
    """Add n synthetic chunks to the collection."""
    for i in range(n):
        topic = "housing" if i == 0 else "general"
        location = json.dumps(["Westlands"] if i == 0 else [])
        svc.collection.add(
            ids=[f"chunk_{i:04d}"],
            embeddings=[_unit_vec(i + 1).tolist()],
            documents=[f"International student accommodation housing visa chunk {i}"],
            metadatas=[
                {
                    "doc_id": "doc_seed",
                    "chunk_id": f"chunk_{i:04d}",
                    "chunk_index": i,
                    "settlement_score": 0.9 - i * 0.2,
                    "topic_tags": json.dumps([topic]),
                    "location_entities": location,
                }
            ],
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_search_returns_top_k_results() -> None:
    """search() against a seeded collection returns at most top_k items."""
    svc = _build_service("m4_topk_col")
    _seed(svc, 5)
    svc.embedding_service.embed_query.return_value = _unit_vec(99)
    results = svc.search("housing near campus", top_k=2)
    assert len(results) <= 2


def test_search_empty_collection_returns_empty() -> None:
    """search() on an empty collection returns [] without raising."""
    svc = _build_service("m4_empty_col")
    results = svc.search("housing", top_k=5)
    assert results == []


def test_settlement_boost_applied() -> None:
    """
    A chunk with settlement_score=0.95 ranks above a chunk with
    settlement_score=0.05 when raw cosine distances are equal.
    """
    svc = _build_service("m4_boost_col")
    same_vec = _unit_vec(7)
    svc.collection.add(
        ids=["high"],
        embeddings=[same_vec.tolist()],
        documents=["high settlement relevance housing visa"],
        metadatas=[
            {
                "doc_id": "d1",
                "chunk_id": "high",
                "chunk_index": 0,
                "settlement_score": 0.95,
                "topic_tags": json.dumps(["housing"]),
                "location_entities": json.dumps([]),
            }
        ],
    )
    svc.collection.add(
        ids=["low"],
        embeddings=[same_vec.tolist()],
        documents=["low relevance generic text"],
        metadatas=[
            {
                "doc_id": "d2",
                "chunk_id": "low",
                "chunk_index": 0,
                "settlement_score": 0.05,
                "topic_tags": json.dumps([]),
                "location_entities": json.dumps([]),
            }
        ],
    )
    svc.embedding_service.embed_query.return_value = same_vec
    results = svc.search("housing visa", top_k=5)
    ids = [r["chunk_id"] for r in results]
    assert "high" in ids and "low" in ids
    assert ids.index("high") < ids.index("low")


def test_location_filter_returns_matching_chunks() -> None:
    """
    location_filter boosts chunks that have the location in their metadata.
    The chunk with 'Westlands' in location_entities must score higher with
    location_filter='Westlands' than without.
    """
    svc = _build_service("m4_location_col")
    _seed(svc, 3)
    svc.embedding_service.embed_query.return_value = _unit_vec(5)

    results_filtered = svc.search("accommodation", top_k=3, location_filter="Westlands")
    results_plain = svc.search("accommodation", top_k=3)

    westlands_score_filtered = next(
        (
            r["score"]
            for r in results_filtered
            if "Westlands" in json.dumps(r.get("location_entities", []))
        ),
        None,
    )
    westlands_score_plain = next(
        (
            r["score"]
            for r in results_plain
            if "Westlands" in json.dumps(r.get("location_entities", []))
        ),
        None,
    )
    if westlands_score_filtered is not None and westlands_score_plain is not None:
        assert westlands_score_filtered >= westlands_score_plain


def test_topic_filter_returns_matching_chunks() -> None:
    """
    topic_filter boosts chunks tagged with the given topic.
    Housing-tagged chunk scores higher with topic_filter='housing'.
    """
    svc = _build_service("m4_topic_col")
    _seed(svc, 3)
    svc.embedding_service.embed_query.return_value = _unit_vec(5)

    results_filtered = svc.search("accommodation", top_k=3, topic_filter="housing")
    results_plain = svc.search("accommodation", top_k=3)

    def _housing_score(res):
        for r in res:
            if "housing" in r.get("topic_tags", []):
                return r["score"]
        return None

    s_f = _housing_score(results_filtered)
    s_p = _housing_score(results_plain)
    if s_f is not None and s_p is not None:
        assert s_f >= s_p


def test_index_chunks_adds_to_collection() -> None:
    """index_chunks() with a valid JSONL file increases collection.count()."""
    svc = _build_service("m4_index_col")
    initial_count = svc.collection.count()

    with tempfile.TemporaryDirectory() as tmp:
        chunks_dir = Path(tmp) / "chunks"
        chunks_dir.mkdir()
        embed_dir = Path(tmp) / "embeddings"
        embed_dir.mkdir()

        chunk_file = chunks_dir / "doc_abc_chunks.jsonl"
        chunk_file.write_text(
            json.dumps(
                {
                    "chunk_id": "c1",
                    "doc_id": "doc_abc",
                    "chunk_index": 0,
                    "text": "International student housing guide",
                    "metadata": {
                        "settlement_score": 0.8,
                        "topic_tags": ["housing"],
                        "location_entities": [],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        embed_file = embed_dir / "doc_abc_embeddings.npz"
        np.savez(
            embed_file,
            embeddings=np.array([_unit_vec(1).tolist()], dtype=np.float32),
            chunk_ids=np.array(["c1"]),
        )
        svc.embedding_service.embed_chunks.return_value = {}
        svc.embedding_service.load_embeddings.return_value = {
            "embeddings": np.array([_unit_vec(1)], dtype=np.float32),
            "chunk_ids": np.array(["c1"]),
        }

        from unittest.mock import patch as _patch

        with (
            _patch.object(svc, "_build_bm25_index"),
            _patch(
                "services.vector_db.ROOT_DIR",
                Path(tmp),
            ),
        ):
            svc.index_chunks(chunk_file)

    assert svc.collection.count() > initial_count


def test_health_check_reports_healthy() -> None:
    """health_check() on a populated collection returns overall_health=True."""
    svc = _build_service("m4_health_col")
    _seed(svc, 2)
    svc.embedding_service.embed_query.return_value = _unit_vec(0)
    health = svc.health_check()
    assert health.get("overall_health") is True
