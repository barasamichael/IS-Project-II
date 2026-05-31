import json
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional

import numpy as np
import chromadb
from chromadb.config import Settings

from config.constants import BM25_RRF_K
from config.constants import BM25_INDEX_TOP_K
from config.constants import CROSS_ENCODER_MODEL
from config.settings import settings
from config.settings import ROOT_DIR
from services.embeddings import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_db")

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi

    _BM25_AVAILABLE = True
except ImportError:
    _BM25Okapi = None  # type: ignore[assignment,misc]
    _BM25_AVAILABLE = False
    logger.warning("BM25 unavailable: rank_bm25 not installed; BM25 retrieval disabled")


class VectorDBError(Exception):
    """Custom exception for vector database errors."""

    pass


class VectorDBService:
    """
    Production-grade vector database service optimized for settlement content.
    Retrieval uses dense ChromaDB search fused with BM25 via Reciprocal Rank
    Fusion, then reranked by a local cross-encoder.
    """

    _reranker: Any = None
    _reranker_warning_logged: bool = False

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        """
        Initialise VectorDBService with ChromaDB, embedding service, and BM25 index.
        :param embedding_service: Optional[EmbeddingService] - Pre-built embedding
               service; a new instance is created if not supplied.
        """
        try:
            self.embedding_service = embedding_service or EmbeddingService()
            self.dimension = self.embedding_service.dimension
            self.collection_name = settings.vector_db.collection_name

            self.db_path = ROOT_DIR / "database" / "chroma_db"
            self.db_path.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            self.collection = self._get_or_create_collection()

            self._initialize_settlement_filters()

            # BM25 sparse index — populated from collection contents
            self._bm25_index: Any = None
            self._bm25_texts: List[str] = []
            self._bm25_docs: List[Dict[str, Any]] = []
            self._build_bm25_index()

            logger.info(f"ChromaDB initialized at {self.db_path}")
            logger.info(
                f"Collection '{self.collection_name}' loaded with "
                f"{self.collection.count()} vectors"
            )

        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            raise VectorDBError(f"Database initialization failed: {str(e)}")

    def _get_or_create_collection(self):
        """
        Get existing ChromaDB collection or create a new one.
        :return: chromadb.Collection - The named collection.
        """
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
            return collection
        except Exception:
            logger.info(f"Creating new collection: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16,
                    "settlement_optimized": True,
                    "description": f"SettleBot {self.collection_name} settlement content",
                },
            )

    def _initialize_settlement_filters(self) -> None:
        """
        Initialise topic weights and location boost mappings for settlement scoring.
        :return: None
        """
        self.topic_weights = {
            "housing": 1.2,
            "transportation": 1.1,
            "education": 1.1,
            "legal": 1.3,
            "finance": 1.1,
            "safety": 1.3,
            "healthcare": 1.1,
            "culture": 1.0,
        }

        self.location_boost = {
            "nairobi": 1.3,
            "westlands": 1.2,
            "kilimani": 1.2,
            "karen": 1.2,
            "lavington": 1.2,
        }

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def _build_bm25_index(self) -> None:
        """
        Build the BM25Okapi sparse index from all documents in the collection.
        Called at startup and after every index_chunks() invocation.
        Sets _bm25_index to None silently when the collection is empty or
        rank_bm25 is unavailable.
        :return: None
        """
        if not _BM25_AVAILABLE:
            logger.warning("BM25 index skipped: rank_bm25 not installed")
            return

        try:
            if self.collection.count() == 0:
                self._bm25_index = None
                self._bm25_texts = []
                self._bm25_docs = []
                return

            all_docs = self.collection.get(include=["documents", "metadatas"])
            texts: List[str] = all_docs.get("documents") or []
            metadatas: List[Dict[str, Any]] = all_docs.get("metadatas") or []

            if not texts:
                self._bm25_index = None
                return

            tokenized = [t.lower().split() for t in texts]
            self._bm25_index = _BM25Okapi(tokenized)
            self._bm25_texts = texts
            self._bm25_docs = metadatas

            logger.info(f"BM25 index built from {len(texts)} documents")

        except Exception as exc:
            logger.warning(f"BM25 index build failed: {exc}")
            self._bm25_index = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def initialize_collection(self, recreate: bool = False) -> None:
        """
        Initialise or recreate the vector collection.
        :param recreate: bool - When True, delete and recreate the collection.
        :return: None
        """
        try:
            if recreate:
                try:
                    self.client.delete_collection(name=self.collection_name)
                    logger.info(f"Deleted existing collection: {self.collection_name}")
                except Exception as e:
                    logger.warning(f"No existing collection to delete: {e}")

                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "hnsw:construction_ef": 200,
                        "hnsw:M": 16,
                        "settlement_optimized": True,
                    },
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                logger.info(
                    f"Collection already exists with {self.collection.count()} vectors"
                )

        except Exception as e:
            logger.error(f"Collection init failed: {e}")
            raise VectorDBError(f"Collection initialization failed: {str(e)}")

    def index_chunks(self, chunks_file: Optional[Union[str, Path]] = None) -> None:
        """
        Index chunks with settlement-specific optimisation, then rebuild the
        BM25 index from the updated collection.
        :param chunks_file: Optional[Union[str, Path]] - Path to a specific chunk
               file; when omitted all files in data/chunks/ are processed.
        :return: None
        """
        try:
            dedup_file = (
                ROOT_DIR / "data" / "deduplicated" / "deduplicated_chunks.jsonl"
            )

            if dedup_file.exists():
                logger.info("Found deduplicated chunks, indexing those")
                self._index_deduplicated_chunks(dedup_file)
                self._build_bm25_index()
                return

            if chunks_file:
                chunks_file = Path(chunks_file)
                chunks_dir = chunks_file.parent
            else:
                chunks_dir = ROOT_DIR / "data" / "chunks"

            if not chunks_dir.exists():
                raise VectorDBError(f"Chunks directory not found: {chunks_dir}")

            files_to_process = (
                [chunks_file]
                if chunks_file
                else list(chunks_dir.glob("*_chunks.jsonl"))
            )

            if not files_to_process:
                raise VectorDBError("No chunk files found to index")

            successful = 0
            failed = 0

            for chunk_file in files_to_process:
                try:
                    self._index_chunks_file(chunk_file)
                    successful += 1
                except Exception as e:
                    logger.error(f"Failed to index {chunk_file.name}: {e}")
                    failed += 1

            logger.info(
                f"Indexing complete: {successful} succeeded, {failed} failed; "
                f"total vectors: {self.collection.count()}"
            )

            self._build_bm25_index()

        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            raise VectorDBError(f"Indexing failed: {str(e)}")

    def _index_chunks_file(self, chunks_file: Path) -> None:
        """
        Index chunks from a single JSONL file with settlement metadata.
        :param chunks_file: Path - Path to the chunks JSONL file.
        :return: None
        """
        try:
            logger.info(f"Indexing chunks from: {chunks_file.name}")

            doc_id = chunks_file.stem.replace("_chunks", "")

            embeddings_dir = ROOT_DIR / "data" / "embeddings"
            embeddings_file = embeddings_dir / f"{doc_id}_embeddings.npz"

            if not embeddings_file.exists():
                logger.info(f"Generating embeddings for {doc_id}")
                self.embedding_service.embed_chunks(chunks_file)

            embeddings_data = self.embedding_service.load_embeddings(embeddings_file)
            if embeddings_data is None:
                raise VectorDBError(f"Failed to load embeddings for {doc_id}")

            embeddings = embeddings_data["embeddings"]
            chunk_ids = embeddings_data["chunk_ids"]

            chunks = []
            with open(chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line: {e}")

            if len(chunks) != len(embeddings):
                logger.warning(
                    f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
                )
                min_len = min(len(chunks), len(embeddings))
                chunks = chunks[:min_len]
                embeddings = embeddings[:min_len]
                chunk_ids = chunk_ids[:min_len]

            ids = [str(chunk["chunk_id"]) for chunk in chunks]
            documents = [chunk["text"] for chunk in chunks]
            metadatas = []

            for chunk in chunks:
                metadata: Dict[str, Any] = {
                    "doc_id": chunk["doc_id"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_id": chunk["chunk_id"],
                }
                chunk_metadata = chunk.get("metadata", {})
                if "settlement_score" in chunk_metadata:
                    metadata["settlement_score"] = chunk_metadata["settlement_score"]
                if "topic_tags" in chunk_metadata:
                    metadata["topic_tags"] = json.dumps(chunk_metadata["topic_tags"])
                if "location_entities" in chunk_metadata:
                    metadata["location_entities"] = json.dumps(
                        chunk_metadata["location_entities"]
                    )
                if "cost_entities" in chunk_metadata:
                    metadata["cost_entities"] = json.dumps(
                        chunk_metadata["cost_entities"]
                    )
                metadatas.append(metadata)

            batch_size = 100
            for i in range(0, len(ids), batch_size):
                self.collection.add(
                    ids=ids[i : i + batch_size],
                    embeddings=embeddings[i : i + batch_size].tolist(),
                    documents=documents[i : i + batch_size],
                    metadatas=metadatas[i : i + batch_size],
                )

            logger.info(f"Successfully indexed {len(ids)} chunks from {doc_id}")

        except Exception as e:
            logger.error(f"Error indexing chunks file {chunks_file.name}: {e}")
            raise VectorDBError(f"Failed to index {chunks_file.name}: {str(e)}")

    def _index_deduplicated_chunks(self, dedup_file: Path) -> None:
        """
        Index deduplicated chunks with enhanced metadata.
        :param dedup_file: Path - Path to the deduplicated chunks JSONL file.
        :return: None
        """
        try:
            embeddings_dir = ROOT_DIR / "data" / "embeddings"
            embeddings_file = embeddings_dir / "deduplicated_embeddings.npz"

            if not embeddings_file.exists():
                logger.info("Generating embeddings for deduplicated chunks")
                self.embedding_service.embed_deduplicated_chunks()

            embeddings_data = self.embedding_service.load_embeddings(embeddings_file)
            if embeddings_data is None:
                raise VectorDBError("Failed to load deduplicated embeddings")

            embeddings = embeddings_data["embeddings"]

            chunks = []
            with open(dedup_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line: {e}")

            if len(chunks) != len(embeddings):
                logger.warning(
                    f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
                )
                min_len = min(len(chunks), len(embeddings))
                chunks = chunks[:min_len]
                embeddings = embeddings[:min_len]

            ids = [str(chunk["chunk_id"]) for chunk in chunks]
            documents = [chunk["text"] for chunk in chunks]
            metadatas = []

            for chunk in chunks:
                metadata: Dict[str, Any] = {
                    "doc_id": chunk.get("doc_id", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "chunk_id": chunk["chunk_id"],
                }
                chunk_metadata = chunk.get("metadata", {})
                if "settlement_score" in chunk_metadata:
                    metadata["settlement_score"] = chunk_metadata["settlement_score"]
                if "is_merged" in chunk_metadata:
                    metadata["is_merged"] = True
                    metadata["merged_count"] = chunk_metadata.get("merge_count", 0)
                metadatas.append(metadata)

            batch_size = 100
            for i in range(0, len(ids), batch_size):
                self.collection.add(
                    ids=ids[i : i + batch_size],
                    embeddings=embeddings[i : i + batch_size].tolist(),
                    documents=documents[i : i + batch_size],
                    metadatas=metadatas[i : i + batch_size],
                )

            logger.info(f"Successfully indexed {len(ids)} deduplicated chunks")

        except Exception as e:
            logger.error(f"Error indexing deduplicated chunks: {e}")
            raise VectorDBError(f"Failed to index deduplicated chunks: {str(e)}")

    # ------------------------------------------------------------------
    # Retrieval — dense + BM25 + RRF + cross-encoder
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        dense: List[Dict[str, Any]],
        sparse: List[Dict[str, Any]],
        k: int,
    ) -> List[Dict[str, Any]]:
        """
        Merge dense and sparse retrieval lists with Reciprocal Rank Fusion.
        Documents present in only one list contribute 1/(k + rank) from that list.
        :param dense: List[Dict[str, Any]] - Dense retrieval results in rank order.
        :param sparse: List[Dict[str, Any]] - BM25 retrieval results in rank order.
        :param k: int - RRF smoothing constant (typically 60).
        :return: List[Dict[str, Any]] - Merged, deduplicated list sorted by
                 descending RRF score; each entry gains an "rrf_score" key.
        """
        rrf_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Dict[str, Any]] = {}

        for rank, result in enumerate(dense):
            cid = result.get("chunk_id", "")
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            doc_lookup[cid] = result

        for rank, result in enumerate(sparse):
            cid = result.get("chunk_id", "")
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in doc_lookup:
                doc_lookup[cid] = result

        merged = sorted(
            [
                {"rrf_score": score, **doc_lookup[cid]}
                for cid, score in rrf_scores.items()
            ],
            key=lambda x: x["rrf_score"],
            reverse=True,
        )
        return merged

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_doc_id: Optional[str] = None,
        topic_filter: Optional[str] = None,
        location_filter: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search with settlement-specific optimisation: dense ChromaDB retrieval
        fused with BM25 via Reciprocal Rank Fusion, then reranked by a
        cross-encoder.
        :param query: str - The query string.
        :param top_k: int - Maximum number of results to return.
        :param filter_doc_id: Optional[str] - Restrict results to one document.
        :param topic_filter: Optional[str] - Boost results matching this topic.
        :param location_filter: Optional[str] - Boost results matching this location.
        :param embedding: Optional[np.ndarray] - Pre-computed query embedding;
               skips embed_query call when provided.
        :return: List[Dict[str, Any]] - Ranked list of matching chunks.
        """
        try:
            if self.collection.count() == 0:
                logger.warning("Collection is empty")
                return []

            query_embedding = (
                embedding
                if embedding is not None
                else self.embedding_service.embed_query(query)
            )

            if query_embedding is None:
                raise VectorDBError("Failed to generate query embedding")

            where_filter: Dict[str, Any] = {}
            if filter_doc_id:
                where_filter["doc_id"] = filter_doc_id

            # --- Dense retrieval ---
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k * 2, self.collection.count()),
                where=where_filter if where_filter else None,
                include=["metadatas", "documents", "distances"],
            )

            dense_results: List[Dict[str, Any]] = []
            if results and results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    base_score = 1 - results["distances"][0][i]
                    boosted_score = self._apply_settlement_boost(
                        text=results["documents"][0][i],
                        metadata=results["metadatas"][0][i],
                        base_score=base_score,
                        query=query,
                        topic_filter=topic_filter,
                        location_filter=location_filter,
                    )
                    result: Dict[str, Any] = {
                        "chunk_id": results["metadatas"][0][i].get("chunk_id", ""),
                        "doc_id": results["metadatas"][0][i].get("doc_id", ""),
                        "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                        "text": results["documents"][0][i],
                        "score": boosted_score,
                        "base_score": base_score,
                    }
                    metadata = results["metadatas"][0][i]
                    if "settlement_score" in metadata:
                        result["settlement_score"] = metadata["settlement_score"]
                    if "topic_tags" in metadata:
                        try:
                            result["topic_tags"] = json.loads(metadata["topic_tags"])
                        except Exception:
                            result["topic_tags"] = []
                    if "location_entities" in metadata:
                        try:
                            result["location_entities"] = json.loads(
                                metadata["location_entities"]
                            )
                        except Exception:
                            result["location_entities"] = []
                    dense_results.append(result)

            # --- BM25 sparse retrieval ---
            bm25_results: List[Dict[str, Any]] = []
            if self._bm25_index is not None:
                query_tokens = query.lower().split()
                bm25_scores = self._bm25_index.get_scores(query_tokens)
                top_indices = np.argsort(bm25_scores)[::-1][:BM25_INDEX_TOP_K]
                for idx in top_indices:
                    if int(idx) < len(self._bm25_docs) and float(bm25_scores[idx]) > 0:
                        meta = self._bm25_docs[int(idx)]
                        bm25_results.append(
                            {
                                "chunk_id": meta.get("chunk_id", ""),
                                "doc_id": meta.get("doc_id", ""),
                                "chunk_index": meta.get("chunk_index", 0),
                                "text": self._bm25_texts[int(idx)],
                                "score": float(bm25_scores[idx]),
                                "base_score": float(bm25_scores[idx]),
                            }
                        )

            # --- Reciprocal Rank Fusion ---
            if bm25_results:
                fused = self._reciprocal_rank_fusion(
                    dense_results, bm25_results, k=BM25_RRF_K
                )
            else:
                fused = sorted(dense_results, key=lambda x: x["score"], reverse=True)

            # --- Cross-encoder reranking ---
            reranker = VectorDBService._reranker
            if reranker is not None and fused:
                candidates = fused[: top_k * 2]
                pairs = [(query, r["text"]) for r in candidates]
                try:
                    rerank_scores = reranker.predict(pairs)
                    reranked = sorted(
                        zip(candidates, rerank_scores),
                        key=lambda x: float(x[1]),
                        reverse=True,
                    )
                    return [r[0] for r in reranked[:top_k]]
                except Exception as exc:
                    if not VectorDBService._reranker_warning_logged:
                        logger.warning(f"Reranker predict failed: {exc}")
                        VectorDBService._reranker_warning_logged = True
                    return fused[:top_k]
            else:
                if (
                    fused
                    and reranker is None
                    and not VectorDBService._reranker_warning_logged
                ):
                    logger.warning(
                        "Reranker unavailable: cross-encoder not loaded; "
                        "returning RRF-fused results"
                    )
                    VectorDBService._reranker_warning_logged = True
                return fused[:top_k]

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise VectorDBError(f"Search failed: {str(e)}")

    def _apply_settlement_boost(
        self,
        text: str,
        metadata: Dict[str, Any],
        base_score: float,
        query: str,
        topic_filter: Optional[str] = None,
        location_filter: Optional[str] = None,
    ) -> float:
        """
        Apply settlement-specific score boosting to a dense retrieval result.
        :param text: str - Chunk text.
        :param metadata: Dict - ChromaDB chunk metadata.
        :param base_score: float - Raw cosine similarity score.
        :param query: str - Original query string.
        :param topic_filter: Optional[str] - Topic to boost.
        :param location_filter: Optional[str] - Location to boost.
        :return: float - Boosted score capped at 1.0.
        """
        boosted_score = base_score
        text_lower = text.lower()
        query_lower = query.lower()

        settlement_score = metadata.get("settlement_score", 0.5)
        boosted_score *= 1 + settlement_score * 0.2

        if "topic_tags" in metadata:
            try:
                topic_tags = json.loads(metadata["topic_tags"])
                for topic in topic_tags:
                    topic_boost = self.topic_weights.get(topic, 1.0)
                    boosted_score *= topic_boost
                    if topic in query_lower:
                        boosted_score *= 1.15
            except Exception:
                pass

        if "location_entities" in metadata:
            try:
                locations = json.loads(metadata["location_entities"])
                for location in locations:
                    location_boost = self.location_boost.get(location.lower(), 1.0)
                    boosted_score *= location_boost
                    if location.lower() in query_lower:
                        boosted_score *= 1.2
            except Exception:
                pass

        high_value_keywords = [
            "international student",
            "accommodation",
            "visa",
            "safety",
            "transport",
            "cost",
            "university",
            "nairobi",
        ]
        keyword_matches = sum(1 for kw in high_value_keywords if kw in text_lower)
        if keyword_matches > 0:
            boosted_score *= 1 + keyword_matches * 0.05

        if topic_filter and topic_filter in metadata.get("topic_tags", ""):
            boosted_score *= 1.3

        if location_filter:
            location_entities = metadata.get("location_entities", "")
            if location_filter.lower() in location_entities.lower():
                boosted_score *= 1.3

        if metadata.get("is_merged", False):
            boosted_score *= 1.1

        return min(boosted_score, 1.0)

    # ------------------------------------------------------------------
    # Multi-query expansion
    # ------------------------------------------------------------------

    def multi_query_search(
        self,
        query: str,
        top_k: int = 20,
        filter_doc_id: Optional[str] = None,
        topic_filter: Optional[str] = None,
        locale: Any = None,  # LocaleConfig-compatible; full type enforced in Milestone 8
    ) -> List[Dict[str, Any]]:
        """
        Multi-query search with settlement-specific query expansion.
        :param query: str - The original user query.
        :param top_k: int - Maximum results to return.
        :param filter_doc_id: Optional[str] - Restrict to one document.
        :param topic_filter: Optional[str] - Topic boost filter.
        :param locale: Any - Optional locale object with .city and .country attributes.
        :return: List[Dict[str, Any]] - Ranked combined results.
        """
        try:
            original_results = self.search(
                query=query,
                top_k=int(top_k * 0.6),
                filter_doc_id=filter_doc_id,
                topic_filter=topic_filter,
            )

            alt_queries = self._generate_settlement_queries(query, locale=locale)

            all_results = original_results.copy()
            seen_chunks = {r["chunk_id"] for r in original_results}

            for alt_query in alt_queries:
                if len(all_results) >= top_k:
                    break
                try:
                    alt_results = self.search(
                        query=alt_query,
                        top_k=5,
                        filter_doc_id=filter_doc_id,
                        topic_filter=topic_filter,
                    )
                    for result in alt_results:
                        if result["chunk_id"] not in seen_chunks:
                            result["score"] *= 0.95
                            result["query_type"] = "alternative"
                            all_results.append(result)
                            seen_chunks.add(result["chunk_id"])
                except Exception as e:
                    logger.warning(f"Alternative query failed: {e}")
                    continue

            all_results.sort(key=lambda x: x["score"], reverse=True)
            return all_results[:top_k]

        except Exception as e:
            logger.error(f"Multi-query search error: {e}")
            try:
                return self.search(query, top_k, filter_doc_id, topic_filter)
            except Exception:
                return []

    def _generate_settlement_queries(
        self,
        original_query: str,
        locale: Any = None,  # LocaleConfig-compatible; full type enforced in Milestone 8
    ) -> List[str]:
        """
        Generate settlement-specific alternative queries for multi-query expansion.
        Uses locale.city and locale.country when a locale object is supplied;
        skips location-specific expansion when locale is None.
        :param original_query: str - The original user query.
        :param locale: Any - Optional locale object with .city and .country.
        :return: List[str] - Up to four alternative query strings.
        """
        query_lower = original_query.lower()
        alternatives: List[str] = []

        city = locale.city if locale is not None else ""
        country = locale.country if locale is not None else ""

        # Add location context only when the query lacks it and locale is available
        if city and country:
            if city.lower() not in query_lower and country.lower() not in query_lower:
                alternatives.append(f"{original_query} in {city} {country}")

        # Add student context if missing
        if "student" not in query_lower and "international" not in query_lower:
            alternatives.append(f"international student {original_query}")

        # Topic-specific expansions
        if any(w in query_lower for w in ["house", "room", "accommodation"]):
            alternatives.extend(
                [
                    f"{original_query} for international students",
                    f"student {original_query} near university",
                ]
            )

        elif any(w in query_lower for w in ["transport", "travel", "commute"]):
            location_suffix = f" {city}" if city else ""
            alternatives.extend(
                [
                    f"{original_query} public transport{location_suffix}",
                    f"student {original_query} university",
                ]
            )

        elif any(w in query_lower for w in ["cost", "price", "budget"]):
            country_suffix = f" {country}" if country else ""
            alternatives.extend(
                [
                    f"cost of {original_query}{country_suffix}",
                    f"{original_query} international student budget",
                ]
            )

        elif any(w in query_lower for w in ["safe", "security", "danger"]):
            location_suffix = f" {city}" if city else ""
            alternatives.extend(
                [
                    f"{original_query} student safety{location_suffix}",
                    f"international student {original_query}",
                ]
            )

        # Generic alternatives
        alternatives.extend(
            [
                f"{original_query} guide",
                f"{original_query} information",
                f"{original_query} tips",
            ]
        )

        return alternatives[:4]

    # ------------------------------------------------------------------
    # Topic and location search helpers
    # ------------------------------------------------------------------

    def search_by_topic(self, topic: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search specifically by settlement topic.
        :param topic: str - Settlement topic name.
        :param top_k: int - Maximum results to return.
        :return: List[Dict[str, Any]] - Ranked results for the topic.
        """
        try:
            topic_queries = {
                "housing": "accommodation housing rent apartment room student",
                "transportation": "transport matatu bus taxi travel commute",
                "education": "university college campus student academic",
                "legal": "visa permit immigration passport embassy",
                "finance": "bank money cost budget payment mpesa",
                "safety": "safe security crime police emergency",
                "healthcare": "hospital clinic doctor medical insurance",
                "culture": "culture language food custom tradition",
            }
            query = topic_queries.get(topic.lower(), topic)
            return self.search(query=query, top_k=top_k, topic_filter=topic)
        except Exception as e:
            logger.error(f"Topic search error: {e}")
            return []

    def search_by_location(
        self, location: str, query: str = "", top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search by location name.
        :param location: str - Location name to search for.
        :param query: str - Optional additional query context.
        :param top_k: int - Maximum results to return.
        :return: List[Dict[str, Any]] - Ranked results for the location.
        """
        try:
            location_query = (
                f"{query} {location}" if query else f"{location} information"
            )
            return self.search(
                query=location_query, top_k=top_k, location_filter=location
            )
        except Exception as e:
            logger.error(f"Location search error: {e}")
            return []

    # ------------------------------------------------------------------
    # Stats, optimisation, health
    # ------------------------------------------------------------------

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive collection statistics.
        :return: Dict[str, Any] - Stats including count, topic distribution,
                 and average settlement score.
        """
        try:
            total_count = self.collection.count()
            stats: Dict[str, Any] = {
                "name": self.collection_name,
                "count": total_count,
                "dimension": self.dimension,
                "settlement_optimized": True,
                "bm25_index_active": self._bm25_index is not None,
                "reranker_active": VectorDBService._reranker is not None,
            }

            if total_count > 0:
                sample_results = self.collection.query(
                    query_embeddings=[[0.0] * self.dimension],
                    n_results=min(100, total_count),
                    include=["metadatas"],
                )

                if sample_results and sample_results["metadatas"]:
                    topic_distribution: Dict[str, int] = {}
                    settlement_scores: List[float] = []

                    for metadata in sample_results["metadatas"][0]:
                        if "topic_tags" in metadata:
                            try:
                                topics = json.loads(metadata["topic_tags"])
                                for t in topics:
                                    topic_distribution[t] = (
                                        topic_distribution.get(t, 0) + 1
                                    )
                            except Exception:
                                pass
                        if "settlement_score" in metadata:
                            settlement_scores.append(metadata["settlement_score"])

                    stats["topic_distribution"] = topic_distribution
                    if settlement_scores:
                        stats["avg_settlement_score"] = sum(settlement_scores) / len(
                            settlement_scores
                        )

            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    def optimize_collection(self) -> Dict[str, Any]:
        """
        Return collection status with planned optimisation notes.
        The BM25 hybrid retrieval and cross-encoder reranker introduced in
        Milestone 7 are active; this endpoint records their availability.
        :return: Dict[str, Any] - Status and optimisation notes.
        """
        try:
            stats = self.get_collection_stats()
            return {
                "status": "no_op",
                "collection_stats": stats,
                "optimizations_available": [
                    "BM25 hybrid retrieval with Reciprocal Rank Fusion (active)",
                    "Cross-encoder reranker (active when model loaded)",
                    "Settlement-specific metadata indexing (active)",
                    "Topic-aware scoring (active)",
                    "Location-based boosting (active)",
                ],
            }
        except Exception as e:
            logger.error(f"Collection optimization failed: {e}")
            return {"status": "failed", "error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check.
        :return: Dict[str, Any] - Health status including BM25 and reranker state.
        """
        try:
            health: Dict[str, Any] = {
                "database_accessible": True,
                "collection_exists": True,
                "vector_count": self.collection.count(),
                "embedding_service_available": True,
                "settlement_optimization_active": True,
                "bm25_index_active": self._bm25_index is not None,
                "reranker_active": VectorDBService._reranker is not None,
            }

            try:
                self.search("test query", top_k=1)
                health["search_functional"] = True
            except Exception:
                health["search_functional"] = False

            try:
                test_embedding = self.embedding_service.embed_query("test")
                health["embedding_generation_functional"] = test_embedding is not None
            except Exception:
                health["embedding_generation_functional"] = False

            health["overall_health"] = all(
                [
                    health["database_accessible"],
                    health["collection_exists"],
                    health["search_functional"],
                    health["embedding_generation_functional"],
                ]
            )

            return health

        except Exception as e:
            return {
                "overall_health": False,
                "error": str(e),
                "database_accessible": False,
            }


# ------------------------------------------------------------------
# Module-level cross-encoder initialisation (once per process)
# ------------------------------------------------------------------
try:
    from sentence_transformers import CrossEncoder as _CrossEncoder

    VectorDBService._reranker = _CrossEncoder(CROSS_ENCODER_MODEL)
    logger.info(f"Cross-encoder initialized: {CROSS_ENCODER_MODEL}")
except Exception as _ce_exc:
    logger.warning(f"Cross-encoder init failed: {_ce_exc}")
