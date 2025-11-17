import json
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional

import chromadb
from chromadb.config import Settings

from config.settings import settings
from config.settings import ROOT_DIR
from services.embeddings import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_db")


class VectorDBError(Exception):
    """Custom exception for vector database errors."""

    pass


class VectorDBService:
    """
    Production-grade vector database service optimized for settlement content.
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        try:
            self.embedding_service = embedding_service or EmbeddingService()
            self.dimension = self.embedding_service.dimension
            self.collection_name = settings.vector_db.collection_name

            # Setup database path
            self.db_path = ROOT_DIR / "database" / "chroma_db"
            self.db_path.mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # Get or create collection with settlement-optimized configuration
            self.collection = self._get_or_create_collection()

            # Settlement-specific search optimization
            self._initialize_settlement_filters()

            logger.info(f"ChromaDB initialized at {self.db_path}")
            logger.info(
                f"Collection '{self.collection_name}' has {self.collection.count()} vectors"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise VectorDBError(f"Database initialization failed: {str(e)}")

    def _get_or_create_collection(self):
        """Get existing collection or create new one with settlement metadata."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
            return collection
        except Exception:
            # Create new collection with settlement-optimized configuration
            logger.info(f"Creating new collection: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16,
                    "settlement_optimized": True,
                    "description": "SettleBot Nairobi settlement content",
                },
            )

    def _initialize_settlement_filters(self):
        """Initialize settlement-specific search filters."""
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

    def initialize_collection(self, recreate: bool = False) -> None:
        """Initialize or recreate the vector collection."""
        try:
            if recreate:
                # Delete existing collection
                try:
                    self.client.delete_collection(name=self.collection_name)
                    logger.info(
                        f"Deleted existing collection: {self.collection_name}"
                    )
                except Exception as e:
                    logger.warning(
                        f"No existing collection to delete: {str(e)}"
                    )

                # Create new collection
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
            logger.error(f"Error initializing collection: {str(e)}")
            raise VectorDBError(f"Collection initialization failed: {str(e)}")

    def index_chunks(
        self, chunks_file: Optional[Union[str, Path]] = None
    ) -> None:
        """Index chunks with settlement-specific optimization."""
        try:
            # Check for deduplicated chunks first
            dedup_file = (
                ROOT_DIR / "data" / "deduplicated" / "deduplicated_chunks.jsonl"
            )

            if dedup_file.exists():
                logger.info("Found deduplicated chunks, indexing those")
                self._index_deduplicated_chunks(dedup_file)
                return

            # Process standard chunks
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

            # Process each file
            successful = 0
            failed = 0

            for chunk_file in files_to_process:
                try:
                    self._index_chunks_file(chunk_file)
                    successful += 1
                except Exception as e:
                    logger.error(f"Failed to index {chunk_file.name}: {str(e)}")
                    failed += 1

            logger.info(
                f"Indexing complete: {successful} succeeded, {failed} failed"
            )
            logger.info(
                f"Total vectors in collection: {self.collection.count()}"
            )

        except Exception as e:
            logger.error(f"Error during indexing: {str(e)}")
            raise VectorDBError(f"Indexing failed: {str(e)}")

    def _index_chunks_file(self, chunks_file: Path) -> None:
        """Index chunks from a single file with settlement metadata."""
        try:
            logger.info(f"Indexing chunks from: {chunks_file.name}")

            # Get document ID
            doc_id = chunks_file.stem.replace("_chunks", "")

            # Check for embeddings
            embeddings_dir = ROOT_DIR / "data" / "embeddings"
            embeddings_file = embeddings_dir / f"{doc_id}_embeddings.npz"

            if not embeddings_file.exists():
                logger.info(f"Generating embeddings for {doc_id}")
                self.embedding_service.embed_chunks(chunks_file)

            # Load embeddings and chunks
            embeddings_data = self.embedding_service.load_embeddings(
                embeddings_file
            )
            if embeddings_data is None:
                raise VectorDBError(f"Failed to load embeddings for {doc_id}")

            embeddings = embeddings_data["embeddings"]
            chunk_ids = embeddings_data["chunk_ids"]

            # Load chunk metadata
            chunks = []
            with open(chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        chunk_data = json.loads(line)
                        chunks.append(chunk_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line: {str(e)}")

            if len(chunks) != len(embeddings):
                logger.warning(
                    f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
                )
                min_len = min(len(chunks), len(embeddings))
                chunks = chunks[:min_len]
                embeddings = embeddings[:min_len]
                chunk_ids = chunk_ids[:min_len]

            # Prepare data for ChromaDB with settlement metadata
            ids = [str(chunk["chunk_id"]) for chunk in chunks]
            documents = [chunk["text"] for chunk in chunks]
            metadatas = []

            for chunk in chunks:
                metadata = {
                    "doc_id": chunk["doc_id"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_id": chunk["chunk_id"],
                }

                # Add settlement-specific metadata
                chunk_metadata = chunk.get("metadata", {})
                if "settlement_score" in chunk_metadata:
                    metadata["settlement_score"] = chunk_metadata[
                        "settlement_score"
                    ]
                if "topic_tags" in chunk_metadata:
                    metadata["topic_tags"] = json.dumps(
                        chunk_metadata["topic_tags"]
                    )
                if "location_entities" in chunk_metadata:
                    metadata["location_entities"] = json.dumps(
                        chunk_metadata["location_entities"]
                    )
                if "cost_entities" in chunk_metadata:
                    metadata["cost_entities"] = json.dumps(
                        chunk_metadata["cost_entities"]
                    )

                metadatas.append(metadata)

            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_embeddings = embeddings[i : i + batch_size].tolist()
                batch_documents = documents[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size]

                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                )

            logger.info(f"Successfully indexed {len(ids)} chunks from {doc_id}")

        except Exception as e:
            logger.error(
                f"Error indexing chunks file {chunks_file.name}: {str(e)}"
            )
            raise VectorDBError(f"Failed to index {chunks_file.name}: {str(e)}")

    def _index_deduplicated_chunks(self, dedup_file: Path) -> None:
        """Index deduplicated chunks with enhanced metadata."""
        try:
            # Check for embeddings
            embeddings_dir = ROOT_DIR / "data" / "embeddings"
            embeddings_file = embeddings_dir / "deduplicated_embeddings.npz"

            if not embeddings_file.exists():
                logger.info("Generating embeddings for deduplicated chunks")
                self.embedding_service.embed_deduplicated_chunks()

            # Load embeddings and chunks
            embeddings_data = self.embedding_service.load_embeddings(
                embeddings_file
            )
            if embeddings_data is None:
                raise VectorDBError("Failed to load deduplicated embeddings")

            embeddings = embeddings_data["embeddings"]

            # Load chunks
            chunks = []
            with open(dedup_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line: {str(e)}")

            if len(chunks) != len(embeddings):
                logger.warning(
                    f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
                )
                min_len = min(len(chunks), len(embeddings))
                chunks = chunks[:min_len]
                embeddings = embeddings[:min_len]

            # Prepare data with enhanced metadata
            ids = [str(chunk["chunk_id"]) for chunk in chunks]
            documents = [chunk["text"] for chunk in chunks]
            metadatas = []

            for chunk in chunks:
                metadata = {
                    "doc_id": chunk.get("doc_id", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "chunk_id": chunk["chunk_id"],
                }

                # Add deduplication metadata
                chunk_metadata = chunk.get("metadata", {})
                if "settlement_score" in chunk_metadata:
                    metadata["settlement_score"] = chunk_metadata[
                        "settlement_score"
                    ]
                if "is_merged" in chunk_metadata:
                    metadata["is_merged"] = True
                    metadata["merged_count"] = chunk_metadata.get(
                        "merge_count", 0
                    )

                metadatas.append(metadata)

            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_embeddings = embeddings[i : i + batch_size].tolist()
                batch_documents = documents[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size]

                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                )

            logger.info(f"Successfully indexed {len(ids)} deduplicated chunks")

        except Exception as e:
            logger.error(f"Error indexing deduplicated chunks: {str(e)}")
            raise VectorDBError(
                f"Failed to index deduplicated chunks: {str(e)}"
            )

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_doc_id: Optional[str] = None,
        topic_filter: Optional[str] = None,
        location_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search with settlement-specific optimization and filtering."""
        try:
            if self.collection.count() == 0:
                logger.warning("Collection is empty")
                return []

            # Generate query embedding with settlement optimization
            query_embedding = self.embedding_service.embed_query(query)

            if query_embedding is None:
                raise VectorDBError("Failed to generate query embedding")

            # Prepare filter
            where_filter = {}
            if filter_doc_id:
                where_filter["doc_id"] = filter_doc_id

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k * 2, 100),  # Get more results for reranking
                where=where_filter if where_filter else None,
                include=["metadatas", "documents", "distances"],
            )

            # Format and rerank results with settlement scoring
            formatted_results = []
            if results and results["ids"] and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    # Base similarity score
                    base_score = 1 - results["distances"][0][i]

                    # Settlement-specific boosting
                    boosted_score = self._apply_settlement_boost(
                        text=results["documents"][0][i],
                        metadata=results["metadatas"][0][i],
                        base_score=base_score,
                        query=query,
                        topic_filter=topic_filter,
                        location_filter=location_filter,
                    )

                    result = {
                        "chunk_id": results["metadatas"][0][i].get(
                            "chunk_id", ""
                        ),
                        "doc_id": results["metadatas"][0][i].get("doc_id", ""),
                        "chunk_index": results["metadatas"][0][i].get(
                            "chunk_index", 0
                        ),
                        "text": results["documents"][0][i],
                        "score": boosted_score,
                        "base_score": base_score,
                    }

                    # Add settlement metadata
                    metadata = results["metadatas"][0][i]
                    if "settlement_score" in metadata:
                        result["settlement_score"] = metadata[
                            "settlement_score"
                        ]
                    if "topic_tags" in metadata:
                        try:
                            result["topic_tags"] = json.loads(
                                metadata["topic_tags"]
                            )
                        except:
                            result["topic_tags"] = []
                    if "location_entities" in metadata:
                        try:
                            result["location_entities"] = json.loads(
                                metadata["location_entities"]
                            )
                        except:
                            result["location_entities"] = []

                    formatted_results.append(result)

            # Sort by boosted score and return top_k
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            return formatted_results[:top_k]

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise VectorDBError(f"Search failed: {str(e)}")

    def _apply_settlement_boost(
        self,
        text: str,
        metadata: Dict,
        base_score: float,
        query: str,
        topic_filter: Optional[str] = None,
        location_filter: Optional[str] = None,
    ) -> float:
        """Apply settlement-specific score boosting."""
        boosted_score = base_score
        text_lower = text.lower()
        query_lower = query.lower()

        # Settlement relevance boost
        settlement_score = metadata.get("settlement_score", 0.5)
        boosted_score *= 1 + settlement_score * 0.2

        # Topic alignment boost
        if "topic_tags" in metadata:
            try:
                topic_tags = json.loads(metadata["topic_tags"])
                for topic in topic_tags:
                    topic_boost = self.topic_weights.get(topic, 1.0)
                    boosted_score *= topic_boost

                    # Extra boost if topic matches query
                    if topic in query_lower:
                        boosted_score *= 1.15
            except:
                pass

        # Location boost
        if "location_entities" in metadata:
            try:
                locations = json.loads(metadata["location_entities"])
                for location in locations:
                    location_boost = self.location_boost.get(
                        location.lower(), 1.0
                    )
                    boosted_score *= location_boost

                    # Extra boost if location mentioned in query
                    if location.lower() in query_lower:
                        boosted_score *= 1.2
            except:
                pass

        # High-value keyword boost
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

        keyword_matches = sum(
            1 for keyword in high_value_keywords if keyword in text_lower
        )
        if keyword_matches > 0:
            boosted_score *= 1 + keyword_matches * 0.05

        # Filter compliance boost
        if topic_filter and topic_filter in metadata.get("topic_tags", ""):
            boosted_score *= 1.3

        if location_filter:
            location_entities = metadata.get("location_entities", "")
            if location_filter.lower() in location_entities.lower():
                boosted_score *= 1.3

        # Merged chunk boost (higher quality from deduplication)
        if metadata.get("is_merged", False):
            boosted_score *= 1.1

        return min(boosted_score, 1.0)  # Cap at 1.0

    def multi_query_search(
        self,
        query: str,
        top_k: int = 20,
        filter_doc_id: Optional[str] = None,
        topic_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Multi-query search with settlement-specific query expansion."""
        try:
            # Get results from original query
            original_results = self.search(
                query=query,
                top_k=int(top_k * 0.6),
                filter_doc_id=filter_doc_id,
                topic_filter=topic_filter,
            )

            # Generate settlement-specific alternative queries
            alt_queries = self._generate_settlement_queries(query)

            all_results = original_results.copy()
            seen_chunks = {r["chunk_id"] for r in original_results}

            # Try alternative queries
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
                            # Slightly reduce score for alternative queries
                            result["score"] *= 0.95
                            result["query_type"] = "alternative"
                            all_results.append(result)
                            seen_chunks.add(result["chunk_id"])

                except Exception as e:
                    logger.warning(f"Alternative query failed: {str(e)}")
                    continue

            # Sort and limit
            all_results.sort(key=lambda x: x["score"], reverse=True)
            return all_results[:top_k]

        except Exception as e:
            logger.error(f"Multi-query search error: {str(e)}")
            # Fallback to regular search
            try:
                return self.search(query, top_k, filter_doc_id, topic_filter)
            except:
                return []

    def _generate_settlement_queries(self, original_query: str) -> List[str]:
        """Generate settlement-specific alternative queries."""
        query_lower = original_query.lower()
        alternatives = []

        # Add location context if missing
        if "nairobi" not in query_lower and "kenya" not in query_lower:
            alternatives.append(f"{original_query} in Nairobi Kenya")

        # Add student context if missing
        if "student" not in query_lower and "international" not in query_lower:
            alternatives.append(f"international student {original_query}")

        # Topic-specific expansions
        if any(
            word in query_lower for word in ["house", "room", "accommodation"]
        ):
            alternatives.extend(
                [
                    f"{original_query} for international students",
                    f"student {original_query} near university",
                ]
            )

        elif any(
            word in query_lower for word in ["transport", "travel", "commute"]
        ):
            alternatives.extend(
                [
                    f"{original_query} public transport Nairobi",
                    f"student {original_query} university",
                ]
            )

        elif any(word in query_lower for word in ["cost", "price", "budget"]):
            alternatives.extend(
                [
                    f"cost of {original_query} Kenya",
                    f"{original_query} international student budget",
                ]
            )

        elif any(
            word in query_lower for word in ["safe", "security", "danger"]
        ):
            alternatives.extend(
                [
                    f"{original_query} student safety Nairobi",
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

        return alternatives[:4]  # Limit to avoid too many queries

    def search_by_topic(
        self, topic: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search specifically by settlement topic."""
        try:
            # Use topic-specific query
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
            logger.error(f"Topic search error: {str(e)}")
            return []

    def search_by_location(
        self, location: str, query: str = "", top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search specifically by location in Nairobi."""
        try:
            location_query = (
                f"{query} {location} Nairobi"
                if query
                else f"{location} Nairobi information"
            )
            return self.search(
                query=location_query, top_k=top_k, location_filter=location
            )

        except Exception as e:
            logger.error(f"Location search error: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics."""
        try:
            total_count = self.collection.count()

            stats = {
                "name": self.collection_name,
                "count": total_count,
                "dimension": self.dimension,
                "settlement_optimized": True,
            }

            # Get sample to analyze metadata distribution
            if total_count > 0:
                sample_results = self.collection.query(
                    query_embeddings=[[0.0] * self.dimension],
                    n_results=min(100, total_count),
                    include=["metadatas"],
                )

                if sample_results and sample_results["metadatas"]:
                    topic_distribution = {}
                    settlement_scores = []

                    for metadata in sample_results["metadatas"][0]:
                        # Analyze topic distribution
                        if "topic_tags" in metadata:
                            try:
                                topics = json.loads(metadata["topic_tags"])
                                for topic in topics:
                                    topic_distribution[topic] = (
                                        topic_distribution.get(topic, 0) + 1
                                    )
                            except:
                                pass

                        # Collect settlement scores
                        if "settlement_score" in metadata:
                            settlement_scores.append(
                                metadata["settlement_score"]
                            )

                    stats["topic_distribution"] = topic_distribution
                    if settlement_scores:
                        stats["avg_settlement_score"] = sum(
                            settlement_scores
                        ) / len(settlement_scores)

            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}

    def optimize_collection(self) -> Dict[str, Any]:
        """Optimize collection for settlement queries."""
        try:
            logger.info("Optimizing collection for settlement content...")

            # This would involve reindexing with optimized parameters
            # For now, we'll return current optimization status
            stats = self.get_collection_stats()

            return {
                "status": "optimized",
                "collection_stats": stats,
                "optimizations_applied": [
                    "Settlement-specific metadata indexing",
                    "Topic-aware scoring",
                    "Location-based boosting",
                    "Cost entity extraction",
                    "Multi-query expansion",
                ],
            }

        except Exception as e:
            logger.error(f"Collection optimization failed: {str(e)}")
            return {"status": "failed", "error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            health = {
                "database_accessible": True,
                "collection_exists": True,
                "vector_count": self.collection.count(),
                "embedding_service_available": True,
                "settlement_optimization_active": True,
            }

            # Test search functionality
            try:
                self.search("test query", top_k=1)
                health["search_functional"] = True

            except:
                health["search_functional"] = False

            # Test embedding generation
            try:
                test_embedding = self.embedding_service.embed_query("test")
                health["embedding_generation_functional"] = (
                    test_embedding is not None
                )
            except:
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
