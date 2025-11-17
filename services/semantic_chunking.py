import logging
from typing import Dict
from typing import List
from typing import Tuple
from dataclasses import dataclass

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic_chunking")

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


@dataclass
class Chunk:
    """Represents a semantic chunk with metadata."""

    text: str
    start_idx: int
    end_idx: int
    sentence_count: int
    semantic_score: float
    topic_coherence: float
    settlement_relevance: float
    chunk_type: str  # 'semantic' or 'fixed'

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "sentence_count": self.sentence_count,
            "semantic_score": self.semantic_score,
            "topic_coherence": self.topic_coherence,
            "settlement_relevance": self.settlement_relevance,
            "chunk_type": self.chunk_type,
        }


class SemanticChunker:
    """
    Semantic chunking implementation optimized for settlement-related content.
    """

    def __init__(self):
        self.strategy = settings.chunking.strategy
        self.chunk_size = settings.chunking.chunk_size
        self.chunk_overlap = settings.chunking.chunk_overlap
        self.semantic_threshold = settings.chunking.semantic_threshold

        # Load lightweight sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.model_available = True
        except Exception as e:
            logger.warning(f"Sentence transformer not available: {e}")
            self.model_available = False

        # Settlement-specific topic indicators
        self.topic_indicators = {
            "housing": [
                "accommodation",
                "housing",
                "rent",
                "apartment",
                "room",
                "landlord",
                "deposit",
                "utilities",
                "furnished",
                "bedsitter",
                "studio",
            ],
            "transport": [
                "transport",
                "matatu",
                "bus",
                "taxi",
                "uber",
                "boda",
                "route",
                "fare",
                "commute",
                "travel",
                "direction",
            ],
            "finance": [
                "cost",
                "price",
                "money",
                "budget",
                "bank",
                "mpesa",
                "payment",
                "currency",
                "exchange",
                "ATM",
                "account",
            ],
            "safety": [
                "safe",
                "safety",
                "security",
                "crime",
                "theft",
                "police",
                "dangerous",
                "avoid",
                "risk",
                "emergency",
            ],
            "education": [
                "university",
                "college",
                "campus",
                "student",
                "academic",
                "enrollment",
                "registration",
                "semester",
                "course",
                "tuition",
            ],
            "legal": [
                "visa",
                "permit",
                "passport",
                "immigration",
                "embassy",
                "documentation",
                "requirements",
                "renewal",
                "extension",
            ],
            "healthcare": [
                "hospital",
                "clinic",
                "doctor",
                "medical",
                "insurance",
                "pharmacy",
                "medicine",
                "health",
                "treatment",
            ],
            "culture": [
                "culture",
                "language",
                "custom",
                "tradition",
                "religion",
                "food",
                "dress",
                "etiquette",
                "adaptation",
            ],
        }

    def create_chunks(self, text: str, doc_id: str) -> List[Chunk]:
        """
        Create optimized chunks based on the configured strategy.

        Args:
            text: Document text to chunk
            doc_id: Document identifier

        Returns:
            List of Chunk objects
        """
        if self.strategy == "semantic" and self.model_available:
            return self._semantic_chunking(text)
        else:
            return self._fixed_size_chunking(text)

    def _semantic_chunking(self, text: str) -> List[Chunk]:
        """Create semantically coherent chunks."""
        try:
            # Split into sentences
            sentences = sent_tokenize(text)
            if len(sentences) < 2:
                return self._create_single_chunk(text, "semantic")

            # Get sentence embeddings
            embeddings = self.sentence_model.encode(sentences)

            # Calculate semantic similarity between adjacent sentences
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[i + 1].reshape(1, -1),
                )[0][0]
                similarities.append(sim)

            # Find chunk boundaries based on semantic breaks
            boundaries = self._find_semantic_boundaries(similarities, sentences)

            # Create chunks from boundaries
            chunks = []
            for i, (start, end) in enumerate(boundaries):
                chunk_sentences = sentences[start : end + 1]
                chunk_text = " ".join(chunk_sentences)

                # Calculate chunk quality metrics
                semantic_score = self._calculate_semantic_coherence(
                    chunk_sentences
                )
                topic_coherence = self._calculate_topic_coherence(chunk_text)
                settlement_relevance = self._calculate_settlement_relevance(
                    chunk_text
                )

                chunk = Chunk(
                    text=chunk_text,
                    start_idx=start,
                    end_idx=end,
                    sentence_count=len(chunk_sentences),
                    semantic_score=semantic_score,
                    topic_coherence=topic_coherence,
                    settlement_relevance=settlement_relevance,
                    chunk_type="semantic",
                )
                chunks.append(chunk)

            return self._optimize_chunk_sizes(chunks, sentences)

        except Exception as e:
            logger.error(f"Semantic chunking failed: {e}")
            return self._fixed_size_chunking(text)

    def _find_semantic_boundaries(
        self, similarities: List[float], sentences: List[str]
    ) -> List[Tuple[int, int]]:
        """Find optimal chunk boundaries based on semantic similarity."""
        boundaries = []
        current_start = 0
        current_length = 0

        for i, similarity in enumerate(similarities):
            current_length += len(sentences[i])

            # Check for semantic break or size limit
            is_semantic_break = similarity < self.semantic_threshold
            is_size_limit = current_length > self.chunk_size
            is_topic_boundary = self._is_topic_boundary(
                sentences[i], sentences[i + 1]
            )

            if (
                is_semantic_break or is_size_limit or is_topic_boundary
            ) and i > current_start:
                boundaries.append((current_start, i))
                current_start = i + 1
                current_length = 0

        # Add final chunk
        if current_start < len(sentences):
            boundaries.append((current_start, len(sentences) - 1))

        return boundaries

    def _is_topic_boundary(self, sent1: str, sent2: str) -> bool:
        """Detect topic boundaries using settlement-specific indicators."""
        sent1_topics = self._extract_topics(sent1)
        sent2_topics = self._extract_topics(sent2)

        # If topics are completely different, it's likely a boundary
        if sent1_topics and sent2_topics:
            overlap = set(sent1_topics).intersection(set(sent2_topics))
            if not overlap:
                return True

        # Check for transition phrases
        transition_phrases = [
            "on the other hand",
            "however",
            "meanwhile",
            "in contrast",
            "moving on to",
            "next topic",
            "another important",
            "different aspect",
            "switching to",
        ]

        return any(phrase in sent2.lower() for phrase in transition_phrases)

    def _extract_topics(self, sentence: str) -> List[str]:
        """Extract topics from sentence using indicator keywords."""
        topics = []
        sentence_lower = sentence.lower()

        for topic, indicators in self.topic_indicators.items():
            if any(indicator in sentence_lower for indicator in indicators):
                topics.append(topic)

        return topics

    def _calculate_semantic_coherence(self, sentences: List[str]) -> float:
        """Calculate semantic coherence score for chunk sentences."""
        if len(sentences) < 2 or not self.model_available:
            return 0.5

        try:
            embeddings = self.sentence_model.encode(sentences)
            similarities = []

            for i in range(len(embeddings) - 1):
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[i + 1].reshape(1, -1),
                )[0][0]
                similarities.append(sim)

            return float(np.mean(similarities))

        except Exception:
            return 0.5

    def _calculate_topic_coherence(self, text: str) -> float:
        """Calculate topic coherence based on consistent theme."""
        topics_found = self._extract_topics(text)

        if not topics_found:
            return 0.3  # Low coherence if no clear topics

        # Higher coherence for single topic, lower for mixed topics
        unique_topics = set(topics_found)
        coherence = 1.0 / len(unique_topics) if unique_topics else 0.3

        return min(coherence, 1.0)

    def _calculate_settlement_relevance(self, text: str) -> float:
        """Calculate relevance to international student settlement."""
        text_lower = text.lower()

        # Settlement-specific keywords with weights
        keyword_weights = {
            # High relevance
            "international student": 3.0,
            "nairobi": 2.5,
            "kenya": 2.0,
            "university": 2.0,
            "accommodation": 2.5,
            "visa": 2.5,
            "immigration": 2.5,
            # Medium relevance
            "housing": 2.0,
            "transport": 1.5,
            "safety": 2.0,
            "cost": 1.5,
            "bank": 1.5,
            "hospital": 1.5,
            "culture": 1.5,
            # Lower relevance
            "student": 1.0,
            "education": 1.0,
            "money": 1.0,
            "language": 1.0,
        }

        total_score = 0.0
        total_weight = 0.0

        for keyword, weight in keyword_weights.items():
            if keyword in text_lower:
                count = text_lower.count(keyword)
                total_score += count * weight
                total_weight += weight

        # Normalize score
        if total_weight > 0:
            return min(total_score / total_weight, 1.0)

        return 0.2  # Default low relevance

    def _optimize_chunk_sizes(
        self, chunks: List[Chunk], sentences: List[str]
    ) -> List[Chunk]:
        """Optimize chunk sizes by merging small chunks or splitting large ones."""
        optimized_chunks = []
        min_chunk_size = self.chunk_size // 4
        max_chunk_size = self.chunk_size * 1.5

        i = 0
        while i < len(chunks):
            current_chunk = chunks[i]

            # If chunk is too small, try to merge with next
            if len(current_chunk.text) < min_chunk_size and i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                merged_text = current_chunk.text + " " + next_chunk.text

                # Only merge if combined size is reasonable
                if len(merged_text) <= max_chunk_size:
                    merged_chunk = Chunk(
                        text=merged_text,
                        start_idx=current_chunk.start_idx,
                        end_idx=next_chunk.end_idx,
                        sentence_count=current_chunk.sentence_count
                        + next_chunk.sentence_count,
                        semantic_score=(
                            current_chunk.semantic_score
                            + next_chunk.semantic_score
                        )
                        / 2,
                        topic_coherence=(
                            current_chunk.topic_coherence
                            + next_chunk.topic_coherence
                        )
                        / 2,
                        settlement_relevance=max(
                            current_chunk.settlement_relevance,
                            next_chunk.settlement_relevance,
                        ),
                        chunk_type="semantic",
                    )
                    optimized_chunks.append(merged_chunk)
                    i += 2  # Skip next chunk as it's been merged
                    continue

            # If chunk is too large, split it
            if len(current_chunk.text) > max_chunk_size:
                split_chunks = self._split_large_chunk(current_chunk, sentences)
                optimized_chunks.extend(split_chunks)
            else:
                optimized_chunks.append(current_chunk)

            i += 1

        return optimized_chunks

    def _split_large_chunk(
        self, chunk: Chunk, sentences: List[str]
    ) -> List[Chunk]:
        """Split a large chunk into smaller semantic units."""
        chunk_sentences = sentences[chunk.start_idx : chunk.end_idx + 1]

        if len(chunk_sentences) <= 2:
            return [chunk]  # Can't split further

        # Find midpoint with topic boundary if possible
        mid_point = len(chunk_sentences) // 2

        # Look for topic boundary near midpoint
        for offset in range(3):  # Check 3 positions around midpoint
            for direction in [-1, 1]:
                check_idx = mid_point + (offset * direction)
                if 0 < check_idx < len(chunk_sentences) - 1:
                    if self._is_topic_boundary(
                        chunk_sentences[check_idx],
                        chunk_sentences[check_idx + 1],
                    ):
                        mid_point = check_idx + 1
                        break

        # Create two chunks
        first_chunk = Chunk(
            text=" ".join(chunk_sentences[:mid_point]),
            start_idx=chunk.start_idx,
            end_idx=chunk.start_idx + mid_point - 1,
            sentence_count=mid_point,
            semantic_score=chunk.semantic_score
            * 0.9,  # Slightly reduce score for split
            topic_coherence=chunk.topic_coherence,
            settlement_relevance=chunk.settlement_relevance,
            chunk_type="semantic",
        )

        second_chunk = Chunk(
            text=" ".join(chunk_sentences[mid_point:]),
            start_idx=chunk.start_idx + mid_point,
            end_idx=chunk.end_idx,
            sentence_count=len(chunk_sentences) - mid_point,
            semantic_score=chunk.semantic_score * 0.9,
            topic_coherence=chunk.topic_coherence,
            settlement_relevance=chunk.settlement_relevance,
            chunk_type="semantic",
        )

        return [first_chunk, second_chunk]

    def _fixed_size_chunking(self, text: str) -> List[Chunk]:
        """Create fixed-size chunks with overlap as fallback."""
        chunks = []
        text_length = len(text)
        start = 0
        chunk_idx = 0

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            # Adjust end to sentence boundary if possible
            chunk_text = text[start:end]

            # Try to end at sentence boundary
            if end < text_length:
                last_period = chunk_text.rfind(".")
                last_newline = chunk_text.rfind("\n")
                boundary = max(last_period, last_newline)

                if (
                    boundary > start + self.chunk_size // 2
                ):  # Don't make chunks too small
                    end = start + boundary + 1
                    chunk_text = text[start:end]

            # Calculate basic metrics
            settlement_relevance = self._calculate_settlement_relevance(
                chunk_text
            )

            chunk = Chunk(
                text=chunk_text.strip(),
                start_idx=start,
                end_idx=end - 1,
                sentence_count=len(sent_tokenize(chunk_text)),
                semantic_score=0.5,  # Default for fixed chunking
                topic_coherence=0.5,
                settlement_relevance=settlement_relevance,
                chunk_type="fixed",
            )

            chunks.append(chunk)

            # Calculate next start with overlap
            start = max(end - self.chunk_overlap, start + 1)
            chunk_idx += 1

        return chunks

    def _create_single_chunk(self, text: str, chunk_type: str) -> List[Chunk]:
        """Create a single chunk for very short texts."""
        chunk = Chunk(
            text=text,
            start_idx=0,
            end_idx=len(text) - 1,
            sentence_count=len(sent_tokenize(text)),
            semantic_score=1.0,
            topic_coherence=1.0,
            settlement_relevance=self._calculate_settlement_relevance(text),
            chunk_type=chunk_type,
        )
        return [chunk]

    def get_chunking_stats(self) -> Dict[str, any]:
        """Get statistics about the chunking configuration."""
        return {
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "semantic_threshold": self.semantic_threshold,
            "model_available": self.model_available,
            "topic_indicators": len(self.topic_indicators),
            "supported_features": [
                "Semantic coherence detection",
                "Topic boundary detection",
                "Settlement relevance scoring",
                "Adaptive chunk sizing",
                "Quality optimization",
            ],
        }
