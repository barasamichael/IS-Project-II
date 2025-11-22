import os
import re
import json
import logging
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from dataclasses import dataclass

import nltk
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic_chunking")

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class ChunkingStrategy(str, Enum):
    SEMANTIC_ADAPTIVE = "semantic_adaptive"
    SEMANTIC_FIXED = "semantic_fixed"
    SETTLEMENT_OPTIMIZED = "settlement_optimized"
    TOPIC_AWARE = "topic_aware"


class ChunkType(str, Enum):
    PARAGRAPH = "paragraph"
    SECTION = "section"
    TOPIC_SEGMENT = "topic_segment"
    MERGED = "merged"
    SPLIT = "split"


@dataclass
class SemanticChunk:
    """Represents a semantic chunk with metadata."""

    chunk_id: str
    doc_id: str
    text: str
    start_pos: int
    end_pos: int
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    semantic_score: float = 0.0
    topic_coherence: float = 0.0
    settlement_relevance: float = 0.0
    word_count: int = 0
    char_count: int = 0

    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = len(self.text.split())
        if self.char_count == 0:
            self.char_count = len(self.text)


class SemanticChunker:
    """
    LLM-powered semantic chunker optimized for settlement content.
    Uses OpenAI API for semantic analysis instead of local models.
    """

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.SETTLEMENT_OPTIMIZED,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        target_chunk_size: int = 512,
        overlap_size: int = 50,
        settlement_optimization: bool = True,
    ):
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)

        # Configuration
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size
        self.settlement_optimization = settlement_optimization

        # Settlement-specific patterns and topics
        self._initialize_settlement_patterns()

        logger.info(
            f"LLM-based SemanticChunker initialized with strategy: {strategy}"
        )

    def _initialize_settlement_patterns(self):
        """Initialize settlement-specific patterns for content analysis."""
        self.settlement_topics = {
            "housing": [
                "accommodation",
                "housing",
                "rent",
                "apartment",
                "room",
                "bedsitter",
                "studio",
                "landlord",
                "tenant",
                "deposit",
                "utilities",
                "furnished",
                "lease",
                "rental",
                "dormitory",
                "hostel",
            ],
            "transportation": [
                "transport",
                "matatu",
                "bus",
                "taxi",
                "uber",
                "bolt",
                "boda",
                "motorcycle",
                "public transport",
                "route",
                "fare",
                "commute",
            ],
            "education": [
                "university",
                "college",
                "campus",
                "admission",
                "enrollment",
                "academic",
                "semester",
                "library",
                "tuition",
                "fees",
                "student services",
            ],
            "legal": [
                "visa",
                "permit",
                "passport",
                "immigration",
                "embassy",
                "consulate",
                "legal",
                "requirements",
                "documents",
                "renewal",
                "extension",
            ],
            "finance": [
                "bank",
                "banking",
                "account",
                "money",
                "cost",
                "budget",
                "mpesa",
                "currency",
                "exchange",
                "payment",
                "transfer",
                "loan",
            ],
            "safety": [
                "safe",
                "safety",
                "security",
                "crime",
                "police",
                "emergency",
                "dangerous",
                "avoid",
                "caution",
                "guard",
            ],
            "healthcare": [
                "hospital",
                "clinic",
                "medical",
                "health",
                "doctor",
                "insurance",
                "medicine",
                "pharmacy",
                "vaccination",
            ],
            "culture": [
                "culture",
                "language",
                "custom",
                "tradition",
                "food",
                "religion",
                "etiquette",
                "adaptation",
                "social",
            ],
        }

        self.nairobi_locations = [
            "Westlands",
            "Kilimani",
            "Karen",
            "Lavington",
            "Kileleshwa",
            "Parklands",
            "Hurlingham",
            "Riverside",
            "Runda",
            "Muthaiga",
            "Gigiri",
            "Spring Valley",
            "CBD",
            "Eastleigh",
            "Kasarani",
            "Ruiru",
            "Ngong",
            "Langata",
        ]

        self.topic_transition_markers = [
            "however",
            "furthermore",
            "moreover",
            "additionally",
            "on the other hand",
            "in contrast",
            "similarly",
            "meanwhile",
            "next",
            "finally",
            "in conclusion",
            "moving on",
            "another aspect",
            "regarding",
            "concerning",
            "with respect to",
        ]

    def create_chunks(
        self, text: str, doc_id: str, preserve_structure: bool = True
    ) -> List[SemanticChunk]:
        """
        Create semantic chunks using LLM-powered analysis.

        Args:
            text: Input text to chunk
            doc_id: Document identifier
            preserve_structure: Whether to preserve document structure

        Returns:
            List of semantic chunks
        """
        if not text or not text.strip():
            return []

        try:
            # Preprocess text
            cleaned_text = self._preprocess_text(text)

            # Choose chunking approach based on strategy
            if self.strategy == ChunkingStrategy.SETTLEMENT_OPTIMIZED:
                chunks = self._settlement_optimized_chunking(
                    cleaned_text, doc_id
                )
            elif self.strategy == ChunkingStrategy.TOPIC_AWARE:
                chunks = self._topic_aware_chunking(cleaned_text, doc_id)
            elif self.strategy == ChunkingStrategy.SEMANTIC_ADAPTIVE:
                chunks = self._semantic_adaptive_chunking(cleaned_text, doc_id)
            else:
                chunks = self._semantic_fixed_chunking(cleaned_text, doc_id)

            # Post-process and validate chunks
            validated_chunks = self._validate_and_fix_chunks(chunks)

            # Calculate semantic scores using LLM
            enriched_chunks = self._enrich_chunks_with_llm_analysis(
                validated_chunks
            )

            logger.info(
                f"Created {len(enriched_chunks)} semantic chunks for doc {doc_id}"
            )
            return enriched_chunks

        except Exception as e:
            logger.error(f"Error in semantic chunking: {str(e)}")
            # Fallback to simple chunking
            return self._fallback_chunking(text, doc_id)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better chunking."""
        # Normalize whitespace
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\t+", " ", text)

        # Fix common formatting issues
        text = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", text)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

        return text.strip()

    def _settlement_optimized_chunking(
        self, text: str, doc_id: str
    ) -> List[SemanticChunk]:
        """Settlement-optimized chunking using LLM for topic detection."""
        # First, get LLM analysis of the entire text
        self._analyze_text_with_llm(text)

        # Split into paragraphs
        paragraphs = self._split_into_paragraphs(text)

        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0

        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if adding this paragraph would exceed max size
            potential_chunk = (
                current_chunk + ("\n\n" if current_chunk else "") + paragraph
            )

            if (
                len(potential_chunk.split()) > self.max_chunk_size // 4
            ):  # Rough word count estimate
                # Finalize current chunk if it's substantial
                if (
                    current_chunk
                    and len(current_chunk.split()) >= self.min_chunk_size // 4
                ):
                    chunk = self._create_chunk(
                        chunk_id=f"{doc_id}_{chunk_index:04d}",
                        doc_id=doc_id,
                        text=current_chunk,
                        start_pos=current_start,
                        end_pos=current_start + len(current_chunk),
                        chunk_type=ChunkType.TOPIC_SEGMENT,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk
                current_chunk = paragraph
                current_start = text.find(paragraph, current_start)
            else:
                # Add to current chunk
                current_chunk = potential_chunk
                if not current_chunk.strip():
                    current_start = text.find(paragraph)

        # Handle remaining text
        if (
            current_chunk
            and len(current_chunk.split()) >= self.min_chunk_size // 4
        ):
            chunk = self._create_chunk(
                chunk_id=f"{doc_id}_{chunk_index:04d}",
                doc_id=doc_id,
                text=current_chunk,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk),
                chunk_type=ChunkType.TOPIC_SEGMENT,
            )
            chunks.append(chunk)

        return chunks

    def _topic_aware_chunking(
        self, text: str, doc_id: str
    ) -> List[SemanticChunk]:
        """Topic-aware chunking using LLM to identify topic boundaries."""
        # Get topic boundaries from LLM
        topic_boundaries = self._identify_topic_boundaries_with_llm(text)

        chunks = []
        chunk_index = 0

        for i, (start, end, topic) in enumerate(topic_boundaries):
            chunk_text = text[start:end].strip()

            if len(chunk_text.split()) >= self.min_chunk_size // 4:
                chunk = self._create_chunk(
                    chunk_id=f"{doc_id}_{chunk_index:04d}",
                    doc_id=doc_id,
                    text=chunk_text,
                    start_pos=start,
                    end_pos=end,
                    chunk_type=ChunkType.TOPIC_SEGMENT,
                )
                chunk.topic_coherence = (
                    0.8  # High coherence for topic-based chunks
                )
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _semantic_adaptive_chunking(
        self, text: str, doc_id: str
    ) -> List[SemanticChunk]:
        """Adaptive chunking that adjusts size based on content complexity."""
        # Analyze text complexity with LLM
        complexity_analysis = self._analyze_complexity_with_llm(text)

        # Adjust chunk sizes based on complexity
        if complexity_analysis.get("complexity", "medium") == "high":
            effective_chunk_size = self.target_chunk_size // 2
        elif complexity_analysis.get("complexity", "medium") == "low":
            effective_chunk_size = self.target_chunk_size * 2
        else:
            effective_chunk_size = self.target_chunk_size

        # Use sentence-based chunking with adaptive sizing
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0

        for sentence in sentences:
            potential_chunk = (
                current_chunk + (" " if current_chunk else "") + sentence
            )

            if len(potential_chunk) > effective_chunk_size:
                if current_chunk:
                    chunk = self._create_chunk(
                        chunk_id=f"{doc_id}_{chunk_index:04d}",
                        doc_id=doc_id,
                        text=current_chunk.strip(),
                        start_pos=current_start,
                        end_pos=current_start + len(current_chunk),
                        chunk_type=ChunkType.PARAGRAPH,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                current_chunk = sentence
                current_start = text.find(sentence, current_start)
            else:
                current_chunk = potential_chunk

        # Handle remaining text
        if current_chunk.strip():
            chunk = self._create_chunk(
                chunk_id=f"{doc_id}_{chunk_index:04d}",
                doc_id=doc_id,
                text=current_chunk.strip(),
                start_pos=current_start,
                end_pos=current_start + len(current_chunk),
                chunk_type=ChunkType.PARAGRAPH,
            )
            chunks.append(chunk)

        return chunks

    def _semantic_fixed_chunking(
        self, text: str, doc_id: str
    ) -> List[SemanticChunk]:
        """Fixed-size semantic chunking with overlap."""
        chunks = []
        chunk_index = 0
        start_pos = 0

        while start_pos < len(text):
            # Calculate end position
            end_pos = min(start_pos + self.target_chunk_size, len(text))

            # Adjust to sentence boundary if possible
            chunk_text = text[start_pos:end_pos]

            # Find last complete sentence
            sentences = self._split_into_sentences(chunk_text)
            if len(sentences) > 1:
                # Use all but potentially incomplete last sentence
                complete_sentences = sentences[:-1]
                chunk_text = " ".join(complete_sentences)
                end_pos = start_pos + len(chunk_text)

            if chunk_text.strip():
                chunk = self._create_chunk(
                    chunk_id=f"{doc_id}_{chunk_index:04d}",
                    doc_id=doc_id,
                    text=chunk_text.strip(),
                    start_pos=start_pos,
                    end_pos=end_pos,
                    chunk_type=ChunkType.SECTION,
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move start position with overlap
            start_pos = max(end_pos - self.overlap_size, start_pos + 1)

        return chunks

    def _analyze_text_with_llm(self, text: str) -> Dict[str, Any]:
        """Analyze text using LLM for settlement-specific insights."""
        prompt = f"""
Analyze this text for international student settlement content in Nairobi, Kenya.

Text: "{text[:2000]}..."

Provide analysis in this JSON format:
{{
    "main_topics": ["topic1", "topic2"],
    "settlement_relevance": 0.0-1.0,
    "primary_focus": "housing|transport|education|legal|finance|safety|culture|other",
    "contains_practical_info": true/false,
    "mentions_locations": ["location1", "location2"],
    "mentions_costs": true/false,
    "complexity": "low|medium|high"
}}

Only return the JSON, no explanations.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )

            result = json.loads(response.choices[0].message.content.strip())
            return result

        except Exception as e:
            logger.warning(f"LLM text analysis failed: {str(e)}")
            return {
                "main_topics": [],
                "settlement_relevance": 0.5,
                "primary_focus": "other",
                "contains_practical_info": False,
                "mentions_locations": [],
                "mentions_costs": False,
                "complexity": "medium",
            }

    def _identify_topic_boundaries_with_llm(
        self, text: str
    ) -> List[Tuple[int, int, str]]:
        """Identify topic boundaries using LLM analysis."""
        # For long texts, work with chunks and identify boundaries
        if len(text) > 3000:
            # Split into overlapping segments for analysis
            segments = []
            segment_size = 2000
            overlap = 200

            for i in range(0, len(text), segment_size - overlap):
                segment = text[i : i + segment_size]
                segments.append((i, segment))
        else:
            segments = [(0, text)]

        boundaries = []

        for start_pos, segment in segments:
            prompt = f"""
Identify topic transitions in this text about international student settlement in Nairobi.

Text: "{segment}"

Mark where topics change by indicating character positions and the new topic.
Format as JSON list: [
    {{"position": 150, "topic": "housing"}},
    {{"position": 450, "topic": "transport"}}
]

Topics can be: housing, transport, education, legal, finance, safety, healthcare, culture, general

Only return the JSON array, no explanations.
"""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=200,
                )

                segment_boundaries = json.loads(
                    response.choices[0].message.content.strip()
                )

                # Convert relative positions to absolute
                for boundary in segment_boundaries:
                    abs_pos = start_pos + boundary["position"]
                    if abs_pos < len(text):
                        boundaries.append(abs_pos)

            except Exception as e:
                logger.warning(f"Topic boundary detection failed: {str(e)}")
                continue

        # Convert boundaries to (start, end, topic) tuples
        if not boundaries:
            return [(0, len(text), "general")]

        boundaries.sort()
        topic_segments = []

        for i in range(len(boundaries)):
            start = boundaries[i] if i > 0 else 0
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)

            # Determine topic for this segment (simplified)
            segment_text = text[start:end]
            topic = self._determine_segment_topic(segment_text)

            topic_segments.append((start, end, topic))

        return topic_segments

    def _analyze_complexity_with_llm(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity using LLM."""
        prompt = f"""
Analyze the complexity of this settlement information text:

Text: "{text[:1500]}..."

Rate complexity based on:
- Technical language use
- Procedural complexity
- Information density
- Settlement-specific terminology

Respond with JSON:
{{
    "complexity": "low|medium|high",
    "reasoning": "brief explanation",
    "recommended_chunk_size": "small|medium|large"
}}

Only return the JSON, no explanations.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150,
            )

            result = json.loads(response.choices[0].message.content.strip())
            return result

        except Exception as e:
            logger.warning(f"Complexity analysis failed: {str(e)}")
            return {
                "complexity": "medium",
                "reasoning": "analysis failed",
                "recommended_chunk_size": "medium",
            }

    def _enrich_chunks_with_llm_analysis(
        self, chunks: List[SemanticChunk]
    ) -> List[SemanticChunk]:
        """Enrich chunks with LLM-based semantic analysis."""
        enriched_chunks = []

        for chunk in chunks:
            try:
                # Analyze chunk with LLM
                analysis = self._analyze_chunk_with_llm(chunk.text)

                # Update chunk with analysis results
                chunk.semantic_score = analysis.get("semantic_coherence", 0.7)
                chunk.topic_coherence = analysis.get("topic_coherence", 0.7)
                chunk.settlement_relevance = analysis.get(
                    "settlement_relevance", 0.5
                )

                enriched_chunks.append(chunk)

            except Exception as e:
                logger.warning(
                    f"Chunk enrichment failed for {chunk.chunk_id}: {str(e)}"
                )
                # Use default scores
                chunk.semantic_score = 0.6
                chunk.topic_coherence = 0.6
                chunk.settlement_relevance = 0.4
                enriched_chunks.append(chunk)

        return enriched_chunks

    def _analyze_chunk_with_llm(self, chunk_text: str) -> Dict[str, float]:
        """Analyze individual chunk with LLM."""
        prompt = f"""
Analyze this settlement information chunk:

Text: "{chunk_text}"

Rate on scale 0.0-1.0:
- semantic_coherence: How well the text flows and connects
- topic_coherence: How focused the text is on a single topic  
- settlement_relevance: How useful this is for international students in Nairobi

Respond with JSON:
{{
    "semantic_coherence": 0.0-1.0,
    "topic_coherence": 0.0-1.0,
    "settlement_relevance": 0.0-1.0
}}

Only return the JSON, no explanations.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
            )

            result = json.loads(response.choices[0].message.content.strip())
            return result

        except Exception as e:
            logger.warning(f"Chunk analysis failed: {str(e)}")
            return {
                "semantic_coherence": 0.6,
                "topic_coherence": 0.6,
                "settlement_relevance": 0.4,
            }

    def _determine_segment_topic(self, segment_text: str) -> str:
        """Determine the primary topic of a text segment."""
        text_lower = segment_text.lower()

        topic_scores = {}
        for topic, keywords in self.settlement_topics.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                topic_scores[topic] = score

        if topic_scores:
            return max(topic_scores, key=topic_scores.get)

        return "general"

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        try:
            from nltk.tokenize import sent_tokenize

            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed: {str(e)}")
            # Fallback to simple splitting
            sentences = re.split(r"[.!?]+", text)
            return [s.strip() for s in sentences if s.strip()]

    def _create_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        text: str,
        start_pos: int,
        end_pos: int,
        chunk_type: ChunkType,
    ) -> SemanticChunk:
        """Create a semantic chunk with basic metadata."""
        return SemanticChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            start_pos=start_pos,
            end_pos=end_pos,
            chunk_type=chunk_type,
            word_count=len(text.split()),
            char_count=len(text),
        )

    def _validate_and_fix_chunks(
        self, chunks: List[SemanticChunk]
    ) -> List[SemanticChunk]:
        """Validate and fix chunk issues."""
        validated_chunks = []

        for chunk in chunks:
            # Skip empty or too small chunks
            if chunk.word_count < self.min_chunk_size // 10:
                continue

            # Split chunks that are too large
            if chunk.word_count > self.max_chunk_size // 4:
                split_chunks = self._split_large_chunk(chunk)
                validated_chunks.extend(split_chunks)
            else:
                validated_chunks.append(chunk)

        return validated_chunks

    def _split_large_chunk(self, chunk: SemanticChunk) -> List[SemanticChunk]:
        """Split a chunk that's too large."""
        sentences = self._split_into_sentences(chunk.text)

        if len(sentences) <= 1:
            return [chunk]  # Can't split further

        split_chunks = []
        current_text = ""
        split_index = 0

        for sentence in sentences:
            potential_text = (
                current_text + (" " if current_text else "") + sentence
            )

            if len(potential_text.split()) > self.target_chunk_size // 4:
                if current_text:
                    # Create split chunk
                    split_chunk = SemanticChunk(
                        chunk_id=f"{chunk.chunk_id}_split_{split_index}",
                        doc_id=chunk.doc_id,
                        text=current_text.strip(),
                        start_pos=chunk.start_pos,  # Approximate
                        end_pos=chunk.start_pos
                        + len(current_text),  # Approximate
                        chunk_type=ChunkType.SPLIT,
                        word_count=len(current_text.split()),
                        char_count=len(current_text),
                    )
                    split_chunks.append(split_chunk)
                    split_index += 1

                current_text = sentence
            else:
                current_text = potential_text

        # Handle remaining text
        if current_text.strip():
            split_chunk = SemanticChunk(
                chunk_id=f"{chunk.chunk_id}_split_{split_index}",
                doc_id=chunk.doc_id,
                text=current_text.strip(),
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
                chunk_type=ChunkType.SPLIT,
                word_count=len(current_text.split()),
                char_count=len(current_text),
            )
            split_chunks.append(split_chunk)

        return split_chunks if split_chunks else [chunk]

    def _fallback_chunking(self, text: str, doc_id: str) -> List[SemanticChunk]:
        """Simple fallback chunking when LLM methods fail."""
        chunks = []
        chunk_index = 0
        words = text.split()

        chunk_size_words = (
            self.target_chunk_size // 4
        )  # Rough conversion to word count

        for i in range(0, len(words), chunk_size_words):
            chunk_words = words[i : i + chunk_size_words]
            chunk_text = " ".join(chunk_words)

            chunk = SemanticChunk(
                chunk_id=f"{doc_id}_{chunk_index:04d}",
                doc_id=doc_id,
                text=chunk_text,
                start_pos=i * 5,  # Rough estimate
                end_pos=(i + len(chunk_words)) * 5,  # Rough estimate
                chunk_type=ChunkType.SECTION,
                word_count=len(chunk_words),
                char_count=len(chunk_text),
            )

            chunks.append(chunk)
            chunk_index += 1

        return chunks

    def get_chunking_stats(self) -> Dict[str, Any]:
        """Get chunking configuration and statistics."""
        return {
            "strategy": self.strategy.value,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "target_chunk_size": self.target_chunk_size,
            "overlap_size": self.overlap_size,
            "settlement_optimization": self.settlement_optimization,
            "llm_powered": True,
            "supported_strategies": [
                strategy.value for strategy in ChunkingStrategy
            ],
            "settlement_topics": list(self.settlement_topics.keys()),
            "nairobi_locations_count": len(self.nairobi_locations),
        }
