import os
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from datetime import datetime
from dataclasses import dataclass

import spacy
import nltk
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import SitemapLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredEmailLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

from config.settings import ROOT_DIR
from services.semantic_chunking import SemanticChunker
from services.embeddings import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("document_processor")

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


@dataclass
class ProcessedChunk:
    """Represents a processed chunk with settlement-specific metadata."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any]
    settlement_score: float = 0.0
    topic_tags: List[str] = None
    location_entities: List[str] = None
    cost_entities: List[str] = None

    def __post_init__(self):
        if self.topic_tags is None:
            self.topic_tags = []
        if self.location_entities is None:
            self.location_entities = []
        if self.cost_entities is None:
            self.cost_entities = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "metadata": {
                **self.metadata,
                "settlement_score": self.settlement_score,
                "topic_tags": self.topic_tags,
                "location_entities": self.location_entities,
                "cost_entities": self.cost_entities,
            },
        }


class DocumentProcessor:
    """
    Production-grade document processor optimized for settlement information.
    """

    def __init__(
        self,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        chunk_dir: Optional[Union[str, Path]] = None,
        dedup_dir: Optional[Union[str, Path]] = None,
        embedding_service: Optional[EmbeddingService] = None,
        enable_deduplication: bool = True,
        similarity_threshold: float = 0.92,
        pdf_loader_strategy: str = "auto",
    ):
        # Initialize paths
        self.raw_dir = Path(raw_dir) if raw_dir else ROOT_DIR / "data" / "raw"
        self.processed_dir = (
            Path(processed_dir)
            if processed_dir
            else ROOT_DIR / "data" / "processed"
        )
        self.chunk_dir = (
            Path(chunk_dir) if chunk_dir else ROOT_DIR / "data" / "chunks"
        )
        self.dedup_dir = (
            Path(dedup_dir) if dedup_dir else ROOT_DIR / "data" / "deduplicated"
        )

        # Create directories
        for dir_path in [
            self.raw_dir,
            self.processed_dir,
            self.chunk_dir,
            self.dedup_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize services
        self.semantic_chunker = SemanticChunker()
        self.embedding_service = embedding_service or EmbeddingService()

        # Configuration
        self.enable_deduplication = enable_deduplication
        self.similarity_threshold = similarity_threshold
        self.pdf_loader_strategy = pdf_loader_strategy

        # Document index
        self.document_index_path = self.processed_dir / "document_index.json"
        self.document_index = self._load_document_index()

        # Settlement-specific processors
        self._initialize_settlement_processors()

        # Document loaders
        self.loader_mapping = self._initialize_loader_mapping()

        logger.info(
            "DocumentProcessor initialized for settlement content processing"
        )

    def _initialize_settlement_processors(self):
        """Initialize settlement-specific content processors."""
        # Nairobi location patterns
        self.nairobi_locations = {
            "neighborhoods": [
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
                "Loresho",
                "Kitisuru",
                "Rosslyn",
                "Kabete",
                "Langata",
                "Ngong",
                "Kasarani",
                "Ruiru",
            ],
            "universities": [
                "University of Nairobi",
                "Kenyatta University",
                "JKUAT",
                "Strathmore University",
                "USIU",
                "Daystar University",
                "Catholic University",
                "Kenya Methodist University",
                "Multimedia University",
                "Technical University of Kenya",
            ],
            "landmarks": [
                "KICC",
                "Uhuru Park",
                "Nairobi National Park",
                "City Market",
                "Sarit Centre",
                "Westgate Mall",
                "Junction Mall",
                "Village Market",
                "Two Rivers Mall",
                "Garden City Mall",
                "Yaya Centre",
            ],
        }

        # Cost pattern recognition
        self.cost_patterns = [
            r"KSh?\s*[\d,]+(?:\.\d{2})?",  # Kenyan Shilling amounts
            r"USD?\s*[\d,]+(?:\.\d{2})?",  # US Dollar amounts
            r"[\d,]+\s*(?:shilling|bob|KES)",  # Alternative currency formats
            r"rent.*?KSh?\s*[\d,]+",  # Rent costs
            r"deposit.*?KSh?\s*[\d,]+",  # Deposit amounts
            r"fee.*?KSh?\s*[\d,]+",  # Fee amounts
        ]

        # Settlement topics
        self.settlement_topics = {
            "housing": [
                "accommodation",
                "housing",
                "rent",
                "apartment",
                "room",
            ],
            "transportation": [
                "transport",
                "matatu",
                "bus",
                "taxi",
                "uber",
                "boda",
            ],
            "education": [
                "university",
                "college",
                "student",
                "academic",
                "course",
            ],
            "legal": ["visa", "permit", "immigration", "passport", "embassy"],
            "finance": ["bank", "money", "cost", "budget", "mpesa", "payment"],
            "safety": ["safe", "security", "crime", "police", "dangerous"],
            "healthcare": [
                "hospital",
                "clinic",
                "doctor",
                "medical",
                "insurance",
            ],
            "culture": ["culture", "language", "food", "custom", "tradition"],
        }

    def _initialize_loader_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive document loader mapping."""
        return {
            # Text formats
            ".txt": {
                "loader_class": TextLoader,
                "loader_kwargs": {"encoding": "utf-8"},
                "doc_type": "text",
            },
            ".md": {
                "loader_class": UnstructuredMarkdownLoader,
                "loader_kwargs": {},
                "doc_type": "markdown",
            },
            ".csv": {
                "loader_class": CSVLoader,
                "loader_kwargs": {"encoding": "utf-8"},
                "doc_type": "csv",
            },
            ".json": {
                "loader_class": JSONLoader,
                "loader_kwargs": {"jq_schema": ".", "text_content": False},
                "doc_type": "json",
            },
            # PDF formats
            ".pdf": {
                "loader_class": self._get_pdf_loader_class,
                "loader_kwargs": {},
                "doc_type": "pdf",
            },
            # Microsoft Office formats
            ".docx": {
                "loader_class": UnstructuredWordDocumentLoader,
                "loader_kwargs": {},
                "doc_type": "docx",
                "fallback_loader": Docx2txtLoader,
            },
            ".doc": {
                "loader_class": UnstructuredWordDocumentLoader,
                "loader_kwargs": {},
                "doc_type": "doc",
            },
            ".xlsx": {
                "loader_class": UnstructuredExcelLoader,
                "loader_kwargs": {},
                "doc_type": "xlsx",
            },
            ".xls": {
                "loader_class": UnstructuredExcelLoader,
                "loader_kwargs": {},
                "doc_type": "xls",
            },
            ".pptx": {
                "loader_class": UnstructuredPowerPointLoader,
                "loader_kwargs": {},
                "doc_type": "pptx",
            },
            ".ppt": {
                "loader_class": UnstructuredPowerPointLoader,
                "loader_kwargs": {},
                "doc_type": "ppt",
            },
            # Web/HTML formats
            ".html": {
                "loader_class": UnstructuredHTMLLoader,
                "loader_kwargs": {},
                "doc_type": "html",
                "fallback_loader": BSHTMLLoader,
            },
            ".htm": {
                "loader_class": UnstructuredHTMLLoader,
                "loader_kwargs": {},
                "doc_type": "html",
                "fallback_loader": BSHTMLLoader,
            },
            ".xml": {
                "loader_class": UnstructuredXMLLoader,
                "loader_kwargs": {},
                "doc_type": "xml",
            },
            # Email formats
            ".eml": {
                "loader_class": UnstructuredEmailLoader,
                "loader_kwargs": {},
                "doc_type": "email",
            },
        }

    def process_document(
        self, file_path: Union[str, Path]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a document with settlement-specific optimization.

        Args:
            file_path: Path to document

        Returns:
            Processing metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.is_file_supported(file_path):
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        logger.info(f"Processing settlement document: {file_path.name}")

        try:
            # Extract text with fallback handling
            text, doc_type = self._extract_text_with_fallback(file_path)

            if not text or not text.strip():
                logger.warning(f"No text extracted from {file_path.name}")
                return None

            # Generate document ID
            doc_id = self._generate_document_id(file_path)

            # Clean and preprocess text for settlement content
            cleaned_text = self._clean_settlement_text(text)

            # Save processed text
            processed_path = self.processed_dir / f"{doc_id}.txt"
            with open(processed_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            # Create optimized chunks
            chunks = self._create_settlement_chunks(
                cleaned_text, doc_id, str(file_path)
            )

            if not chunks:
                logger.warning(f"No chunks created for {file_path.name}")
                return None

            # Save chunks with settlement metadata
            chunk_path = self.chunk_dir / f"{doc_id}_chunks.jsonl"
            self._save_settlement_chunks(chunks, chunk_path)

            # Store metadata
            metadata = {
                "doc_id": doc_id,
                "file_name": file_path.name,
                "file_path": str(file_path),
                "doc_type": doc_type,
                "processed_path": str(processed_path),
                "chunks_path": str(chunk_path),
                "num_chunks": len(chunks),
                "file_size": file_path.stat().st_size,
                "last_modified": file_path.stat().st_mtime,
                "processed_date": datetime.now().timestamp(),
                "settlement_optimized": True,
                "chunking_strategy": self.semantic_chunker.strategy,
                "avg_settlement_score": sum(
                    chunk.settlement_score for chunk in chunks
                )
                / len(chunks),
            }

            # Update document index
            self.document_index[doc_id] = metadata
            self._save_document_index()

            logger.info(
                f"Successfully processed {file_path.name}: {len(chunks)} chunks"
            )
            return metadata

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            raise

    def _extract_text_with_fallback(self, file_path: Path) -> tuple[str, str]:
        """Extract text with comprehensive fallback strategy."""
        file_extension = file_path.suffix.lower()
        loader_info = self.loader_mapping.get(file_extension)

        if not loader_info:
            # Generic fallback
            try:
                loader = UnstructuredFileLoader(str(file_path))
                documents = loader.load()
                text = "\n\n".join(doc.page_content for doc in documents)
                return text, "generic"
            except Exception:
                raise ValueError(f"Unsupported file type: {file_extension}")

        # Get loader configuration
        loader_class = loader_info["loader_class"]
        loader_kwargs = loader_info.get("loader_kwargs", {})
        doc_type = loader_info["doc_type"]
        fallback_loader = loader_info.get("fallback_loader")

        # Handle PDF strategy
        if file_extension == ".pdf" and callable(loader_class):
            loader_class = loader_class()

        try:
            # Try primary loader
            loader = loader_class(str(file_path), **loader_kwargs)
            documents = loader.load()
            text = "\n\n".join(doc.page_content for doc in documents)

            if text.strip():
                return text, doc_type
            else:
                raise Exception("No text extracted")

        except Exception as e:
            logger.warning(
                f"Primary loader failed for {file_path.name}: {str(e)}"
            )

            # Try fallback loader
            if fallback_loader:
                try:
                    loader = fallback_loader(str(file_path))
                    documents = loader.load()
                    text = "\n\n".join(doc.page_content for doc in documents)
                    if text.strip():
                        return text, doc_type
                except Exception as fallback_error:
                    logger.warning(
                        f"Fallback loader failed: {str(fallback_error)}"
                    )

            # Final fallback for PDFs
            if file_extension == ".pdf":
                pdf_loaders = [PyPDFLoader, PDFPlumberLoader, PyMuPDFLoader]
                for pdf_loader_class in pdf_loaders:
                    try:
                        loader = pdf_loader_class(str(file_path))
                        documents = loader.load()
                        text = "\n\n".join(
                            doc.page_content for doc in documents
                        )
                        if text.strip():
                            return text, doc_type
                    except Exception:
                        continue

            raise Exception(
                f"All extraction methods failed for {file_path.name}"
            )

    def _clean_settlement_text(self, text: str) -> str:
        """Clean text with settlement-specific preprocessing."""
        if not text:
            return ""

        # Basic cleaning
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\t+", " ", text)

        # Settlement-specific cleaning
        # Normalize location names
        for category, locations in self.nairobi_locations.items():
            for location in locations:
                # Make location references consistent
                pattern = re.compile(re.escape(location), re.IGNORECASE)
                text = pattern.sub(location, text)

        # Normalize currency mentions
        text = re.sub(r"Kenya\s+Shilling", "KSh", text, flags=re.IGNORECASE)
        text = re.sub(r"Kenyan\s+Shilling", "KSh", text, flags=re.IGNORECASE)
        text = re.sub(r"\bKES\b", "KSh", text)

        # Clean up contact information formatting
        text = re.sub(r"\+254\s*(\d)", r"+254-\1", text)

        return text.strip()

    def _create_settlement_chunks(
        self, text: str, doc_id: str, source_file: str
    ) -> List[ProcessedChunk]:
        """Create chunks optimized for settlement content."""
        # Use semantic chunker
        raw_chunks = self.semantic_chunker.create_chunks(text, doc_id)

        processed_chunks = []
        for i, chunk in enumerate(raw_chunks):
            # Extract settlement-specific entities
            location_entities = self._extract_location_entities(chunk.text)
            cost_entities = self._extract_cost_entities(chunk.text)
            topic_tags = self._extract_topic_tags(chunk.text)
            settlement_score = self._calculate_settlement_relevance_score(
                chunk.text
            )

            processed_chunk = ProcessedChunk(
                chunk_id=f"{doc_id}_{i:04d}",
                doc_id=doc_id,
                chunk_index=i,
                text=chunk.text,
                metadata={
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "char_count": len(chunk.text),
                    "word_count": len(chunk.text.split()),
                    "semantic_score": chunk.semantic_score,
                    "topic_coherence": chunk.topic_coherence,
                    "chunk_type": chunk.chunk_type,
                    "source_file": source_file,
                },
                settlement_score=settlement_score,
                topic_tags=topic_tags,
                location_entities=location_entities,
                cost_entities=cost_entities,
            )
            processed_chunks.append(processed_chunk)

        return processed_chunks

    def _extract_location_entities(self, text: str) -> List[str]:
        """Extract Nairobi-specific location entities."""
        locations = []
        text_lower = text.lower()

        # Check for known locations
        for category, location_list in self.nairobi_locations.items():
            for location in location_list:
                if location.lower() in text_lower:
                    locations.append(location)

        # Use spaCy for additional location detection
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["GPE", "LOC"] and len(ent.text) > 2:
                    locations.append(ent.text)
        except Exception:
            pass

        return list(set(locations))

    def _extract_cost_entities(self, text: str) -> List[str]:
        """Extract cost and price information."""
        costs = []

        for pattern in self.cost_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            costs.extend(matches)

        return list(set(costs))

    def _extract_topic_tags(self, text: str) -> List[str]:
        """Extract settlement topic tags."""
        tags = []
        text_lower = text.lower()

        for topic, keywords in self.settlement_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(topic)

        return tags

    def _calculate_settlement_relevance_score(self, text: str) -> float:
        """Calculate settlement relevance score."""
        text_lower = text.lower()
        score = 0.0

        # High-value settlement keywords
        high_value_keywords = {
            "international student": 3.0,
            "nairobi": 2.5,
            "kenya": 2.0,
            "accommodation": 2.5,
            "housing": 2.0,
            "university": 2.0,
            "visa": 2.5,
            "immigration": 2.5,
            "cost of living": 2.5,
        }

        # Medium-value keywords
        medium_value_keywords = {
            "transport": 1.5,
            "safety": 2.0,
            "bank": 1.5,
            "hospital": 1.5,
            "culture": 1.0,
            "language": 1.0,
        }

        # Calculate weighted score
        total_weight = 0.0
        for keyword, weight in {
            **high_value_keywords,
            **medium_value_keywords,
        }.items():
            if keyword in text_lower:
                count = text_lower.count(keyword)
                score += count * weight
                total_weight += weight

        # Normalize
        if total_weight > 0:
            return min(score / total_weight, 1.0)

        return 0.1  # Default low relevance

    def _save_settlement_chunks(
        self, chunks: List[ProcessedChunk], output_path: Path
    ):
        """Save chunks with settlement metadata."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(
                        json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n"
                    )

            logger.debug(
                f"Saved {len(chunks)} settlement-optimized chunks to {output_path}"
            )

        except Exception as e:
            raise Exception(f"Failed to save chunks: {str(e)}")

    def _get_pdf_loader_class(self):
        """Get PDF loader based on strategy."""
        strategy_mapping = {
            "pypdf": PyPDFLoader,
            "pdfplumber": PDFPlumberLoader,
            "pymupdf": PyMuPDFLoader,
            "auto": PyPDFLoader,
        }
        return strategy_mapping.get(self.pdf_loader_strategy, PyPDFLoader)

    def _load_document_index(self) -> Dict[str, Dict[str, Any]]:
        """Load document index."""
        try:
            if self.document_index_path.exists():
                with open(self.document_index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading document index: {str(e)}")
            return {}

    def _save_document_index(self) -> bool:
        """Save document index."""
        try:
            with open(self.document_index_path, "w", encoding="utf-8") as f:
                json.dump(self.document_index, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving document index: {str(e)}")
            return False

    def _generate_document_id(self, file_path: Path) -> str:
        """Generate unique document ID."""
        try:
            file_stat = file_path.stat()
            unique_string = (
                f"{file_path}_{file_stat.st_mtime}_{file_stat.st_size}"
            )
            return hashlib.md5(unique_string.encode()).hexdigest()
        except Exception:
            fallback_string = f"{str(file_path)}_{datetime.now().timestamp()}"
            return hashlib.md5(fallback_string.encode()).hexdigest()

    def is_file_supported(self, file_path: Path) -> bool:
        """Check if file type is supported."""
        extension = file_path.suffix.lower()
        return extension in self.loader_mapping

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return list(self.loader_mapping.keys())

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all processed documents."""
        return list(self.document_index.values())

    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document information."""
        return self.document_index.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Delete document and associated files."""
        try:
            if doc_id not in self.document_index:
                return False

            metadata = self.document_index[doc_id]

            # Delete files
            for file_path in [
                metadata.get("processed_path"),
                metadata.get("chunks_path"),
            ]:
                if file_path and Path(file_path).exists():
                    Path(file_path).unlink()

            # Remove from index
            del self.document_index[doc_id]
            self._save_document_index()

            return True

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {str(e)}")
            return False

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_documents": len(self.document_index),
            "supported_formats": len(self.get_supported_extensions()),
            "deduplication_enabled": self.enable_deduplication,
            "chunking_strategy": self.semantic_chunker.strategy,
            "settlement_optimized": True,
            "avg_settlement_score": self._calculate_avg_settlement_score(),
        }

    def _calculate_avg_settlement_score(self) -> float:
        """Calculate average settlement score across all documents."""
        scores = [
            doc.get("avg_settlement_score", 0)
            for doc in self.document_index.values()
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def process_url(
        self, url: str, output_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Process content from a URL for settlement information."""
        try:
            logger.info(f"Processing URL: {url}")

            loader = WebBaseLoader(url)
            documents = loader.load()

            if not documents:
                logger.warning(f"No content extracted from URL: {url}")
                return None

            text = "\n\n".join(doc.page_content for doc in documents)

            if not text.strip():
                logger.warning(f"Empty content from URL: {url}")
                return None

            # Generate document ID based on URL
            doc_id = hashlib.md5(url.encode()).hexdigest()

            # Use provided name or generate from URL
            if not output_name:
                from urllib.parse import urlparse

                parsed = urlparse(url)
                output_name = f"{parsed.netloc}_{parsed.path.replace('/', '_')}"

            # Clean text with settlement optimization
            cleaned_text = self._clean_settlement_text(text)

            # Save processed text
            processed_path = self.processed_dir / f"{doc_id}.txt"
            with open(processed_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            # Create settlement-optimized chunks
            chunks = self._create_settlement_chunks(cleaned_text, doc_id, url)

            # Save chunks with settlement metadata
            chunk_path = self.chunk_dir / f"{doc_id}_chunks.jsonl"
            self._save_settlement_chunks(chunks, chunk_path)

            # Calculate settlement relevance
            avg_settlement_score = (
                sum(chunk.settlement_score for chunk in chunks) / len(chunks)
                if chunks
                else 0
            )

            metadata = {
                "doc_id": doc_id,
                "file_name": output_name,
                "file_path": url,
                "doc_type": "web",
                "processed_path": str(processed_path),
                "chunks_path": str(chunk_path),
                "num_chunks": len(chunks),
                "last_modified": None,
                "processed_date": datetime.now().timestamp(),
                "settlement_optimized": True,
                "chunking_strategy": self.semantic_chunker.strategy,
                "avg_settlement_score": avg_settlement_score,
                "source_type": "web_url",
            }

            # Update document index
            self.document_index[doc_id] = metadata
            self._save_document_index()

            logger.info(
                f"Processed URL {url}: {len(chunks)} chunks, settlement score: {avg_settlement_score:.3f}"
            )
            return metadata

        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            raise Exception(f"Failed to process URL {url}: {str(e)}")

    def process_sitemap(
        self, sitemap_url: str, max_pages: int = 50
    ) -> List[Dict[str, Any]]:
        """Process multiple pages from a sitemap for settlement content."""
        try:
            logger.info(f"Processing sitemap: {sitemap_url}")

            loader = SitemapLoader(sitemap_url)
            documents = loader.load()

            results = []
            settlement_pages = 0

            for i, doc in enumerate(documents[:max_pages]):
                try:
                    # Generate unique doc_id for each page
                    source_url = doc.metadata.get(
                        "source", f"{sitemap_url}_page_{i}"
                    )
                    doc_id = hashlib.md5(source_url.encode()).hexdigest()

                    text = doc.page_content
                    if not text.strip():
                        continue

                    # Check if page is relevant to settlement before processing
                    if not self._is_settlement_relevant(text):
                        logger.debug(
                            f"Skipping non-settlement page: {source_url}"
                        )
                        continue

                    # Clean text with settlement optimization
                    cleaned_text = self._clean_settlement_text(text)

                    # Save processed text
                    processed_path = self.processed_dir / f"{doc_id}.txt"
                    with open(processed_path, "w", encoding="utf-8") as f:
                        f.write(cleaned_text)

                    # Create settlement-optimized chunks
                    chunks = self._create_settlement_chunks(
                        cleaned_text, doc_id, source_url
                    )

                    # Save chunks
                    chunk_path = self.chunk_dir / f"{doc_id}_chunks.jsonl"
                    self._save_settlement_chunks(chunks, chunk_path)

                    # Generate page name
                    from urllib.parse import urlparse

                    parsed = urlparse(source_url)
                    page_name = (
                        f"{parsed.netloc}_{parsed.path.replace('/', '_')}"
                    )

                    # Calculate settlement relevance
                    avg_settlement_score = (
                        sum(chunk.settlement_score for chunk in chunks)
                        / len(chunks)
                        if chunks
                        else 0
                    )

                    metadata = {
                        "doc_id": doc_id,
                        "file_name": page_name,
                        "file_path": source_url,
                        "doc_type": "web_sitemap",
                        "processed_path": str(processed_path),
                        "chunks_path": str(chunk_path),
                        "num_chunks": len(chunks),
                        "last_modified": None,
                        "processed_date": datetime.now().timestamp(),
                        "settlement_optimized": True,
                        "chunking_strategy": self.semantic_chunker.strategy,
                        "avg_settlement_score": avg_settlement_score,
                        "source_type": "web_sitemap",
                    }

                    # Update document index
                    self.document_index[doc_id] = metadata
                    results.append(metadata)
                    settlement_pages += 1

                    logger.debug(
                        f"Processed sitemap page: {page_name} (settlement score: {avg_settlement_score:.3f})"
                    )

                except Exception as e:
                    logger.error(f"Error processing sitemap page {i}: {str(e)}")
                    continue

            self._save_document_index()
            logger.info(
                f"Processed {settlement_pages} settlement-relevant pages from sitemap"
            )
            return results

        except Exception as e:
            logger.error(f"Error processing sitemap {sitemap_url}: {str(e)}")
            raise Exception(f"Failed to process sitemap: {str(e)}")

    def _is_settlement_relevant(self, text: str) -> bool:
        """Check if webpage content is relevant to international student settlement."""
        text_lower = text.lower()

        # High-relevance keywords for settlement content
        settlement_keywords = [
            "international student",
            "student accommodation",
            "housing",
            "nairobi",
            "kenya",
            "visa",
            "immigration",
            "university",
            "college",
            "transport",
            "safety",
            "cost of living",
            "bank",
            "hospital",
            "culture",
            "language",
            "settlement",
            "relocation",
            "expat",
            "foreign student",
        ]

        # Location-specific keywords for Nairobi
        nairobi_keywords = [
            "westlands",
            "kilimani",
            "karen",
            "lavington",
            "kileleshwa",
            "parklands",
            "cbd",
            "uhuru park",
            "city market",
            "matatu",
            "boda boda",
            "mpesa",
        ]

        # Education-specific keywords
        education_keywords = [
            "university of nairobi",
            "kenyatta university",
            "strathmore",
            "usiu",
            "jkuat",
            "daystar",
            "catholic university",
            "admission",
            "enrollment",
            "academic",
            "semester",
            "course",
            "degree",
            "diploma",
        ]

        all_keywords = (
            settlement_keywords + nairobi_keywords + education_keywords
        )

        # Count keyword matches
        matches = sum(1 for keyword in all_keywords if keyword in text_lower)

        # Also check for substantial content (not just navigation/boilerplate)
        word_count = len(text.split())

        # Consider relevant if:
        # 1. Has multiple settlement keywords, OR
        # 2. Has at least one high-relevance keyword and substantial content
        if matches >= 3:
            return True
        elif matches >= 1 and word_count > 200:
            return True
        elif any(
            keyword in text_lower
            for keyword in [
                "international student",
                "student accommodation",
                "nairobi university",
            ]
        ):
            return True

        return False
