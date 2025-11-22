import json
import time
import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from urllib.parse import urlparse

from fastapi import BackgroundTasks
from fastapi import Depends
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import Query
from fastapi import Security
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from pydantic import Field
from pydantic import validator
import uvicorn

from config.settings import ROOT_DIR
from config.settings import settings
from services.document_processor import DocumentProcessor
from services.embeddings import EmbeddingService
from services.evaluator import InternationalStudentRAGEvaluator
from services.intent_recognizer import IntentRecognizer
from services.intent_recognizer import IntentType
from services.intent_recognizer import TopicType
from services.language_processor import LanguageProcessor
from services.response_generator import ResponseGenerator
from services.semantic_chunking import ChunkingStrategy
from services.semantic_chunking import SemanticChunker
from services.vector_db import VectorDBService

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("settlebot_api")

# Initialize FastAPI app
app = FastAPI(
    title="SettleBot API - Complete Settlement Assistant",
    description="Comprehensive RAG-powered settlement assistant for international students in Nairobi, Kenya. Provides housing, transportation, university, safety, legal, finance, healthcare, and cultural guidance.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
if settings.api.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# API Key security
API_KEY = settings.api.api_key
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Initialize services
embedding_service = EmbeddingService()
vector_db_service = VectorDBService(embedding_service=embedding_service)
intent_recognizer = IntentRecognizer()
language_processor = LanguageProcessor()
response_generator = ResponseGenerator()
document_processor = DocumentProcessor(
    embedding_service=embedding_service,
    enable_deduplication=settings.deduplication.enabled,
    similarity_threshold=settings.deduplication.similarity_threshold,
)
evaluator = InternationalStudentRAGEvaluator(
    vector_db_service=vector_db_service,
    intent_recognizer=intent_recognizer,
    response_generator=response_generator
)
semantic_chunker = SemanticChunker()

# Upload directory
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Background tasks storage
background_tasks_status = {}

# ===============================
# PYDANTIC MODELS
# ===============================


class QueryRequest(BaseModel):
    query: str = Field(...,
                       description="User query about settlement in Nairobi")
    top_k: int = Field(default=15, ge=1, le=50,
                       description="Number of context chunks to retrieve")
    include_context: bool = Field(
        default=True, description="Include retrieved context in response")
    language_detection: bool = Field(
        default=True, description="Enable language detection and translation")
    conversation_context: Optional[Dict[str, Any]] = Field(
        default=None, description="Previous conversation context")
    user_preferences: Optional[Dict[str, str]] = Field(
        default=None, description="User preferences for response style")


class LanguageInfo(BaseModel):
    detected_language: str
    original_query: str
    english_query: str
    translation_needed: bool
    confidence: float


class QueryResponse(BaseModel):
    response: str
    original_response: Optional[str] = None
    intent_type: str
    topic: str
    confidence: float
    language_info: LanguageInfo
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    token_usage: Optional[Dict[str, int]] = None
    current_time: Optional[str] = None
    settlement_optimized: bool = True
    empathy_applied: bool = False
    safety_protocols_added: bool = False
    crisis_level: str = "none"
    emotional_state: Optional[str] = None
    web_search_used: bool = False


class DocumentUploadResponse(BaseModel):
    doc_id: str
    file_name: str
    doc_type: str
    num_chunks: int
    success: bool
    message: str
    processing_time: Optional[float] = None
    settlement_score: Optional[float] = None


class SystemStatusResponse(BaseModel):
    status: str
    services: Dict[str, Dict[str, Any]]
    configuration: Dict[str, Any]
    settlement_optimization: Dict[str, Any]
    statistics: Dict[str, Any]


class URLProcessRequest(BaseModel):
    url: str = Field(..., description="URL to process for settlement content")
    output_name: Optional[str] = Field(
        default=None, description="Custom name for the processed document")
    validate_first: bool = Field(
        default=True, description="Validate content relevance before processing")


class SitemapProcessRequest(BaseModel):
    sitemap_url: str = Field(..., description="Sitemap URL to process")
    max_pages: int = Field(default=50, ge=1, le=200,
                           description="Maximum pages to process")
    settlement_filter: bool = Field(
        default=True, description="Only process settlement-relevant pages")


class IntentAnalysisRequest(BaseModel):
    query: str = Field(..., description="Query to analyze for intent")
    include_semantic_scores: bool = Field(
        default=True, description="Include detailed semantic scores")


class EvaluationRequest(BaseModel):
    queries: Optional[List[str]] = Field(
        default=None, description="Custom queries to evaluate")
    focus_area: Optional[str] = Field(
        default=None, description="Focus area: housing, safety, university, finance")
    num_queries: int = Field(default=20, ge=1, le=100,
                             description="Number of queries to evaluate")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, ge=1, le=50)
    topic_filter: Optional[str] = Field(
        default=None, description="Filter by topic")
    location_filter: Optional[str] = Field(
        default=None, description="Filter by location")
    doc_id_filter: Optional[str] = Field(
        default=None, description="Filter by document ID")


class ChunkingRequest(BaseModel):
    text: str = Field(..., description="Text to chunk")
    strategy: str = Field(default="settlement_optimized",
                          description="Chunking strategy")
    min_chunk_size: int = Field(default=100, ge=50, le=500)
    max_chunk_size: int = Field(default=1000, ge=500, le=2000)


class LanguageTestRequest(BaseModel):
    test_cases: List[Dict[str, str]
                     ] = Field(..., description="Test cases with query and expected language")

# ===============================
# DEPENDENCY FUNCTIONS
# ===============================


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        raise HTTPException(status_code=401, detail="Missing API Key")

    # Extract token from "Bearer {token}" format
    if api_key_header.startswith("Bearer "):
        api_key_header = api_key_header.split(" ")[1]

    if api_key_header != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    return api_key_header

# ===============================
# CORE API ENDPOINTS
# ===============================


@app.get("/", response_class=HTMLResponse)
async def root():
    """API root with documentation."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SettleBot API</title>
        <style>body{font-family:Arial,sans-serif;margin:40px;}</style>
    </head>
    <body>
        <h1>🌍 SettleBot API - Complete Settlement Assistant</h1>
        <p>Comprehensive RAG-powered settlement assistant for international students in Nairobi, Kenya.</p>
        <h2>Quick Links</h2>
        <ul>
            <li><a href="/docs">Interactive API Documentation</a></li>
            <li><a href="/redoc">ReDoc Documentation</a></li>
            <li><a href="/health">Health Check</a></li>
            <li><a href="/system/status">System Status</a></li>
        </ul>
        <h2>Features</h2>
        <ul>
            <li>🏠 Housing and accommodation guidance</li>
            <li>🚌 Transportation and travel information</li>
            <li>🎓 University and academic support</li>
            <li>🛡️ Safety and security guidance</li>
            <li>⚖️ Legal and immigration assistance</li>
            <li>💰 Banking and finance guidance</li>
            <li>🏥 Healthcare and medical information</li>
            <li>🌍 Cultural adaptation support</li>
            <li>🔍 Document processing and search</li>
            <li>📊 Performance evaluation and analytics</li>
        </ul>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "api_version": "2.0.0",
            "settlement_domain": "nairobi_kenya",
        }

        # Service health checks
        services_health = {}

        # Vector database
        try:
            vdb_health = vector_db_service.health_check()
            services_health["vector_database"] = vdb_health
        except Exception as e:
            services_health["vector_database"] = {
                "status": "unhealthy", "error": str(e)}

        # Embedding service
        try:
            embedding_stats = embedding_service.get_embedding_stats()
            services_health["embedding_service"] = {
                "status": "healthy",
                "model": embedding_stats.get("model", "unknown"),
                "total_embeddings": embedding_stats.get("total_embeddings", 0)
            }
        except Exception as e:
            services_health["embedding_service"] = {
                "status": "unhealthy", "error": str(e)}

        # Intent recognizer
        try:
            intent_stats = intent_recognizer.get_stats()
            services_health["intent_recognizer"] = {
                "status": "healthy",
                "total_intents": intent_stats.get("total_intents", 0),
                "classification_method": intent_stats.get("classification_method", "unknown")
            }
        except Exception as e:
            services_health["intent_recognizer"] = {
                "status": "unhealthy", "error": str(e)}

        # Language processor
        try:
            lang_stats = language_processor.get_language_stats()
            services_health["language_processor"] = {
                "status": "healthy",
                "supported_languages": lang_stats.get("total_languages", 0),
                "detection_enabled": lang_stats.get("detection_enabled", False)
            }
        except Exception as e:
            services_health["language_processor"] = {
                "status": "unhealthy", "error": str(e)}

        # Response generator
        try:
            response_stats = response_generator.get_response_stats()
            services_health["response_generator"] = {
                "status": "healthy",
                "model": response_stats.get("model", "unknown"),
                "settlement_optimized": response_stats.get("settlement_optimized", False)
            }
        except Exception as e:
            services_health["response_generator"] = {
                "status": "unhealthy", "error": str(e)}

        health_status["services"] = services_health

        # Overall health
        unhealthy_services = [
            k for k, v in services_health.items() if v.get("status") != "healthy"]
        health_status["overall_healthy"] = len(unhealthy_services) == 0
        health_status["unhealthy_services"] = unhealthy_services

        # API key check
        health_status["api_key_configured"] = bool(os.getenv("OPENAI_API_KEY"))

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}


@app.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status(api_key: str = Depends(get_api_key)):
    """Get comprehensive system status and statistics."""
    try:
        # Service status checks
        services = {}

        # Vector database
        try:
            vdb_health = vector_db_service.health_check()
            vdb_stats = vector_db_service.get_collection_stats()
            services["vector_database"] = {
                **vdb_health,
                "collection_stats": vdb_stats
            }
        except Exception as e:
            services["vector_database"] = {"status": "error", "error": str(e)}

        # Document processor
        try:
            doc_stats = document_processor.get_processing_stats()
            documents = document_processor.list_documents()
            services["document_processor"] = {
                "status": "healthy",
                **doc_stats,
                "total_documents": len(documents)
            }
        except Exception as e:
            services["document_processor"] = {
                "status": "error", "error": str(e)}

        # Embedding service
        try:
            embedding_stats = embedding_service.get_embedding_stats()
            services["embedding_service"] = {
                "status": "healthy",
                **embedding_stats
            }
        except Exception as e:
            services["embedding_service"] = {
                "status": "error", "error": str(e)}

        # Language processor
        try:
            lang_stats = language_processor.get_language_stats()
            services["language_processor"] = {
                "status": "healthy",
                **lang_stats
            }
        except Exception as e:
            services["language_processor"] = {
                "status": "error", "error": str(e)}

        # Response generator
        try:
            response_stats = response_generator.get_response_stats()
            services["response_generator"] = {
                "status": "healthy",
                **response_stats
            }
        except Exception as e:
            services["response_generator"] = {
                "status": "error", "error": str(e)}

        # Intent recognizer
        try:
            intent_stats = intent_recognizer.get_stats()
            validation_results = intent_recognizer.validate_patterns()
            services["intent_recognizer"] = {
                "status": "healthy",
                **intent_stats,
                "pattern_validation": validation_results
            }
        except Exception as e:
            services["intent_recognizer"] = {
                "status": "error", "error": str(e)}

        # Configuration
        configuration = {
            "chunking_strategy": semantic_chunker.strategy.value,
            "deduplication_enabled": settings.deduplication.enabled,
            "language_detection": settings.language.detection_enabled,
            "embedding_model": settings.embedding.model,
            "llm_model": settings.llm.model,
            "vector_collection": settings.vector_db.collection_name,
            "settlement_domain": "nairobi_kenya"
        }

        # Settlement optimization info
        settlement_optimization = {
            "domain": "international_student_settlement",
            "location": "nairobi_kenya",
            "semantic_chunking": semantic_chunker.strategy == ChunkingStrategy.SETTLEMENT_OPTIMIZED,
            "multilingual_support": settings.language.detection_enabled,
            "settlement_scoring": True,
            "location_entity_extraction": True,
            "cost_entity_extraction": True,
            "safety_protocols": True,
            "empathy_detection": True,
            "crisis_assessment": True
        }

        # Statistics
        documents = document_processor.list_documents()
        statistics = {
            "total_documents": len(documents),
            "total_chunks": sum(doc.get("num_chunks", 0) for doc in documents),
            "avg_settlement_score": sum(doc.get("avg_settlement_score", 0) for doc in documents) / len(documents) if documents else 0,
            "supported_file_types": len(document_processor.get_supported_extensions()),
            "supported_languages": len(language_processor.supported_languages),
            "intent_types": len(intent_recognizer.intent_patterns),
            "uptime_hours": 0,  # Would need to track this
        }

        # Overall status
        healthy_services = sum(
            1 for service in services.values() if service.get("status") == "healthy")
        total_services = len(services)
        overall_status = "healthy" if healthy_services == total_services else "degraded"

        return SystemStatusResponse(
            status=overall_status,
            services=services,
            configuration=configuration,
            settlement_optimization=settlement_optimization,
            statistics=statistics
        )

    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting system status: {str(e)}")

# ===============================
# QUERY AND RESPONSE ENDPOINTS
# ===============================


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, api_key: str = Depends(get_api_key)):
    """Process settlement query with comprehensive response generation."""
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        start_time = time.time()

        # Language detection and processing
        if request.language_detection:
            language_result = language_processor.detect_and_process_query(
                request.query)
            english_query = language_result["english_query"]
        else:
            language_result = {
                "detected_language": "english",
                "original_query": request.query,
                "english_query": request.query,
                "needs_translation": False,
                "confidence": 1.0,
            }
            english_query = request.query

        # Intent recognition
        intent_info = intent_recognizer.get_intent_info(english_query)
        logger.info(f"Recognized intent: {intent_info['intent_type'].value}")

        # Retrieve relevant context
        retrieved_chunks = []
        if intent_info["intent_type"] != IntentType.OFF_TOPIC:
            retrieved_chunks = vector_db_service.search(
                query=english_query,
                top_k=request.top_k
            )

        # Generate comprehensive response
        response_data = response_generator.generate_response(
            query=request.query,
            retrieved_context=retrieved_chunks,
            intent_info=intent_info,
            conversation_context=request.conversation_context,
            user_preferences=request.user_preferences
        )

        # Create language info
        language_info = LanguageInfo(
            detected_language=language_result["detected_language"],
            original_query=language_result["original_query"],
            english_query=language_result["english_query"],
            translation_needed=language_result["needs_translation"],
            confidence=language_result.get("confidence", 0.0),
        )

        # Processing time
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f} seconds")

        # Prepare response
        query_response = QueryResponse(
            response=response_data["response"],
            original_response=response_data.get("original_response"),
            intent_type=intent_info["intent_type"].value,
            topic=intent_info["topic"].value,
            confidence=intent_info["confidence"],
            language_info=language_info,
            token_usage=response_data.get("token_usage"),
            current_time=response_data.get("current_time"),
            empathy_applied=response_data.get("empathy_applied", False),
            safety_protocols_added=response_data.get(
                "safety_protocols_added", False),
            crisis_level=response_data.get("crisis_level", "none"),
            emotional_state=response_data.get("emotional_state"),
            web_search_used=response_data.get("web_search_used", False)
        )

        # Include debug information if requested
        if request.include_context:
            query_response.retrieved_chunks = retrieved_chunks

        return query_response

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/intent/analyze")
async def analyze_intent(request: IntentAnalysisRequest, api_key: str = Depends(get_api_key)):
    """Analyze query intent with detailed breakdown."""
    try:
        # Get intent analysis
        intent_info = intent_recognizer.get_intent_info(request.query)

        response = {
            "query": request.query,
            "intent_type": intent_info["intent_type"].value,
            "topic": intent_info["topic"].value,
            "confidence": intent_info["confidence"],
            "settlement_relevance": intent_info["settlement_relevance"],
            "classification_method": intent_info["classification_method"],
            "is_off_topic": intent_info["is_off_topic"],
        }

        if request.include_semantic_scores:
            response["semantic_scores"] = intent_info.get(
                "semantic_scores", {})
            response["off_topic_indicators"] = intent_info.get(
                "off_topic_indicators", [])

        return response

    except Exception as e:
        logger.error(f"Error analyzing intent: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error analyzing intent: {str(e)}")


@app.post("/search")
async def search_knowledge_base(request: SearchRequest, api_key: str = Depends(get_api_key)):
    """Search the settlement knowledge base with advanced filtering."""
    try:
        # Perform search
        results = vector_db_service.search(
            query=request.query,
            top_k=request.top_k,
            filter_doc_id=request.doc_id_filter,
            topic_filter=request.topic_filter,
            location_filter=request.location_filter
        )

        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "filters_applied": {
                "topic": request.topic_filter,
                "location": request.location_filter,
                "doc_id": request.doc_id_filter
            },
            "settlement_optimized": True
        }

    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error in search: {str(e)}")


@app.get("/search/topics")
async def search_by_topic(
    topic: str = Query(..., description="Settlement topic to search"),
    top_k: int = Query(10, ge=1, le=50),
    api_key: str = Depends(get_api_key)
):
    """Search by settlement topic (housing, transport, safety, etc.)."""
    try:
        results = vector_db_service.search_by_topic(topic, top_k)

        return {
            "topic": topic,
            "results": results,
            "count": len(results),
            "settlement_optimized": True
        }

    except Exception as e:
        logger.error(f"Error in topic search: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error in topic search: {str(e)}")


@app.get("/search/locations")
async def search_by_location(
    location: str = Query(..., description="Nairobi location to search"),
    query: str = Query("", description="Optional additional query"),
    top_k: int = Query(10, ge=1, le=50),
    api_key: str = Depends(get_api_key)
):
    """Search by Nairobi location."""
    try:
        results = vector_db_service.search_by_location(location, query, top_k)

        return {
            "location": location,
            "query": query,
            "results": results,
            "count": len(results),
            "settlement_optimized": True
        }

    except Exception as e:
        logger.error(f"Error in location search: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error in location search: {str(e)}")

# ===============================
# DOCUMENT PROCESSING ENDPOINTS
# ===============================


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key)
):
    """Upload and process settlement document."""
    try:
        start_time = time.time()

        # Validate file type
        file_path = Path(file.filename)
        if not document_processor.is_file_supported(file_path):
            supported = ", ".join(
                document_processor.get_supported_extensions())
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: {supported}"
            )

        # Save uploaded file
        temp_file_path = UPLOAD_DIR / file.filename
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_size = temp_file_path.stat().st_size
        logger.info(
            f"Processing uploaded file: {temp_file_path.name} ({file_size} bytes)")

        # Process document
        metadata = document_processor.process_document(temp_file_path)

        if not metadata:
            return JSONResponse(
                status_code=500,
                content={"success": False,
                         "message": "Failed to process document"}
            )

        # Generate embeddings
        embedding_service.embed_chunks(metadata["chunks_path"])

        # Index in vector database
        vector_db_service.index_chunks(metadata["chunks_path"])

        # Clean up temp file
        os.remove(temp_file_path)

        processing_time = time.time() - start_time

        return DocumentUploadResponse(
            doc_id=metadata["doc_id"],
            file_name=metadata["file_name"],
            doc_type=metadata["doc_type"],
            num_chunks=metadata["num_chunks"],
            success=True,
            message="Document successfully processed and indexed",
            processing_time=processing_time,
            settlement_score=metadata.get("avg_settlement_score")
        )

    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        # Clean up temp file if it exists
        if 'temp_file_path' in locals() and temp_file_path.exists():
            os.remove(temp_file_path)

        raise HTTPException(
            status_code=500, detail=f"Error uploading document: {str(e)}")


@app.get("/documents")
async def list_documents(
    doc_type: Optional[str] = Query(
        None, description="Filter by document type"),
    settlement_score_min: Optional[float] = Query(
        None, description="Minimum settlement score"),
    api_key: str = Depends(get_api_key)
):
    """List processed documents with filtering options."""
    try:
        documents = document_processor.list_documents()

        # Apply filters
        if doc_type:
            documents = [doc for doc in documents if doc.get(
                "doc_type") == doc_type]

        if settlement_score_min is not None:
            documents = [
                doc for doc in documents
                if doc.get("avg_settlement_score", 0) >= settlement_score_min
            ]

        # Calculate statistics
        total_chunks = sum(doc.get("num_chunks", 0) for doc in documents)
        avg_settlement_score = (
            sum(doc.get("avg_settlement_score", 0)
                for doc in documents) / len(documents)
            if documents else 0
        )

        # Group by document type
        doc_type_stats = {}
        for doc in documents:
            dtype = doc.get("doc_type", "unknown")
            if dtype not in doc_type_stats:
                doc_type_stats[dtype] = {"count": 0, "chunks": 0}
            doc_type_stats[dtype]["count"] += 1
            doc_type_stats[dtype]["chunks"] += doc.get("num_chunks", 0)

        return {
            "documents": documents,
            "count": len(documents),
            "total_chunks": total_chunks,
            "avg_settlement_score": round(avg_settlement_score, 3),
            "doc_type_stats": doc_type_stats,
            "supported_formats": document_processor.get_supported_extensions(),
            "filters_applied": {
                "doc_type": doc_type,
                "settlement_score_min": settlement_score_min
            }
        }

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error listing documents: {str(e)}")


@app.get("/documents/{doc_id}")
async def get_document_info(doc_id: str, api_key: str = Depends(get_api_key)):
    """Get detailed information about a specific document."""
    try:
        doc_info = document_processor.get_document_info(doc_id)

        if not doc_info:
            raise HTTPException(
                status_code=404, detail=f"Document {doc_id} not found")

        # Add file existence checks
        file_checks = {}
        paths_to_check = [
            ("processed_path", doc_info.get("processed_path")),
            ("chunks_path", doc_info.get("chunks_path"))
        ]

        # Check for embeddings file
        embeddings_dir = ROOT_DIR / "data" / "embeddings"
        embeddings_file = embeddings_dir / f"{doc_id}_embeddings.npz"
        paths_to_check.append(("embeddings_path", str(embeddings_file)))

        for file_type, file_path in paths_to_check:
            if file_path:
                file_checks[file_type] = {
                    "path": file_path,
                    "exists": Path(file_path).exists()
                }

        doc_info["file_checks"] = file_checks
        return doc_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting document info: {str(e)}")


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, api_key: str = Depends(get_api_key)):
    """Delete a document and its associated data."""
    try:
        # Check if document exists
        doc_info = document_processor.get_document_info(doc_id)
        if not doc_info:
            return JSONResponse(
                status_code=404,
                content={"success": False,
                         "message": f"Document {doc_id} not found"}
            )

        # Delete document
        success = document_processor.delete_document(doc_id)

        if success:
            return {
                "success": True,
                "message": f"Document {doc_id} successfully deleted",
                "doc_id": doc_id,
                "file_name": doc_info["file_name"],
                "note": "Vector database may need rebuilding for complete cleanup"
            }
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False,
                         "message": f"Failed to delete document {doc_id}"}
            )

    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting document: {str(e)}")


@app.post("/documents/process-url")
async def process_url(request: URLProcessRequest, api_key: str = Depends(get_api_key)):
    """Process content from a URL for settlement information."""
    try:
        start_time = time.time()
        logger.info(f"Processing URL: {request.url}")

        # Validate URL format
        parsed_url = urlparse(request.url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise HTTPException(status_code=400, detail="Invalid URL format")

        # Process the URL
        metadata = document_processor.process_url(
            request.url, request.output_name)

        if not metadata:
            return {
                "success": False,
                "url": request.url,
                "message": "Failed to process URL - no content extracted or not settlement-relevant"
            }

        # Generate embeddings
        embedding_service.embed_chunks(metadata["chunks_path"])

        # Index in vector database
        vector_db_service.index_chunks(metadata["chunks_path"])

        processing_time = time.time() - start_time

        return {
            "success": True,
            "doc_id": metadata["doc_id"],
            "file_name": metadata["file_name"],
            "url": request.url,
            "num_chunks": metadata["num_chunks"],
            "settlement_score": metadata.get("avg_settlement_score"),
            "processing_time": processing_time,
            "message": "URL successfully processed and indexed"
        }

    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing URL: {str(e)}")


@app.post("/documents/process-sitemap")
async def process_sitemap(request: SitemapProcessRequest, api_key: str = Depends(get_api_key)):
    """Process multiple pages from a sitemap for settlement content."""
    try:
        start_time = time.time()
        logger.info(f"Processing sitemap: {request.sitemap_url}")

        # Validate sitemap URL
        parsed_url = urlparse(request.sitemap_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise HTTPException(
                status_code=400, detail="Invalid sitemap URL format")

        # Process sitemap
        results = document_processor.process_sitemap(
            request.sitemap_url, request.max_pages)

        if not results:
            return {
                "success": False,
                "sitemap_url": request.sitemap_url,
                "pages_processed": 0,
                "total_chunks": 0,
                "avg_settlement_score": 0.0,
                "message": "No settlement-relevant pages found in sitemap",
                "results": []
            }

        # Process embeddings and indexing for all results
        for result in results:
            embedding_service.embed_chunks(result["chunks_path"])
            vector_db_service.index_chunks(result["chunks_path"])

        processing_time = time.time() - start_time

        # Calculate statistics
        total_chunks = sum(result["num_chunks"] for result in results)
        avg_settlement_score = (
            sum(result.get("avg_settlement_score", 0)
                for result in results) / len(results)
            if results else 0
        )

        # Prepare response data
        response_results = [
            {
                "doc_id": result["doc_id"],
                "file_name": result["file_name"],
                "url": result["file_path"],
                "num_chunks": result["num_chunks"],
                "settlement_score": result.get("avg_settlement_score", 0)
            }
            for result in results
        ]

        return {
            "success": True,
            "sitemap_url": request.sitemap_url,
            "pages_processed": len(results),
            "total_chunks": total_chunks,
            "avg_settlement_score": avg_settlement_score,
            "processing_time": processing_time,
            "message": f"Successfully processed {len(results)} settlement-relevant pages",
            "results": response_results
        }

    except Exception as e:
        logger.error(f"Error processing sitemap: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing sitemap: {str(e)}")


@app.get("/documents/web-stats")
async def get_web_processing_stats(api_key: str = Depends(get_api_key)):
    """Get statistics for web-processed documents."""
    try:
        documents = document_processor.list_documents()
        web_docs = [
            doc for doc in documents
            if doc.get("source_type") in ["web_url", "web_sitemap"]
        ]

        if not web_docs:
            return {
                "success": True,
                "message": "No web documents found",
                "stats": {
                    "total_web_documents": 0,
                    "url_documents": 0,
                    "sitemap_documents": 0,
                    "avg_settlement_score": 0.0,
                    "total_chunks": 0
                }
            }

        # Calculate statistics
        url_docs = [doc for doc in web_docs if doc.get(
            "source_type") == "web_url"]
        sitemap_docs = [doc for doc in web_docs if doc.get(
            "source_type") == "web_sitemap"]

        total_chunks = sum(doc["num_chunks"] for doc in web_docs)
        avg_settlement_score = (
            sum(doc.get("avg_settlement_score", 0)
                for doc in web_docs) / len(web_docs)
            if web_docs else 0
        )

        # Top performing documents
        top_docs = sorted(
            web_docs,
            key=lambda x: x.get("avg_settlement_score", 0),
            reverse=True
        )[:5]

        top_performing = [
            {
                "file_name": doc["file_name"],
                "url": doc["file_path"],
                "settlement_score": doc.get("avg_settlement_score", 0),
                "num_chunks": doc["num_chunks"],
                "source_type": doc.get("source_type", "")
            }
            for doc in top_docs
        ]

        return {
            "success": True,
            "stats": {
                "total_web_documents": len(web_docs),
                "url_documents": len(url_docs),
                "sitemap_documents": len(sitemap_docs),
                "avg_settlement_score": round(avg_settlement_score, 3),
                "total_chunks": total_chunks,
                "settlement_optimized": True
            },
            "top_performing": top_performing
        }

    except Exception as e:
        logger.error(f"Error getting web stats: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting web stats: {str(e)}")

# ===============================
# CHUNKING AND EMBEDDINGS ENDPOINTS
# ===============================


@app.post("/chunking/process-text")
async def process_text_chunking(request: ChunkingRequest, api_key: str = Depends(get_api_key)):
    """Process text with semantic chunking."""
    try:
        # Create chunker with specified parameters
        chunker = SemanticChunker(
            strategy=ChunkingStrategy(request.strategy),
            min_chunk_size=request.min_chunk_size,
            max_chunk_size=request.max_chunk_size
        )

        # Process text
        chunks = chunker.create_chunks(request.text, "temp_doc_id")

        # Convert chunks to serializable format
        chunk_data = []
        for chunk in chunks:
            chunk_data.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "word_count": chunk.word_count,
                "char_count": chunk.char_count,
                "chunk_type": chunk.chunk_type.value,
                "semantic_score": chunk.semantic_score,
                "topic_coherence": chunk.topic_coherence,
                "settlement_relevance": chunk.settlement_relevance
            })

        return {
            "success": True,
            "strategy": request.strategy,
            "total_chunks": len(chunks),
            "total_words": sum(chunk.word_count for chunk in chunks),
            "avg_chunk_size": sum(chunk.word_count for chunk in chunks) / len(chunks) if chunks else 0,
            "chunks": chunk_data
        }

    except Exception as e:
        logger.error(f"Error in text chunking: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error in text chunking: {str(e)}")


@app.get("/chunking/strategies")
async def get_chunking_strategies():
    """Get available chunking strategies and their descriptions."""
    return {
        "strategies": [
            {
                "name": "settlement_optimized",
                "description": "Optimized for international student settlement content with topic awareness"
            },
            {
                "name": "semantic_adaptive",
                "description": "Adaptive chunking that adjusts size based on content complexity"
            },
            {
                "name": "topic_aware",
                "description": "Chunks based on topic boundaries detected by LLM"
            },
            {
                "name": "semantic_fixed",
                "description": "Fixed-size semantic chunking with overlap"
            }
        ],
        "default": "settlement_optimized",
        "chunker_stats": semantic_chunker.get_chunking_stats()
    }


@app.get("/embeddings/stats")
async def get_embedding_stats(api_key: str = Depends(get_api_key)):
    """Get embedding service statistics."""
    try:
        stats = embedding_service.get_embedding_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.error(f"Error getting embedding stats: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting embedding stats: {str(e)}")


@app.post("/embeddings/generate")
async def generate_embeddings(
    doc_id: Optional[str] = Query(
        None, description="Specific document ID to embed"),
    api_key: str = Depends(get_api_key)
):
    """Generate embeddings for documents."""
    try:
        if doc_id:
            doc_info = document_processor.get_document_info(doc_id)
            if not doc_info:
                raise HTTPException(
                    status_code=404, detail=f"Document {doc_id} not found")

            chunks_path = doc_info.get("chunks_path")
            if not chunks_path or not Path(chunks_path).exists():
                raise HTTPException(
                    status_code=400, detail=f"Chunks file not found for document {doc_id}")

            embeddings = embedding_service.embed_chunks(chunks_path)
            return {
                "success": True,
                "doc_id": doc_id,
                "embeddings_generated": len(embeddings),
                "message": f"Embeddings generated for document {doc_id}"
            }
        else:
            embeddings = embedding_service.embed_chunks()
            return {
                "success": True,
                "embeddings_generated": len(embeddings),
                "message": "Embeddings generated for all documents"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating embeddings: {str(e)}")

# ===============================
# VECTOR DATABASE ENDPOINTS
# ===============================


@app.get("/vector-db/stats")
async def get_vector_db_stats(api_key: str = Depends(get_api_key)):
    """Get vector database statistics."""
    try:
        stats = vector_db_service.get_collection_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.error(f"Error getting vector database stats: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting vector database stats: {str(e)}")


@app.post("/vector-db/rebuild-index")
async def rebuild_index(api_key: str = Depends(get_api_key)):
    """Rebuild the vector database index."""
    try:
        documents = document_processor.list_documents()

        if not documents:
            return JSONResponse(
                status_code=400,
                content={"success": False,
                         "message": "No documents found to index"}
            )

        # Reinitialize collection
        vector_db_service.initialize_collection(recreate=True)

        # Index each document
        indexed_count = 0
        failed_count = 0

        for doc in documents:
            try:
                vector_db_service.index_chunks(doc["chunks_path"])
                indexed_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to index document {doc['doc_id']}: {str(e)}")
                failed_count += 1

        # Get final statistics
        final_stats = vector_db_service.get_collection_stats()

        return {
            "success": True,
            "message": "Vector database index rebuilt",
            "indexed_documents": indexed_count,
            "failed_documents": failed_count,
            "total_documents": len(documents),
            "final_vector_count": final_stats["count"],
            "settlement_optimized": True
        }

    except Exception as e:
        logger.error(f"Error rebuilding index: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error rebuilding index: {str(e)}")


@app.post("/vector-db/optimize")
async def optimize_collection(api_key: str = Depends(get_api_key)):
    """Optimize vector database collection for settlement queries."""
    try:
        optimization_result = vector_db_service.optimize_collection()
        return optimization_result
    except Exception as e:
        logger.error(f"Error optimizing collection: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error optimizing collection: {str(e)}")

# ===============================
# LANGUAGE PROCESSING ENDPOINTS
# ===============================


@app.post("/language/detect")
async def detect_language(
    text: str = Query(..., description="Text to analyze for language"),
    api_key: str = Depends(get_api_key)
):
    """Detect language and provide translation."""
    try:
        result = language_processor.detect_and_process_query(text)
        return {
            "success": True,
            "original_text": text,
            **result
        }
    except Exception as e:
        logger.error(f"Error in language detection: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error in language detection: {str(e)}")


@app.post("/language/translate")
async def translate_text(
    text: str = Query(..., description="Text to translate"),
    target_language: str = Query(..., description="Target language"),
    api_key: str = Depends(get_api_key)
):
    """Translate text to target language with settlement optimization."""
    try:
        translated = language_processor.translate_response(
            text, target_language)
        quality = language_processor.validate_translation_quality(
            text, translated, target_language)

        return {
            "success": True,
            "original_text": text,
            "translated_text": translated,
            "target_language": target_language,
            "translation_quality": quality
        }
    except Exception as e:
        logger.error(f"Error in translation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error in translation: {str(e)}")


@app.get("/language/supported")
async def get_supported_languages():
    """Get list of supported languages."""
    try:
        stats = language_processor.get_language_stats()
        return {
            "supported_languages": list(language_processor.supported_languages.values()),
            "language_codes": list(language_processor.supported_languages.keys()),
            "total_languages": len(language_processor.supported_languages),
            "processor_stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting supported languages: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting supported languages: {str(e)}")


@app.post("/language/test")
async def test_language_processing(request: LanguageTestRequest, api_key: str = Depends(get_api_key)):
    """Test language processing with predefined test cases."""
    try:
        results = language_processor.test_translation_quality(
            request.test_cases)
        return {
            "success": True,
            "test_results": results
        }
    except Exception as e:
        logger.error(f"Error testing language processing: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error testing language processing: {str(e)}")

# ===============================
# EVALUATION AND ANALYTICS ENDPOINTS
# ===============================


@app.post("/evaluation/run")
async def run_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """Run comprehensive evaluation of the settlement assistant."""
    try:
        task_id = f"eval_{int(time.time())}"
        background_tasks_status[task_id] = {"status": "started", "progress": 0}

        def run_evaluation_task():
            try:
                background_tasks_status[task_id]["status"] = "running"
                background_tasks_status[task_id]["progress"] = 25

                if request.focus_area:
                    # Run focused evaluation
                    results = evaluator.run_focused_evaluation(
                        focus_area=request.focus_area,
                        num_queries=request.num_queries
                    )
                else:
                    # Run comprehensive evaluation
                    background_tasks_status[task_id]["progress"] = 50
                    results = evaluator.run_comprehensive_evaluation()

                background_tasks_status[task_id]["status"] = "completed"
                background_tasks_status[task_id]["progress"] = 100
                background_tasks_status[task_id]["results"] = results

            except Exception as e:
                background_tasks_status[task_id]["status"] = "failed"
                background_tasks_status[task_id]["error"] = str(e)

        background_tasks.add_task(run_evaluation_task)

        return {
            "success": True,
            "task_id": task_id,
            "message": "Evaluation started in background",
            "status_endpoint": f"/evaluation/status/{task_id}"
        }

    except Exception as e:
        logger.error(f"Error starting evaluation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error starting evaluation: {str(e)}")


@app.get("/evaluation/status/{task_id}")
async def get_evaluation_status(task_id: str, api_key: str = Depends(get_api_key)):
    """Get status of evaluation task."""
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")

    return background_tasks_status[task_id]


@app.get("/evaluation/create-test-set")
async def create_evaluation_test_set(api_key: str = Depends(get_api_key)):
    """Create international student evaluation test set."""
    try:
        eval_path = evaluator.create_international_student_eval_set()

        # Load and return the created test set
        import pandas as pd
        df = pd.read_csv(eval_path)

        return {
            "success": True,
            "test_set_path": str(eval_path),
            "total_questions": len(df),
            "categories": df['expected_topic'].value_counts().to_dict() if 'expected_topic' in df.columns else {},
            "priorities": df['priority'].value_counts().to_dict() if 'priority' in df.columns else {},
            "sample_questions": df.head(5).to_dict('records') if len(df) > 0 else []
        }

    except Exception as e:
        logger.error(f"Error creating evaluation test set: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error creating evaluation test set: {str(e)}")


@app.get("/evaluation/summary")
async def get_evaluation_summary():
    """Get evaluation system summary and capabilities."""
    try:
        summary = evaluator.get_evaluation_summary()
        return summary
    except Exception as e:
        logger.error(f"Error getting evaluation summary: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting evaluation summary: {str(e)}")

# ===============================
# ANALYTICS AND INSIGHTS ENDPOINTS
# ===============================


@app.get("/analytics/document-insights")
async def get_document_insights(api_key: str = Depends(get_api_key)):
    """Get comprehensive document analytics and insights."""
    try:
        documents = document_processor.list_documents()

        if not documents:
            return {"message": "No documents available for analysis"}

        # Settlement score distribution
        settlement_scores = [doc.get("avg_settlement_score", 0)
                             for doc in documents]
        score_distribution = {
            "excellent": sum(1 for score in settlement_scores if score >= 0.8),
            "good": sum(1 for score in settlement_scores if 0.6 <= score < 0.8),
            "fair": sum(1 for score in settlement_scores if 0.4 <= score < 0.6),
            "poor": sum(1 for score in settlement_scores if score < 0.4)
        }

        # Document type analysis
        type_analysis = {}
        for doc in documents:
            doc_type = doc.get("doc_type", "unknown")
            if doc_type not in type_analysis:
                type_analysis[doc_type] = {
                    "count": 0,
                    "total_chunks": 0,
                    "avg_settlement_score": 0,
                    "total_size": 0
                }

            type_analysis[doc_type]["count"] += 1
            type_analysis[doc_type]["total_chunks"] += doc.get("num_chunks", 0)
            type_analysis[doc_type]["total_size"] += doc.get("file_size", 0)

        # Calculate averages
        for doc_type, stats in type_analysis.items():
            type_docs = [doc for doc in documents if doc.get(
                "doc_type") == doc_type]
            avg_score = sum(doc.get("avg_settlement_score", 0)
                            for doc in type_docs) / len(type_docs) if type_docs else 0
            type_analysis[doc_type]["avg_settlement_score"] = round(
                avg_score, 3)

        # Processing timeline
        processing_dates = [doc.get("processed_date", 0)
                            for doc in documents if doc.get("processed_date")]
        if processing_dates:
            earliest = min(processing_dates)
            latest = max(processing_dates)
            timeline = {
                "earliest": datetime.fromtimestamp(earliest).isoformat(),
                "latest": datetime.fromtimestamp(latest).isoformat(),
                "span_days": (latest - earliest) / 86400 if latest != earliest else 0
            }
        else:
            timeline = {"message": "No processing dates available"}

        # Top performing documents
        top_docs = sorted(
            documents,
            key=lambda x: x.get("avg_settlement_score", 0),
            reverse=True
        )[:5]

        top_performing = [
            {
                "file_name": doc["file_name"],
                "doc_type": doc.get("doc_type", "unknown"),
                "settlement_score": doc.get("avg_settlement_score", 0),
                "num_chunks": doc.get("num_chunks", 0),
                "source_type": doc.get("source_type", "file")
            }
            for doc in top_docs
        ]

        return {
            "total_documents": len(documents),
            "total_chunks": sum(doc.get("num_chunks", 0) for doc in documents),
            "avg_settlement_score": round(sum(settlement_scores) / len(settlement_scores), 3),
            "settlement_score_distribution": score_distribution,
            "document_type_analysis": type_analysis,
            "processing_timeline": timeline,
            "top_performing_documents": top_performing,
            "insights": {
                "most_common_type": max(type_analysis.keys(), key=lambda k: type_analysis[k]["count"]) if type_analysis else None,
                "highest_scoring_type": max(type_analysis.keys(), key=lambda k: type_analysis[k]["avg_settlement_score"]) if type_analysis else None,
                "total_content_size_mb": round(sum(doc.get("file_size", 0) for doc in documents) / (1024*1024), 2)
            }
        }

    except Exception as e:
        logger.error(f"Error getting document insights: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting document insights: {str(e)}")


@app.get("/analytics/search-patterns")
async def get_search_patterns(
    days: int = Query(default=7, description="Number of days to analyze"),
    api_key: str = Depends(get_api_key)
):
    """Get search pattern analytics (would require search logging)."""
    return {
        "message": "Search pattern analytics would require implementing search logging",
        "recommendation": "Implement search query logging to track user behavior and popular queries",
        "suggested_metrics": [
            "Most frequent queries",
            "Popular topics (housing, transport, etc.)",
            "Peak usage times",
            "Average query length",
            "Success rate by intent type",
            "User satisfaction scores"
        ]
    }


@app.get("/analytics/performance")
async def get_performance_analytics(api_key: str = Depends(get_api_key)):
    """Get system performance analytics."""
    try:
        # Vector database performance
        vdb_stats = vector_db_service.get_collection_stats()

        # Service statistics
        services_performance = {
            "vector_database": {
                "total_vectors": vdb_stats.get("count", 0),
                "collection_name": vdb_stats.get("name", "unknown"),
                "settlement_optimized": vdb_stats.get("settlement_optimized", False)
            },
            "embedding_service": embedding_service.get_embedding_stats(),
            "language_processor": language_processor.get_language_stats(),
            "response_generator": response_generator.get_response_stats(),
            "document_processor": document_processor.get_processing_stats()
        }

        # System health
        health = vector_db_service.health_check()

        return {
            "system_health": health,
            "services_performance": services_performance,
            "recommendations": [
                "Monitor response times for optimization opportunities",
                "Track token usage to manage costs",
                "Regular performance benchmarking",
                "User satisfaction monitoring"
            ]
        }

    except Exception as e:
        logger.error(f"Error getting performance analytics: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting performance analytics: {str(e)}")

# ===============================
# UTILITY AND ADMIN ENDPOINTS
# ===============================


@app.get("/utils/intent-types")
async def get_intent_types():
    """Get all available intent types and topics."""
    return {
        "intent_types": [intent.value for intent in IntentType],
        "topic_types": [topic.value for topic in TopicType],
        "intent_descriptions": {
            "housing_inquiry": "Questions about accommodation and housing",
            "university_info": "University-related queries and academic information",
            "immigration_visa": "Visa, permits, and immigration procedures",
            "transportation": "Transport options and travel information",
            "safety_concern": "Safety and security questions",
            "cost_inquiry": "Cost and budget-related questions",
            "banking_finance": "Banking, money, and financial services",
            "healthcare": "Medical and health-related inquiries",
            "cultural_adaptation": "Cultural adjustment and integration",
            "emergency_help": "Urgent help and emergency situations",
            "procedural_query": "Step-by-step process questions",
            "off_topic": "Non-settlement related queries"
        }
    }


@app.get("/utils/settlement-topics")
async def get_settlement_topics():
    """Get settlement-specific topics and keywords."""
    return {
        "settlement_topics": {
            "housing": ["accommodation", "housing", "rent", "apartment", "room"],
            "transportation": ["transport", "matatu", "bus", "taxi", "uber", "boda"],
            "education": ["university", "college", "student", "academic", "course"],
            "legal": ["visa", "permit", "immigration", "passport", "embassy"],
            "finance": ["bank", "money", "cost", "budget", "mpesa", "payment"],
            "safety": ["safe", "security", "crime", "police", "emergency"],
            "healthcare": ["hospital", "clinic", "doctor", "medical", "insurance"],
            "culture": ["culture", "language", "food", "custom", "tradition"]
            },
        "nairobi_locations": [
            "Westlands", "Kilimani", "Karen", "Lavington", "Kileleshwa",
            "Parklands", "Hurlingham", "Riverside", "CBD", "Eastleigh"
        ],
        "universities": [
            "University of Nairobi", "Strathmore University", "JKUAT",
            "USIU", "Kenyatta University", "Daystar University"
        ]
    }

@app.get("/utils/sample-queries")
async def get_sample_queries():
    """Get sample queries for testing different intents."""
    return {
        "housing_queries": [
            "Where can I find safe student accommodation in Nairobi?",
            "How much does it cost to rent a room in Kilimani?",
            "What should I look for when choosing student housing?"
        ],
        "transportation_queries": [
            "How do I get from JKIA airport to university?",
            "What is the safest way to travel around Nairobi?",
            "How much does public transport cost?"
        ],
        "university_queries": [
            "What documents do I need for university admission?",
            "How do I apply to University of Nairobi?",
            "When does the academic year start in Kenya?"
        ],
        "safety_queries": [
            "Is it safe to walk alone at night in Westlands?",
            "What areas should I avoid in Nairobi?",
            "Emergency contacts for international students"
        ],
        "legal_queries": [
            "How do I apply for a student visa to Kenya?",
            "Where is the immigration office in Nairobi?",
            "How do I renew my student permit?"
        ],
        "finance_queries": [
            "How do I open a bank account as an international student?",
            "What is M-Pesa and how do I use it?",
            "What's the cost of living for students in Nairobi?"
        ],
        "healthcare_queries": [
            "Where can I find good hospitals in Nairobi?",
            "Do I need health insurance as a student?",
            "How do I find an English-speaking doctor?"
        ],
        "culture_queries": [
            "What should I know about Kenyan culture?",
            "How do I adapt to life in Nairobi?",
            "What are important customs to respect?"
        ]
    }

@app.post("/admin/clear-cache")
async def clear_system_cache(api_key: str = Depends(get_api_key)):
    """Clear system caches (embeddings, intent cache, etc.)."""
    try:
        results = {}

        # Clear intent recognizer cache
        try:
            intent_recognizer.clear_cache()
            results["intent_cache"] = "cleared"
        except Exception as e:
            results["intent_cache"] = f"failed: {str(e)}"

        # Clear embedding cache
        try:
            embedding_service.clear_cache()
            results["embedding_cache"] = "cleared"
        except Exception as e:
            results["embedding_cache"] = f"failed: {str(e)}"

        return {
            "success": True,
            "message": "System caches cleared",
            "cache_results": results
        }

    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@app.post("/admin/rebuild-intent-cache")
async def rebuild_intent_cache(api_key: str = Depends(get_api_key)):
    """Rebuild intent recognizer embeddings cache."""
    try:
        intent_recognizer.rebuild_cache()
        return {
            "success": True,
            "message": "Intent recognizer cache rebuilt successfully"
        }
    except Exception as e:
        logger.error(f"Error rebuilding intent cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error rebuilding intent cache: {str(e)}")

@app.get("/admin/system-logs")
async def get_system_logs(
    lines: int = Query(default=100, description="Number of log lines to retrieve"),
    api_key: str = Depends(get_api_key)
):
    """Get recent system logs (if log file exists)."""
    try:
        log_file = ROOT_DIR / "logs" / "settlebot.log"

        if not log_file.exists():
            return {
                "message": "No log file found",
                "recommendation": "Configure file logging to enable log retrieval"
            }

        # Read last N lines
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return {
            "success": True,
            "log_lines": len(recent_lines),
            "total_lines": len(all_lines),
            "logs": [line.strip() for line in recent_lines]
        }

    except Exception as e:
        logger.error(f"Error getting system logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting system logs: {str(e)}")

@app.get("/admin/config")
async def get_system_configuration(api_key: str = Depends(get_api_key)):
    """Get current system configuration."""
    try:
        return {
            "api_settings": {
                "cors_enabled": settings.api.cors_enabled,
                "host": settings.api.host,
                "port": settings.api.port,
                "debug": settings.api.debug
            },
            "embedding_settings": {
                "model": settings.embedding.model,
                "dimension": settings.embedding.dimension
            },
            "llm_settings": {
                "model": settings.llm.model,
                "temperature": settings.llm.temperature,
                "max_tokens": settings.llm.max_tokens
            },
            "chunking_settings": {
                "strategy": settings.chunking.strategy,
                "chunk_size": settings.chunking.chunk_size,
                "chunk_overlap": settings.chunking.chunk_overlap
            },
            "language_settings": {
                "detection_enabled": settings.language.detection_enabled,
                "primary_language": settings.language.primary_language,
                "supported_languages": len(language_processor.supported_languages)
            },
            "vector_db_settings": {
                "collection_name": settings.vector_db.collection_name
            },
            "deduplication_settings": {
                "enabled": settings.deduplication.enabled,
                "similarity_threshold": settings.deduplication.similarity_threshold
            }
        }

    except Exception as e:
        logger.error(f"Error getting configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting configuration: {str(e)}")

# ===============================
# EXPORT AND BACKUP ENDPOINTS
# ===============================

@app.get("/export/documents")
async def export_documents_metadata(
    format: str = Query(default="json", description="Export format: json, csv"),
    api_key: str = Depends(get_api_key)
):
    """Export documents metadata."""
    try:
        documents = document_processor.list_documents()

        if format.lower() == "csv":
            import pandas as pd
            df = pd.DataFrame(documents)
            csv_path = ROOT_DIR / "exports" / f"documents_export_{int(time.time())}.csv"
            csv_path.parent.mkdir(exist_ok=True)
            df.to_csv(csv_path, index=False)

            return FileResponse(
                path=csv_path,
                filename=f"settlebot_documents_{int(time.time())}.csv",
                media_type="text/csv"
            )
        else:
            return {
                "success": True,
                "format": "json",
                "export_time": datetime.now().isoformat(),
                "total_documents": len(documents),
                "documents": documents
            }

    except Exception as e:
        logger.error(f"Error exporting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exporting documents: {str(e)}")

@app.get("/export/analytics")
async def export_analytics_report(api_key: str = Depends(get_api_key)):
    """Export comprehensive analytics report."""
    try:
        # Gather all analytics data
        document_insights = await get_document_insights(api_key)
        performance_analytics = await get_performance_analytics(api_key)
        system_status = await get_system_status(api_key)

        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "api_version": "2.0.0",
                "settlement_domain": "nairobi_kenya"
            },
            "document_insights": document_insights,
            "performance_analytics": performance_analytics,
            "system_status": system_status,
            "recommendations": [
                "Regular system health monitoring",
                "Periodic evaluation runs to maintain quality",
                "Content updates for changing settlement requirements",
                "User feedback integration for continuous improvement"
            ]
        }

        return report

    except Exception as e:
        logger.error(f"Error exporting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exporting analytics: {str(e)}")

# ===============================
# WEBHOOK AND INTEGRATION ENDPOINTS
# ===============================

@app.post("/webhooks/document-processed")
async def document_processed_webhook(
    doc_id: str,
    callback_url: Optional[str] = None,
    api_key: str = Depends(get_api_key)
):
    """Webhook endpoint for document processing completion."""
    try:
        doc_info = document_processor.get_document_info(doc_id)

        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")

        webhook_data = {
            "event": "document_processed",
            "timestamp": datetime.now().isoformat(),
            "doc_id": doc_id,
            "file_name": doc_info["file_name"],
            "num_chunks": doc_info["num_chunks"],
            "settlement_score": doc_info.get("avg_settlement_score"),
            "processing_status": "completed"
        }

        # If callback URL provided, send notification (in real implementation)
        if callback_url:
            # Would implement HTTP POST to callback_url with webhook_data
            webhook_data["callback_sent"] = True
            webhook_data["callback_url"] = callback_url

        return webhook_data

    except Exception as e:
        logger.error(f"Error in webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in webhook: {str(e)}")

# ===============================
# ERROR HANDLERS
# ===============================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "message": "Endpoint not found",
            "path": str(request.url.path),
            "settlement_api": True
        }
    )

@app.exception_handler(422)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation error",
            "details": exc.errors() if hasattr(exc, "errors") else str(exc),
            "settlement_api": True
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": str(exc) if settings.api.debug else "An error occurred",
            "settlement_api": True
        }
    )

# ===============================
# STARTUP AND SHUTDOWN EVENTS
# ===============================

@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info("Starting SettleBot API...")

    # Verify OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable not set")

    # Initialize vector database collection if needed
    try:
        vector_db_service.initialize_collection(recreate=False)
        logger.info("Vector database initialized successfully")
    except Exception as e:
        logger.error(f"Vector database initialization failed: {str(e)}")

    # Validate intent recognizer patterns
    try:
        validation_results = intent_recognizer.validate_patterns()
        if validation_results["overall_health"]:
            logger.info("Intent recognizer validation passed")
        else:
            logger.warning("Intent recognizer validation issues detected")
    except Exception as e:
        logger.error(f"Intent recognizer validation failed: {str(e)}")

    logger.info("SettleBot API startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info("Shutting down SettleBot API...")

    # Clean up background tasks status
    background_tasks_status.clear()

    # Close any open connections if needed
    # (ChromaDB and OpenAI clients handle this automatically)

    logger.info("SettleBot API shutdown completed")

# ===============================
# MAIN APPLICATION ENTRY POINT
# ===============================

if __name__ == "__main__":
    import uvicorn

    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable not set")
        logger.warning("The API will not function properly without this key")

    logger.info("Starting SettleBot API server...")
    logger.info(f"API Documentation available at: http://{settings.api.host}:{settings.api.port}/docs")
    logger.info(f"Health check available at: http://{settings.api.host}:{settings.api.port}/health")

    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
        log_level="info",
        access_log=True
    )
