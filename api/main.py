import os
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from urllib.parse import urlparse

from fastapi import Depends
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import Query
from fastapi import Security
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from pydantic import Field

from config.settings import settings
from services.document_processor import DocumentProcessor
from services.embeddings import EmbeddingService
from services.intent_recognizer import IntentRecognizer
from services.language_processor import LanguageProcessor
from services.response_generator import ResponseGenerator
from services.vector_db import VectorDBService

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("settlebot_api")

# Initialize FastAPI app
app = FastAPI(
    title="SettleBot API",
    description="RAG-powered settlement assistant for international students in Nairobi, Kenya",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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

# Upload directory
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# API Models
class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=15, ge=1, le=50)
    include_context: bool = True
    language_detection: bool = True


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


class URLProcessRequest(BaseModel):
    url: str
    output_name: Optional[str] = None
    validate_first: bool = True


class SitemapProcessRequest(BaseModel):
    sitemap_url: str
    max_pages: int = Field(default=50, ge=1, le=200)
    settlement_filter: bool = True


class WebValidationResponse(BaseModel):
    url: str
    is_settlement_relevant: bool
    word_count: int
    keyword_matches: List[str]
    recommendation: str


class WebProcessingResponse(BaseModel):
    success: bool
    doc_id: Optional[str] = None
    file_name: Optional[str] = None
    url: str
    num_chunks: int = 0
    settlement_score: Optional[float] = None
    processing_time: Optional[float] = None
    message: str


class SitemapProcessingResponse(BaseModel):
    success: bool
    sitemap_url: str
    pages_processed: int
    total_chunks: int
    avg_settlement_score: float
    processing_time: Optional[float] = None
    message: str
    results: List[Dict[str, Any]]


# Dependency for API key validation
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        raise HTTPException(status_code=401, detail="Missing API Key")

    # Extract token from "Bearer {token}" format
    if api_key_header.startswith("Bearer "):
        api_key_header = api_key_header.split(" ")[1]

    if api_key_header != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    return api_key_header


# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    try:
        # Check core services
        health_status = {
            "status": "healthy",
            "timestamp": response_generator.get_current_nairobi_time()[0],
            "api_version": "1.0.0",
            "settlement_domain": "nairobi_kenya",
        }

        # Quick service checks
        try:
            vector_stats = vector_db_service.get_collection_stats()
            health_status["vector_db"] = {
                "status": "healthy",
                "count": vector_stats["count"],
            }
        except Exception as e:
            health_status["vector_db"] = {"status": "error", "error": str(e)}

        # Check language processing
        try:
            lang_stats = language_processor.get_language_stats()
            health_status["language_processing"] = {
                "status": "healthy",
                "detection_enabled": lang_stats["detection_enabled"],
            }
        except Exception as e:
            health_status["language_processing"] = {
                "status": "error",
                "error": str(e),
            }

        # Check API key
        health_status["openai_api"] = {
            "status": "configured" if os.getenv("OPENAI_API_KEY") else "missing"
        }

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}


@app.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status(api_key: str = Depends(get_api_key)):
    """Get comprehensive system status."""
    try:
        # Service status checks
        services = {}

        # Vector database
        try:
            vdb_health = vector_db_service.health_check()
            services["vector_database"] = vdb_health
        except Exception as e:
            services["vector_database"] = {"status": "error", "error": str(e)}

        # Document processor
        try:
            doc_stats = document_processor.get_processing_stats()
            services["document_processor"] = {
                "status": "healthy",
                "total_documents": doc_stats["total_documents"],
                "supported_formats": doc_stats["supported_formats"],
            }
        except Exception as e:
            services["document_processor"] = {
                "status": "error",
                "error": str(e),
            }

        # Embedding service
        try:
            embedding_stats = embedding_service.get_embedding_stats()
            services["embedding_service"] = {
                "status": "healthy",
                "model": embedding_stats["model"],
                "total_embeddings": embedding_stats["total_embeddings"],
            }
        except Exception as e:
            services["embedding_service"] = {
                "status": "error", "error": str(e)}

        # Language processor
        try:
            lang_stats = language_processor.get_language_stats()
            services["language_processor"] = {
                "status": "healthy",
                "supported_languages": lang_stats["supported_languages"],
                "detection_enabled": lang_stats["detection_enabled"],
            }
        except Exception as e:
            services["language_processor"] = {
                "status": "error",
                "error": str(e),
            }

        # Response generator
        try:
            response_stats = response_generator.get_response_stats()
            services["response_generator"] = {
                "status": "healthy",
                "model": response_stats["model"],
                "settlement_optimized": response_stats["settlement_optimized"],
            }
        except Exception as e:
            services["response_generator"] = {
                "status": "error",
                "error": str(e),
            }

        # Configuration
        configuration = {
            "chunking_strategy": settings.chunking.strategy,
            "deduplication_enabled": settings.deduplication.enabled,
            "language_detection": settings.language.detection_enabled,
            "embedding_model": settings.embedding.model,
            "llm_model": settings.llm.model,
        }

        # Settlement optimization status
        settlement_optimization = {
            "domain": "international_student_settlement",
            "location": "nairobi_kenya",
            "semantic_chunking": settings.chunking.strategy == "semantic",
            "multilingual_support": settings.language.detection_enabled,
            "settlement_scoring": True,
            "location_entity_extraction": True,
            "cost_entity_extraction": True,
        }

        # Overall status
        healthy_services = sum(
            1
            for service in services.values()
            if service.get("status") == "healthy"
        )
        total_services = len(services)
        overall_status = (
            "healthy" if healthy_services == total_services else "degraded"
        )

        return SystemStatusResponse(
            status=overall_status,
            services=services,
            configuration=configuration,
            settlement_optimization=settlement_optimization,
        )

    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting system status: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest, api_key: str = Depends(get_api_key)
):
    """Process settlement query with multilingual support."""
    try:
        logger.info(f"Processing settlement query: {request.query[:100]}...")

        # Language detection and processing
        if request.language_detection:
            language_result = language_processor.detect_and_process_query(
                request.query
            )
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
        intent_info = intent_recognizer.recognize_intent(english_query)
        logger.info(f"Recognized intent: {intent_info['intent_type'].value}")

        # Retrieve relevant context
        retrieved_chunks = []
        if intent_info["intent_type"].value != "off_topic":
            # Use settlement-optimized search
            retrieved_chunks = vector_db_service.search(
                query=english_query, top_k=request.top_k
            )

        # Generate response
        response_data = response_generator.generate_response(
            query=request.query,
            retrieved_context=retrieved_chunks,
            intent_info=intent_info,
        )

        # Create language info
        language_info = LanguageInfo(
            detected_language=language_result["detected_language"],
            original_query=language_result["original_query"],
            english_query=language_result["english_query"],
            translation_needed=language_result["needs_translation"],
            confidence=language_result.get("confidence", 0.0),
        )

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
        )

        # Include debug information if requested
        if request.include_context:
            query_response.retrieved_chunks = retrieved_chunks

        return query_response

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing query: {str(e)}"
        )


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...), api_key: str = Depends(get_api_key)
):
    """Upload and process settlement document."""
    try:
        import time

        start_time = time.time()

        # Validate file type
        file_path = Path(file.filename)
        if not document_processor.is_file_supported(file_path):
            supported = ", ".join(
                document_processor.get_supported_extensions())
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: {supported}",
            )

        # Save uploaded file
        temp_file_path = UPLOAD_DIR / file.filename
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_size = temp_file_path.stat().st_size
        logger.info(
            f"Processing uploaded file: {temp_file_path.name} ({file_size} bytes)"
        )

        # Process document
        metadata = document_processor.process_document(temp_file_path)

        if not metadata:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Failed to process document",
                },
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
            settlement_score=metadata.get("avg_settlement_score"),
        )

    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        # Clean up temp file if it exists
        if "temp_file_path" in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        raise HTTPException(
            status_code=500, detail=f"Error uploading document: {str(e)}"
        )


@app.get("/documents")
async def list_documents(
    doc_type: Optional[str] = Query(
        None, description="Filter by document type"
    ),
    api_key: str = Depends(get_api_key),
):
    """List processed documents."""
    try:
        if doc_type:
            documents = [
                doc
                for doc in document_processor.list_documents()
                if doc.get("doc_type") == doc_type
            ]
        else:
            documents = document_processor.list_documents()

        # Calculate statistics
        total_chunks = sum(doc.get("num_chunks", 0) for doc in documents)
        avg_settlement_score = (
            sum(doc.get("avg_settlement_score", 0) for doc in documents)
            / len(documents)
            if documents
            else 0
        )

        return {
            "documents": documents,
            "count": len(documents),
            "total_chunks": total_chunks,
            "avg_settlement_score": round(avg_settlement_score, 3),
            "supported_formats": document_processor.get_supported_extensions(),
        }

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error listing documents: {str(e)}"
        )


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, api_key: str = Depends(get_api_key)):
    """Delete a document and its associated data."""
    try:
        # Check if document exists
        doc_info = document_processor.get_document_info(doc_id)
        if not doc_info:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": f"Document {doc_id} not found",
                },
            )

        # Delete document
        success = document_processor.delete_document(doc_id)

        if success:
            return {
                "success": True,
                "message": f"Document {doc_id} successfully deleted",
                "doc_id": doc_id,
                "file_name": doc_info["file_name"],
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Failed to delete document {doc_id}",
                },
            )

    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting document: {str(e)}"
        )


@app.post("/system/rebuild-index")
async def rebuild_index(api_key: str = Depends(get_api_key)):
    """Rebuild the vector database index."""
    try:
        documents = document_processor.list_documents()

        if not documents:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "No documents found to index",
                },
            )

        # Reinitialize collection
        vector_db_service.initialize_collection(recreate=True)

        # Index each document
        indexed_count = 0
        for doc in documents:
            vector_db_service.index_chunks(doc["chunks_path"])
            indexed_count += 1

        # Get final statistics
        final_stats = vector_db_service.get_collection_stats()

        return {
            "success": True,
            "message": "Vector database index rebuilt successfully",
            "indexed_documents": indexed_count,
            "total_documents": len(documents),
            "final_vector_count": final_stats["count"],
            "settlement_optimized": True,
        }

    except Exception as e:
        logger.error(f"Error rebuilding index: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error rebuilding index: {str(e)}"
        )


@app.get("/search/topics")
async def search_by_topic(
    topic: str = Query(..., description="Settlement topic to search"),
    top_k: int = Query(10, ge=1, le=50),
    api_key: str = Depends(get_api_key),
):
    """Search by settlement topic."""
    try:
        results = vector_db_service.search_by_topic(topic, top_k)

        return {
            "topic": topic,
            "results": results,
            "count": len(results),
            "settlement_optimized": True,
        }

    except Exception as e:
        logger.error(f"Error in topic search: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error in topic search: {str(e)}"
        )


@app.get("/search/locations")
async def search_by_location(
    location: str = Query(..., description="Nairobi location to search"),
    query: str = Query("", description="Optional additional query"),
    top_k: int = Query(10, ge=1, le=50),
    api_key: str = Depends(get_api_key),
):
    """Search by Nairobi location."""
    try:
        results = vector_db_service.search_by_location(location, query, top_k)

        return {
            "location": location,
            "query": query,
            "results": results,
            "count": len(results),
            "settlement_optimized": True,
        }

    except Exception as e:
        logger.error(f"Error in location search: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error in location search: {str(e)}"
        )


@app.get("/stats/embedding")
async def get_embedding_stats(api_key: str = Depends(get_api_key)):
    """Get embedding service statistics."""
    try:
        stats = embedding_service.get_embedding_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.error(f"Error getting embedding stats: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting embedding stats: {str(e)}"
        )


@app.get("/stats/vector-db")
async def get_vector_db_stats(api_key: str = Depends(get_api_key)):
    """Get vector database statistics."""
    try:
        stats = vector_db_service.get_collection_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.error(f"Error getting vector database stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting vector database stats: {str(e)}",
        )


@app.post("/web/validate-url", response_model=WebValidationResponse)
async def validate_url(
    request: URLProcessRequest, api_key: str = Depends(get_api_key)
):
    """Validate if a URL contains settlement-relevant content."""
    try:
        from langchain_community.document_loaders import WebBaseLoader

        logger.info(f"Validating URL: {request.url}")

        # Load content to validate
        loader = WebBaseLoader(request.url)
        documents = loader.load()

        if not documents:
            raise HTTPException(
                status_code=400, detail="No content found at URL"
            )

        text = "\n\n".join(doc.page_content for doc in documents)

        if not text.strip():
            raise HTTPException(status_code=400, detail="Empty content at URL")

        # Check settlement relevance
        is_relevant = document_processor._is_settlement_relevant(text)

        # Calculate statistics
        word_count = len(text.split())

        # Find settlement keywords
        settlement_keywords = [
            "international student",
            "nairobi",
            "housing",
            "accommodation",
            "visa",
            "university",
            "transport",
            "safety",
            "cost of living",
            "westlands",
            "kilimani",
            "karen",
            "student life",
        ]

        keyword_matches = [
            keyword
            for keyword in settlement_keywords
            if keyword.lower() in text.lower()
        ]

        recommendation = (
            "Recommended for processing - contains settlement-relevant content"
            if is_relevant
            else "May not be relevant - review content before processing"
        )

        return WebValidationResponse(
            url=request.url,
            is_settlement_relevant=is_relevant,
            word_count=word_count,
            keyword_matches=keyword_matches[:10],  # Limit to top 10
            recommendation=recommendation,
        )

    except Exception as e:
        logger.error(f"Error validating URL: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error validating URL: {str(e)}"
        )


@app.post("/web/process-url", response_model=WebProcessingResponse)
async def process_url(
    request: URLProcessRequest, api_key: str = Depends(get_api_key)
):
    """Process content from a URL for settlement information."""
    try:
        import time

        start_time = time.time()

        logger.info(f"Processing URL: {request.url}")

        # Validate URL format
        parsed_url = urlparse(request.url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise HTTPException(status_code=400, detail="Invalid URL format")

        # Optional validation first
        if request.validate_first:
            from langchain_community.document_loaders import WebBaseLoader

            # Quick validation
            loader = WebBaseLoader(request.url)
            documents = loader.load()

            if documents:
                text = "\n\n".join(doc.page_content for doc in documents)
                if not document_processor._is_settlement_relevant(text):
                    logger.warning(
                        f"URL may not be settlement-relevant: {request.url}"
                    )

        # Process the URL
        metadata = document_processor.process_url(
            request.url, request.output_name
        )

        if not metadata:
            return WebProcessingResponse(
                success=False,
                url=request.url,
                message="Failed to process URL - no content extracted or not settlement-relevant",
            )

        # Generate embeddings
        embedding_service.embed_chunks(metadata["chunks_path"])

        # Index in vector database
        vector_db_service.index_chunks(metadata["chunks_path"])

        processing_time = time.time() - start_time

        return WebProcessingResponse(
            success=True,
            doc_id=metadata["doc_id"],
            file_name=metadata["file_name"],
            url=request.url,
            num_chunks=metadata["num_chunks"],
            settlement_score=metadata.get("avg_settlement_score"),
            processing_time=processing_time,
            message="URL successfully processed and indexed",
        )

    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing URL: {str(e)}"
        )


@app.post("/web/process-sitemap", response_model=SitemapProcessingResponse)
async def process_sitemap(
    request: SitemapProcessRequest, api_key: str = Depends(get_api_key)
):
    """Process multiple pages from a sitemap for settlement content."""
    try:
        import time

        start_time = time.time()

        logger.info(f"Processing sitemap: {request.sitemap_url}")

        # Validate sitemap URL
        parsed_url = urlparse(request.sitemap_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise HTTPException(
                status_code=400, detail="Invalid sitemap URL format"
            )

        # Process sitemap
        results = document_processor.process_sitemap(
            request.sitemap_url, request.max_pages
        )

        if not results:
            return SitemapProcessingResponse(
                success=False,
                sitemap_url=request.sitemap_url,
                pages_processed=0,
                total_chunks=0,
                avg_settlement_score=0.0,
                message="No settlement-relevant pages found in sitemap",
                results=[],
            )

        # Process embeddings and indexing for all results
        for result in results:
            embedding_service.embed_chunks(result["chunks_path"])
            vector_db_service.index_chunks(result["chunks_path"])

        processing_time = time.time() - start_time

        # Calculate statistics
        total_chunks = sum(result["num_chunks"] for result in results)
        avg_settlement_score = (
            sum(result.get("avg_settlement_score", 0) for result in results)
            / len(results)
            if results
            else 0
        )

        # Prepare response data
        response_results = [
            {
                "doc_id": result["doc_id"],
                "file_name": result["file_name"],
                "url": result["file_path"],
                "num_chunks": result["num_chunks"],
                "settlement_score": result.get("avg_settlement_score", 0),
            }
            for result in results
        ]

        return SitemapProcessingResponse(
            success=True,
            sitemap_url=request.sitemap_url,
            pages_processed=len(results),
            total_chunks=total_chunks,
            avg_settlement_score=avg_settlement_score,
            processing_time=processing_time,
            message=f"Successfully processed {len(results)} settlement-relevant pages",
            results=response_results,
        )

    except Exception as e:
        logger.error(f"Error processing sitemap: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing sitemap: {str(e)}"
        )


@app.get("/web/stats")
async def get_web_processing_stats(api_key: str = Depends(get_api_key)):
    """Get statistics for web-processed documents."""
    try:
        documents = document_processor.list_documents()
        web_docs = [
            doc
            for doc in documents
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
                    "total_chunks": 0,
                },
            }

        # Calculate statistics
        url_docs = [
            doc for doc in web_docs if doc.get("source_type") == "web_url"
        ]
        sitemap_docs = [
            doc for doc in web_docs if doc.get("source_type") == "web_sitemap"
        ]

        total_chunks = sum(doc["num_chunks"] for doc in web_docs)
        avg_settlement_score = (
            sum(doc.get("avg_settlement_score", 0) for doc in web_docs)
            / len(web_docs)
            if web_docs
            else 0
        )

        # Top performing documents
        top_docs = sorted(
            web_docs,
            key=lambda x: x.get("avg_settlement_score", 0),
            reverse=True,
        )[:5]

        top_performing = [
            {
                "file_name": doc["file_name"],
                "url": doc["file_path"],
                "settlement_score": doc.get("avg_settlement_score", 0),
                "num_chunks": doc["num_chunks"],
                "source_type": doc.get("source_type", ""),
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
                "settlement_optimized": True,
            },
            "top_performing": top_performing,
        }

    except Exception as e:
        logger.error(f"Error getting web stats: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting web stats: {str(e)}"
        )


@app.delete("/web/documents")
async def delete_web_documents(
    source_type: str = Query(
        ..., description="Source type: 'web_url' or 'web_sitemap' or 'all'"
    ),
    api_key: str = Depends(get_api_key),
):
    """Delete web-processed documents by source type."""
    try:
        documents = document_processor.list_documents()

        if source_type == "all":
            web_docs = [
                doc
                for doc in documents
                if doc.get("source_type") in ["web_url", "web_sitemap"]
            ]
        else:
            web_docs = [
                doc
                for doc in documents
                if doc.get("source_type") == source_type
            ]

        if not web_docs:
            return {
                "success": True,
                "message": f"No web documents found for source type: {source_type}",
                "deleted_count": 0,
            }

        # Delete documents
        deleted_count = 0
        failed_deletions = []

        for doc in web_docs:
            try:
                success = document_processor.delete_document(doc["doc_id"])
                if success:
                    deleted_count += 1
                else:
                    failed_deletions.append(doc["doc_id"])
            except Exception as e:
                logger.error(
                    f"Failed to delete document {doc['doc_id']}: {str(e)}"
                )
                failed_deletions.append(doc["doc_id"])

        message = f"Successfully deleted {deleted_count} web documents"
        if failed_deletions:
            message += f". Failed to delete {len(failed_deletions)} documents"

        return {
            "success": True,
            "message": message,
            "deleted_count": deleted_count,
            "failed_deletions": failed_deletions,
            "source_type": source_type,
        }

    except Exception as e:
        logger.error(f"Error deleting web documents: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting web documents: {str(e)}"
        )


# Error handlers


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "message": "Endpoint not found",
            "path": str(request.url.path),
        },
    )


@app.exception_handler(422)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation error",
            "details": exc.errors() if hasattr(exc, "errors") else str(exc),
        },
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
        },
    )


if __name__ == "__main__":
    import uvicorn

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable not set")

    logger.info("Starting SettleBot API server...")
    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
        log_level="info",
    )
