"""Main FastAPI application"""
import structlog
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime

from app.config import get_settings
from app.models.schemas import (
    QueryRequest, QueryResponse, TranscribeResponse,
    HealthResponse, MetricsResponse
)
from app.services.stt import get_stt_service
from app.services.embeddings import get_embedding_service
from app.services.retrieval import get_retrieval_service
from app.services.harness import get_harness

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - load models on startup"""
    logger.info("application_starting")

    # Load embedding model
    embedding_service = get_embedding_service()
    embedding_service.load_model()
    logger.info("embedding_model_loaded")

    # Load FAISS index
    retrieval_service = get_retrieval_service()
    index_loaded = retrieval_service.load_index()
    logger.info("faiss_index_status", loaded=index_loaded)

    yield

    logger.info("application_shutting_down")


# Create FastAPI app
app = FastAPI(
    title="Voice-Enabled RAG System",
    description="Production-quality RAG system with speech-to-text, multi-strategy chunking, FAISS retrieval, reranking, and guardrails",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID middleware
@app.middleware("http")
async def add_request_id(request, call_next):
    """Add request ID to all requests for tracing"""
    import uuid
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Health check endpoint
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    retrieval_service = get_retrieval_service()
    embedding_service = get_embedding_service()

    return HealthResponse(
        status="healthy" if retrieval_service.is_loaded() else "degraded",
        version="1.0.0",
        index_loaded=retrieval_service.is_loaded(),
        embedding_model_loaded=embedding_service.is_loaded()
    )


# Metrics endpoint
@app.get("/api/metrics", response_model=MetricsResponse)
async def metrics():
    """System metrics endpoint"""
    retrieval_service = get_retrieval_service()

    # In a real implementation, these would come from a metrics store
    return MetricsResponse(
        total_queries=0,
        successful_queries=0,
        refused_queries=0,
        avg_latency_ms=0.0,
        p50_latency_ms=0.0,
        p90_latency_ms=0.0,
        p99_latency_ms=0.0,
        avg_retrieval_score=0.0,
        index_size=retrieval_service.get_num_vectors()
    )


# Speech-to-text endpoint
@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(...),
    language: str = "en"
):
    """
    Transcribe audio using Sarvam STT

    Args:
        audio: Audio file upload
        language: Language code (en, hi, bn, etc.)

    Returns:
        Transcription result
    """
    stt_service = get_stt_service()

    # Read audio data
    audio_data = await audio.read()

    # Transcribe
    result = await stt_service.transcribe(
        audio_data=audio_data,
        language=language,
        audio_format=audio.filename.split('.')[-1] if audio.filename else 'wav'
    )

    return result


# Query endpoint
@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process RAG query

    Pipeline:
    1. Query embedding
    2. Vector retrieval
    3. Reranking
    4. Answer generation
    5. Grounding verification
    6. Guardrails

    Returns:
        Answer with citations and metadata
    """
    harness = get_harness()

    # Check if system is ready
    if not harness.is_ready():
        raise HTTPException(
            status_code=503,
            detail="System not ready. Please ensure the FAISS index is built."
        )

    # Process query
    response = harness.process_query(request)

    return response


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Voice-Enabled RAG System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
