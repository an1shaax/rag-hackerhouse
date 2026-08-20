"""Pydantic models for API requests and responses"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ChunkingStrategy(str, Enum):
    """Available chunking strategies"""
    FIXED = "fixed"
    FIXED_OVERLAP = "fixed_overlap"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    METADATA_AWARE = "metadata_aware"


class Language(str, Enum):
    """Supported languages"""
    EN = "en"
    HI = "hi"
    BN = "bn"
    TA = "ta"
    TE = "te"
    MR = "mr"
    GU = "gu"
    KN = "kn"
    ML = "ml"
    PA = "pa"
    OR = "or"
    AS = "as"
    UR = "ur"
    NE = "ne"
    SA = "sa"


# Request Models
class TranscribeRequest(BaseModel):
    """Speech-to-text transcription request"""
    language: Optional[Language] = Language.EN
    audio_format: str = "wav"


class QueryRequest(BaseModel):
    """RAG query request"""
    query: str = Field(..., min_length=1, max_length=2000)
    language: Optional[Language] = Language.EN
    top_k: Optional[int] = None
    chunk_strategy: Optional[ChunkingStrategy] = None


# Response Models
class Citation(BaseModel):
    """Citation for retrieved context"""
    chunk_id: str
    document_id: str
    source: str
    score: float
    text_preview: str = Field(..., max_length=200)


class LatencyBreakdown(BaseModel):
    """Latency breakdown by component"""
    stt_ms: Optional[float] = None
    query_embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    grounding_ms: float = 0.0
    guardrails_ms: float = 0.0
    total_rag_ms: float = 0.0
    total_ms: float = 0.0


class QueryResponse(BaseModel):
    """RAG query response"""
    request_id: str
    query: str
    language: str
    answer: str
    grounded: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    citations: List[Citation]
    latency: LatencyBreakdown
    refused: bool = False
    refusal_reason: Optional[str] = None
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TranscribeResponse(BaseModel):
    """Speech-to-text response"""
    request_id: str
    transcription: str
    language: str
    confidence: float
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    index_loaded: bool
    embedding_model_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetricsResponse(BaseModel):
    """System metrics response"""
    total_queries: int
    successful_queries: int
    refused_queries: int
    avg_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    avg_retrieval_score: float
    index_size: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Internal Models
class Chunk(BaseModel):
    """Document chunk with metadata"""
    chunk_id: str
    document_id: str
    text: str
    language: str
    source: str
    chunking_strategy: str
    position: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """Retrieved chunk with relevance score"""
    chunk: Chunk
    score: float


class GeneratedAnswer(BaseModel):
    """Generated answer from LLM"""
    answer: str
    grounded: bool
    confidence: float
    citations: List[str]
    raw_response: Optional[str] = None


class BenchmarkResult(BaseModel):
    """Single benchmark result"""
    query_id: str
    query: str
    embedding_latency_ms: float
    retrieval_latency_ms: float
    reranking_latency_ms: float
    generation_latency_ms: float
    grounding_latency_ms: float
    total_rag_latency_ms: float
    total_latency_ms: float
    success: bool
    refused: bool
    answer_length: int
    num_citations: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BenchmarkReport(BaseModel):
    """Full benchmark report"""
    total_queries: int
    successful_queries: int
    failed_queries: int
    refused_queries: int
    p50_latency_ms: float
    p70_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    p100_latency_ms: float
    mean_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    avg_component_latencies: Dict[str, float]
    results: List[BenchmarkResult]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
