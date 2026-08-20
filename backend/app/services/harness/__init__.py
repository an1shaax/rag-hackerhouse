"""Main orchestration harness for the RAG pipeline"""
from typing import Optional, List, Dict, Any
import structlog
from app.config import get_settings
from app.models.schemas import (
    QueryRequest, QueryResponse, Citation, LatencyBreakdown,
    RetrievedChunk, GeneratedAnswer
)
from app.services.embeddings import get_embedding_service
from app.services.retrieval import get_retrieval_service
from app.services.reranking import get_reranker_service
from app.services.generation import get_generation_service
from app.services.guardrails import get_guardrails_service
import time
import uuid

logger = structlog.get_logger()


class RAGHarness:
    """Orchestration harness for the complete RAG pipeline"""

    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.retrieval_service = get_retrieval_service()
        self.reranker_service = get_reranker_service()
        self.generation_service = get_generation_service()
        self.guardrails_service = get_guardrails_service()

    def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a RAG query through the complete pipeline

        Pipeline stages:
        1. Input validation
        2. Query embedding
        3. Vector retrieval
        4. Reranking
        5. Relevance filtering
        6. Answer generation
        7. Grounding verification
        8. Guardrails check
        9. Final response

        Args:
            request: Query request

        Returns:
            QueryResponse with answer and metadata
        """
        request_id = str(uuid.uuid4())
        total_start = time.time()

        latencies = LatencyBreakdown()
        refused = False
        refusal_reason = None

        logger.info(
            "query_started",
            request_id=request_id,
            query=request.query[:100],
            language=request.language
        )

        # Stage 1: Input validation
        guardrail_start = time.time()
        is_valid, error = self.guardrails_service.validate_input(request.query)
        latencies.guardrails_ms = (time.time() - guardrail_start) * 1000

        if not is_valid:
            logger.warning("query_rejected", request_id=request_id, reason=error)
            return self._create_refusal_response(
                request_id=request_id,
                query=request.query,
                language=request.language.value,
                reason=error or "Invalid query",
                latencies=latencies,
                total_start=total_start
            )

        # Stage 2: Query embedding
        embedding_start = time.time()
        query_embedding = self.embedding_service.embed_single(request.query)
        latencies.query_embedding_ms = (time.time() - embedding_start) * 1000

        # Stage 3: Vector retrieval
        retrieval_start = time.time()
        # Map language codes: "en" -> "eng" (index uses "eng")
        lang_map = {"en": "eng", "hi": "hin", "bn": "ben", "ta": "tam", "te": "tel"}
        search_lang = lang_map.get(request.language.value, request.language.value)
        chunks, retrieval_latency = self.retrieval_service.search(
            query_embedding,
            top_k=request.top_k or self.settings.top_k,
            language=search_lang
        )
        latencies.retrieval_ms = retrieval_latency

        # Stage 4: Check retrieval quality
        has_context, context_reason = self.guardrails_service.check_retrieval_quality(
            chunks, request.query
        )

        if not has_context:
            logger.info("insufficient_context", request_id=request_id, reason=context_reason)
            return self._create_refusal_response(
                request_id=request_id,
                query=request.query,
                language=request.language.value,
                reason=context_reason or "Insufficient relevant context",
                latencies=latencies,
                total_start=total_start
            )

        # Stage 5: Reranking
        rerank_start = time.time()
        reranked_chunks, rerank_latency = self.reranker_service.rerank(
            query=request.query,
            chunks=chunks,
            top_k=self.settings.rerank_top_k
        )
        latencies.reranking_ms = rerank_latency

        # Stage 6: Answer generation
        generation_start = time.time()
        generated_answer, generation_latency = self.generation_service.generate(
            query=request.query,
            context_chunks=reranked_chunks,
            language=request.language.value
        )
        latencies.generation_ms = generation_latency

        # Stage 7: Grounding verification
        grounding_start = time.time()
        is_grounded, confidence = self.guardrails_service.verify_grounding(
            generated_answer, reranked_chunks, request.query
        )
        latencies.grounding_ms = (time.time() - grounding_start) * 1000

        # Stage 8: Output safety check
        guardrail_start = time.time()
        is_safe, safety_error = self.guardrails_service.check_output_safety(
            generated_answer.answer
        )
        latencies.guardrails_ms += (time.time() - guardrail_start) * 1000

        if not is_grounded or not is_safe:
            logger.info(
                "answer_rejected",
                request_id=request_id,
                grounded=is_grounded,
                safe=is_safe
            )
            return self._create_refusal_response(
                request_id=request_id,
                query=request.query,
                language=request.language.value,
                reason="Could not generate a grounded answer from the available context.",
                latencies=latencies,
                total_start=total_start
            )

        # Stage 9: Build final response
        latencies.total_rag_ms = (time.time() - total_start) * 1000
        latencies.total_ms = latencies.total_rag_ms

        citations = self._build_citations(reranked_chunks, generated_answer.citations)

        logger.info(
            "query_completed",
            request_id=request_id,
            grounded=is_grounded,
            confidence=confidence,
            total_latency_ms=latencies.total_ms
        )

        return QueryResponse(
            request_id=request_id,
            query=request.query,
            language=request.language.value,
            answer=generated_answer.answer,
            grounded=is_grounded,
            confidence=confidence,
            citations=citations,
            latency=latencies,
            refused=False,
            retrieved_chunks=[{"chunk_id": c.chunk.chunk_id, "score": c.score, "text_preview": c.chunk.text[:200]} for c in reranked_chunks[:3]]
        )

    def _create_refusal_response(
        self,
        request_id: str,
        query: str,
        language: str,
        reason: str,
        latencies: LatencyBreakdown,
        total_start: float
    ) -> QueryResponse:
        """Create a refusal response"""
        latencies.total_rag_ms = (time.time() - total_start) * 1000
        latencies.total_ms = latencies.total_rag_ms

        return QueryResponse(
            request_id=request_id,
            query=query,
            language=language,
            answer="I don't have enough information in the provided dataset to answer that.",
            grounded=False,
            confidence=0.0,
            citations=[],
            latency=latencies,
            refused=True,
            refusal_reason=reason
        )

    def _build_citations(
        self,
        chunks: List[RetrievedChunk],
        cited_ids: List[str]
    ) -> List[Citation]:
        """Build citation list from chunks"""
        citations = []
        cited_set = set(cited_ids) if cited_ids else set()

        for chunk in chunks:
            if not cited_set or chunk.chunk.chunk_id in cited_set:
                citations.append(Citation(
                    chunk_id=chunk.chunk.chunk_id,
                    document_id=chunk.chunk.document_id,
                    source=chunk.chunk.source,
                    score=chunk.score,
                    text_preview=chunk.chunk.text[:200]
                ))

        return citations

    def is_ready(self) -> bool:
        """Check if all services are ready"""
        return (
            self.embedding_service.is_loaded() and
            self.retrieval_service.is_loaded()
        )


# Singleton instance
_harness: Optional[RAGHarness] = None


def get_harness() -> RAGHarness:
    """Get RAG harness singleton"""
    global _harness
    if _harness is None:
        _harness = RAGHarness()
    return _harness
