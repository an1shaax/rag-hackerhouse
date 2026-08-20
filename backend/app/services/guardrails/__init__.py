"""Guardrails service for input/output validation and safety"""
from typing import Optional, List, Tuple
import structlog
from app.config import get_settings
from app.models.schemas import RetrievedChunk, GeneratedAnswer
import time
import re

logger = structlog.get_logger()


class GuardrailsService:
    """Guardrails for input validation, safety, and grounding verification"""

    # Patterns that might indicate unsafe or off-topic queries
    UNSAFE_PATTERNS = [
        r'\b(password|credit card|ssn|social security)\b',
        r'\b(hack|exploit|vulnerability)\b',
        r'\b(illegal|drug|weapon)\b',
    ]

    # Maximum input length
    MAX_QUERY_LENGTH = 2000

    def __init__(self):
        self.settings = get_settings()
        self.relevance_threshold = self.settings.relevance_threshold

    def validate_input(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate user input

        Args:
            query: User query

        Returns:
            Tuple of (is_valid, error_message)
        """
        start_time = time.time()

        # Check for empty input
        if not query or not query.strip():
            logger.warning("empty_query")
            return False, "Query cannot be empty."

        # Check for excessively long input
        if len(query) > self.MAX_QUERY_LENGTH:
            logger.warning(
                "query_too_long",
                length=len(query),
                max_length=self.MAX_QUERY_LENGTH
            )
            return False, f"Query too long. Maximum {self.MAX_QUERY_LENGTH} characters allowed."

        # Check for potentially unsafe content
        for pattern in self.UNSAFE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning("unsafe_query_pattern", pattern=pattern)
                return False, "This query contains potentially unsafe content."

        latency_ms = (time.time() - start_time) * 1000
        logger.debug("input_validated", latency_ms=latency_ms)

        return True, None

    def check_retrieval_quality(
        self,
        chunks: List[RetrievedChunk],
        query: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if retrieved context is sufficient

        Args:
            chunks: Retrieved chunks
            query: User query

        Returns:
            Tuple of (is_sufficient, reason)
        """
        if not chunks:
            logger.info("no_chunks_retrieved")
            return False, "No relevant documents found in the dataset."

        # Check if top chunks meet relevance threshold
        top_score = chunks[0].score if chunks else 0.0

        if top_score < self.relevance_threshold:
            logger.info(
                "low_relevance_scores",
                top_score=top_score,
                threshold=self.relevance_threshold
            )
            return False, f"Retrieved documents have low relevance (score: {top_score:.3f})."

        return True, None

    def verify_grounding(
        self,
        answer: GeneratedAnswer,
        chunks: List[RetrievedChunk],
        query: str
    ) -> Tuple[bool, float]:
        """
        Verify that the answer is grounded in the context AND relevant to the query

        Args:
            answer: Generated answer
            chunks: Retrieved context chunks
            query: Original user query

        Returns:
            Tuple of (is_grounded, confidence)
        """
        start_time = time.time()

        # If LLM already flagged as ungrounded, respect that
        if not answer.grounded:
            latency_ms = (time.time() - start_time) * 1000
            logger.info("answer_flagged_ungrounded_by_llm")
            return False, 0.0

        # Check if answer claims insufficient information
        if "don't have enough information" in answer.answer.lower():
            latency_ms = (time.time() - start_time) * 1000
            logger.info("answer_claims_insufficient_info")
            return False, 0.0

        # Verify citations exist in chunks
        chunk_ids = {c.chunk.chunk_id for c in chunks}
        cited_ids = set(answer.citations)

        if cited_ids and not cited_ids.issubset(chunk_ids):
            logger.warning(
                "invalid_citations",
                cited=cited_ids,
                available=chunk_ids
            )
            # Filter to valid citations
            answer.citations = list(cited_ids.intersection(chunk_ids))

        # Simple text overlap check for grounding (answer words in context)
        answer_words = set(answer.answer.lower().split())
        context_words = set()
        for chunk in chunks:
            context_words.update(chunk.chunk.text.lower().split())

        overlap = answer_words.intersection(context_words)
        overlap_ratio = len(overlap) / len(answer_words) if answer_words else 0.0

        # Grounded if:
        # 1. At least 30% of answer words appear in context (grounding)
        # 2. Answer has reasonable confidence (>= 0.5)
        # Note: We don't do keyword-based query-answer relevance check because
        # multilingual queries may get answers in different scripts/languages.
        # The reranker + retrieval quality check should ensure relevance.
        is_grounded = (
            overlap_ratio >= 0.3 and
            answer.confidence >= 0.5
        )

        latency_ms = (time.time() - start_time) * 1000

        logger.debug(
            "grounding_check_complete",
            is_grounded=is_grounded,
            overlap_ratio=overlap_ratio,
            confidence=answer.confidence,
            latency_ms=latency_ms
        )

        return is_grounded, answer.confidence

    def check_output_safety(self, answer: str) -> Tuple[bool, Optional[str]]:
        """
        Check output for safety concerns

        Args:
            answer: Generated answer

        Returns:
            Tuple of (is_safe, error_message)
        """
        # Check for potential PII leakage patterns
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b[A-Z]{2}\d{6}\b',  # Passport number pattern
        ]

        for pattern in pii_patterns:
            if re.search(pattern, answer):
                logger.warning("potential_pii_in_output")
                return False, "Response contains potentially sensitive information."

        return True, None


# Singleton instance
_guardrails_service: Optional[GuardrailsService] = None


def get_guardrails_service() -> GuardrailsService:
    """Get guardrails service singleton"""
    global _guardrails_service
    if _guardrails_service is None:
        _guardrails_service = GuardrailsService()
    return _guardrails_service
