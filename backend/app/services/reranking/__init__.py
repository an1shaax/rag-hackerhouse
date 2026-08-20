"""Reranking service for improving retrieval quality"""
import numpy as np
from typing import List, Optional, Tuple
import structlog
from app.config import get_settings
from app.models.schemas import RetrievedChunk
import time

logger = structlog.get_logger()


class RerankerService:
    """Reranking service using cross-encoder models"""

    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self._loaded = False

    def load_model(self):
        """Load reranking model"""
        if self._loaded:
            return

        try:
            # Use multilingual cross-encoder for better Indic language support
            from sentence_transformers import CrossEncoder

            # Multilingual model that supports 15+ Indic languages
            self.model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
            self._loaded = True

            logger.info("reranker_loaded", model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

        except ImportError:
            logger.warning("sentence_transformers_not_available_using_similarity_reranking")
            self._loaded = True  # Will use similarity scores only
        except Exception as e:
            logger.error("reranker_model_load_error", error=str(e))
            self._loaded = True  # Fallback to similarity

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: Optional[int] = None
    ) -> Tuple[List[RetrievedChunk], float]:
        """
        Rerank retrieved chunks for better relevance

        Args:
            query: Original query
            chunks: Retrieved chunks to rerank
            top_k: Number of top chunks to return

        Returns:
            Tuple of (reranked chunks, latency_ms)
        """
        if not chunks:
            return [], 0.0

        start_time = time.time()

        top_k = top_k or self.settings.rerank_top_k

        if not self._loaded:
            self.load_model()

        # If no model available, use similarity scores only
        if self.model is None:
            sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)[:top_k]
            latency_ms = (time.time() - start_time) * 1000
            return sorted_chunks, latency_ms

        try:
            # Prepare pairs for cross-encoder
            pairs = [[query, c.chunk.text] for c in chunks]

            # Get scores from cross-encoder
            scores = self.model.predict(pairs)

            # Combine with original chunks and sort by cross-encoder score
            reranked = []
            for i, score in enumerate(scores):
                reranked.append(RetrievedChunk(
                    chunk=chunks[i].chunk,
                    score=float(score)
                ))

            # Sort by cross-encoder score (higher is better)
            reranked.sort(key=lambda x: x.score, reverse=True)
            reranked = reranked[:top_k]

            latency_ms = (time.time() - start_time) * 1000

            logger.debug(
                "reranking_complete",
                num_input=len(chunks),
                num_output=len(reranked),
                latency_ms=latency_ms
            )

            return reranked, latency_ms

        except Exception as e:
            logger.error("reranking_error", error=str(e))
            # Fallback to similarity scores
            sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)[:top_k]
            latency_ms = (time.time() - start_time) * 1000
            return sorted_chunks, latency_ms


# Singleton instance
_reranker_service: Optional[RerankerService] = None


def get_reranker_service() -> RerankerService:
    """Get reranker service singleton"""
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service