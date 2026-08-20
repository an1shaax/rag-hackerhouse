"""Embedding service for multilingual text embeddings"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import structlog
from app.config import get_settings
import time

logger = structlog.get_logger()


class EmbeddingService:
    """Multilingual embedding service using sentence-transformers"""

    def __init__(self):
        self.settings = get_settings()
        self.model_name = self.settings.embedding_model
        self.model: Optional[SentenceTransformer] = None
        self._loaded = False

    def load_model(self):
        """Load embedding model (call once at startup)"""
        if self._loaded:
            return

        logger.info("loading_embedding_model", model=self.model_name)
        start_time = time.time()

        self.model = SentenceTransformer(self.model_name)
        self._loaded = True

        load_time = time.time() - start_time
        logger.info(
            "embedding_model_loaded",
            model=self.model_name,
            load_time_seconds=load_time
        )

    def embed(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for texts

        Args:
            texts: List of texts to embed
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        if not self._loaded:
            self.load_model()

        start_time = time.time()

        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )

        latency_ms = (time.time() - start_time) * 1000

        logger.debug(
            "embeddings_generated",
            num_texts=len(texts),
            latency_ms=latency_ms
        )

        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            Numpy array of embedding (embedding_dim,)
        """
        if not self._loaded:
            self.load_model()

        start_time = time.time()

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        latency_ms = (time.time() - start_time) * 1000

        logger.debug(
            "single_embedding_generated",
            text_length=len(text),
            latency_ms=latency_ms
        )

        return embedding

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension"""
        if not self._loaded:
            self.load_model()
        return self.model.get_sentence_embedding_dimension()

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get embedding service singleton"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
