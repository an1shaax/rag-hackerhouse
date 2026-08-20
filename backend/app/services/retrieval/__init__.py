"""FAISS-based vector retrieval service"""
import faiss
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import structlog
from app.config import get_settings
from app.models.schemas import Chunk, RetrievedChunk
import time
import uuid

logger = structlog.get_logger()


class RetrievalService:
    """FAISS-based vector retrieval with metadata mapping"""

    def __init__(self):
        self.settings = get_settings()
        self.index_dir = Path(self.settings.index_dir)

        self.index: Optional[faiss.Index] = None
        self.chunk_metadata: Dict[int, Dict[str, Any]] = {}
        self.id_to_chunk_id: Dict[int, str] = {}
        self._loaded = False

    def load_index(self) -> bool:
        """
        Load FAISS index and metadata from disk

        Returns:
            True if loaded successfully, False otherwise
        """
        if self._loaded:
            return True

        index_path = self.index_dir / "faiss_index.bin"
        metadata_path = self.index_dir / "chunk_metadata.pkl"
        mapping_path = self.index_dir / "id_mapping.json"

        if not index_path.exists():
            logger.warning("index_not_found", path=str(index_path))
            return False

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))

            # Load metadata
            with open(metadata_path, "rb") as f:
                self.chunk_metadata = pickle.load(f)

            # Load ID mapping
            with open(mapping_path, "r") as f:
                self.id_to_chunk_id = json.load(f)

            self._loaded = True

            logger.info(
                "index_loaded",
                num_vectors=self.index.ntotal,
                dimension=self.index.d
            )

            return True

        except Exception as e:
            logger.error("index_load_error", error=str(e))
            return False

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        language: Optional[str] = None
    ) -> Tuple[List[RetrievedChunk], float]:
        """
        Search for similar chunks

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            language: Optional language code to filter results (e.g., 'en', 'hi', 'bn')

        Returns:
            Tuple of (list of retrieved chunks, latency_ms)
        """
        if not self._loaded:
            if not self.load_index():
                logger.error("index_not_loaded")
                return [], 0.0

        start_time = time.time()

        top_k = top_k or self.settings.top_k

        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search with extra candidates to allow for language filtering
        # If language filter is specified, search more candidates to compensate for filtering
        search_k = top_k * 3 if language else top_k
        search_k = min(search_k, self.index.ntotal)

        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)

        latency_ms = (time.time() - start_time) * 1000

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            if idx not in self.chunk_metadata:
                logger.warning("missing_metadata", idx=idx)
                continue

            metadata = self.chunk_metadata[idx]

            # Language filter
            chunk_lang = metadata.get("language", "en")
            if language and chunk_lang != language:
                continue

            chunk = Chunk(
                chunk_id=metadata.get("chunk_id", str(uuid.uuid4())),
                document_id=metadata.get("document_id", ""),
                text=metadata.get("text", ""),
                language=chunk_lang,
                source=metadata.get("source", ""),
                chunking_strategy=metadata.get("chunking_strategy", "unknown"),
                position=metadata.get("position", 0),
                metadata=metadata.get("metadata", {})
            )

            results.append(RetrievedChunk(chunk=chunk, score=float(score)))

            if len(results) >= top_k:
                break

        logger.debug(
            "retrieval_complete",
            num_results=len(results),
            latency_ms=latency_ms,
            language_filter=language,
            top_scores=[r.score for r in results[:3]]
        )

        return results, latency_ms

    def get_num_vectors(self) -> int:
        """Get number of vectors in index"""
        if not self._loaded:
            self.load_index()
        return self.index.ntotal if self.index else 0

    def is_loaded(self) -> bool:
        """Check if index is loaded"""
        return self._loaded


# Singleton instance
_retrieval_service: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    """Get retrieval service singleton"""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
