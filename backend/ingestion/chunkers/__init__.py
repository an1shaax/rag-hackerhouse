"""Chunking strategies for document processing"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
import uuid
import re


@dataclass
class ChunkResult:
    """Result from chunking operation"""
    chunk_id: str
    document_id: str
    text: str
    language: str
    source: str
    chunking_strategy: str
    position: int
    metadata: Dict[str, Any]


class BaseChunker(ABC):
    """Abstract base class for chunking strategies"""

    def __init__(self, chunk_size: int = 512, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap

    @abstractmethod
    def chunk(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[ChunkResult]:
        """
        Chunk text into smaller pieces

        Args:
            text: Text to chunk
            document_id: Unique document identifier
            metadata: Document metadata

        Returns:
            List of ChunkResult objects
        """
        pass

    def _create_chunk(
        self,
        text: str,
        document_id: str,
        position: int,
        metadata: Dict[str, Any]
    ) -> ChunkResult:
        """Create a chunk result"""
        return ChunkResult(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            text=text,
            language=metadata.get("language", "en"),
            source=metadata.get("source", "unknown"),
            chunking_strategy=self.strategy_name,
            position=position,
            metadata=metadata
        )

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of this chunking strategy"""
        pass


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunking with optional overlap"""

    @property
    def strategy_name(self) -> str:
        return "fixed"

    def chunk(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[ChunkResult]:
        """Chunk text into fixed-size pieces"""
        chunks = []
        position = 0

        # Split by characters
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]

            if len(chunk_text.strip()) > 0:
                chunks.append(self._create_chunk(
                    text=chunk_text.strip(),
                    document_id=document_id,
                    position=position,
                    metadata=metadata
                ))
                position += 1

            # Stop if we've processed all text
            if i + self.chunk_size >= len(text):
                break

        return chunks


class FixedSizeOverlapChunker(BaseChunker):
    """Fixed-size chunking with overlap for context continuity"""

    @property
    def strategy_name(self) -> str:
        return "fixed_overlap"

    def chunk(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[ChunkResult]:
        """Chunk text with overlap between chunks"""
        chunks = []
        position = 0
        step = self.chunk_size - self.overlap

        for i in range(0, len(text), step):
            chunk_text = text[i:i + self.chunk_size]

            if len(chunk_text.strip()) > 0:
                chunks.append(self._create_chunk(
                    text=chunk_text.strip(),
                    document_id=document_id,
                    position=position,
                    metadata=metadata
                ))
                position += 1

            if i + self.chunk_size >= len(text):
                break

        return chunks


class SentenceChunker(BaseChunker):
    """Sentence-aware chunking that respects sentence boundaries"""

    @property
    def strategy_name(self) -> str:
        return "sentence"

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be enhanced with spaCy/NLTK)
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[ChunkResult]:
        """Chunk text respecting sentence boundaries"""
        chunks = []
        sentences = self._split_sentences(text)
        position = 0

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(self._create_chunk(
                    text=chunk_text,
                    document_id=document_id,
                    position=position,
                    metadata=metadata
                ))
                position += 1

                # Start new chunk (optionally with overlap)
                if self.overlap > 0 and len(current_chunk) > 1:
                    # Keep last sentence for overlap
                    current_chunk = [current_chunk[-1]]
                    current_length = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk(
                text=chunk_text,
                document_id=document_id,
                position=position,
                metadata=metadata
            ))

        return chunks


class SemanticChunker(BaseChunker):
    """Semantic chunking based on content similarity"""

    @property
    def strategy_name(self) -> str:
        return "semantic"

    def __init__(self, chunk_size: int = 512, similarity_threshold: float = 0.5):
        super().__init__(chunk_size)
        self.similarity_threshold = similarity_threshold

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word-overlap similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def chunk(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[ChunkResult]:
        """Chunk text based on semantic similarity between sentences"""
        # Split into sentences
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = [s.strip() for s in re.split(sentence_endings, text) if s.strip()]

        if not sentences:
            return []

        chunks = []
        position = 0
        current_chunk = [sentences[0]]
        current_length = len(sentences[0])

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_length = len(sentence)

            # Calculate similarity with current chunk
            current_text = " ".join(current_chunk)
            similarity = self._calculate_similarity(current_text, sentence)

            # Decide whether to add to current chunk or start new one
            should_add = (
                current_length + sentence_length <= self.chunk_size and
                similarity >= self.similarity_threshold
            )

            if should_add:
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(self._create_chunk(
                        text=" ".join(current_chunk),
                        document_id=document_id,
                        position=position,
                        metadata=metadata
                    ))
                    position += 1

                # Start new chunk
                current_chunk = [sentence]
                current_length = sentence_length

        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                text=" ".join(current_chunk),
                document_id=document_id,
                position=position,
                metadata=metadata
            ))

        return chunks


class MetadataAwareChunker(BaseChunker):
    """Chunking that considers document metadata and structure"""

    @property
    def strategy_name(self) -> str:
        return "metadata_aware"

    def chunk(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[ChunkResult]:
        """
        Chunk text considering metadata like language, document type, etc.

        For MSMARCO-XI:
        - Preserve query_id, query_type
        - Handle multilingual content appropriately
        - Respect passage boundaries if available
        """
        chunks = []
        position = 0

        # Check if passages are provided in metadata
        if "passages" in metadata:
            passages = metadata["passages"]
            for i, passage in enumerate(passages):
                if isinstance(passage, dict):
                    passage_text = passage.get("text", passage.get("Translated_passages", ""))
                else:
                    passage_text = str(passage)

                if passage_text and len(passage_text.strip()) > 0:
                    # Further chunk if passage is too long
                    if len(passage_text) > self.chunk_size:
                        # Use fixed-size chunking for long passages
                        for j in range(0, len(passage_text), self.chunk_size - self.overlap):
                            chunk_text = passage_text[j:j + self.chunk_size]
                            if chunk_text.strip():
                                chunk_metadata = {**metadata, "passage_position": i}
                                chunks.append(self._create_chunk(
                                    text=chunk_text.strip(),
                                    document_id=document_id,
                                    position=position,
                                    metadata=chunk_metadata
                                ))
                                position += 1
                    else:
                        chunk_metadata = {**metadata, "passage_position": i}
                        chunks.append(self._create_chunk(
                            text=passage_text.strip(),
                            document_id=document_id,
                            position=position,
                            metadata=chunk_metadata
                        ))
                        position += 1
        else:
            # Fall back to sentence-aware chunking
            sentence_chunker = SentenceChunker(self.chunk_size, self.overlap)
            return sentence_chunker.chunk(text, document_id, metadata)

        return chunks


def get_chunker(strategy: str, chunk_size: int = 512, overlap: int = 50) -> BaseChunker:
    """
    Factory function to get chunker by strategy name

    Args:
        strategy: Chunking strategy name
        chunk_size: Size of chunks
        overlap: Overlap between chunks

    Returns:
        Chunker instance
    """
    chunkers = {
        "fixed": FixedSizeChunker(chunk_size, overlap),
        "fixed_overlap": FixedSizeOverlapChunker(chunk_size, overlap),
        "sentence": SentenceChunker(chunk_size, overlap),
        "semantic": SemanticChunker(chunk_size),
        "metadata_aware": MetadataAwareChunker(chunk_size, overlap),
    }

    if strategy not in chunkers:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    return chunkers[strategy]
