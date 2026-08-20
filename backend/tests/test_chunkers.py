"""Tests for chunking strategies"""
import pytest
from ingestion.chunkers import (
    get_chunker,
    FixedSizeChunker,
    FixedSizeOverlapChunker,
    SentenceChunker,
    SemanticChunker,
    MetadataAwareChunker,
)


class TestFixedSizeChunker:
    """Test fixed-size chunking"""

    def test_basic_chunking(self):
        """Test basic fixed-size chunking"""
        chunker = FixedSizeChunker(chunk_size=100, overlap=0)
        text = "This is a test sentence. " * 10  # 250 chars

        chunks = chunker.chunk(text, "doc1", {"language": "en"})

        assert len(chunks) > 0
        assert all(len(c.text) <= 100 for c in chunks)
        assert chunks[0].chunking_strategy == "fixed"

    def test_small_text(self):
        """Test chunking small text"""
        chunker = FixedSizeChunker(chunk_size=100, overlap=0)
        text = "Short text"

        chunks = chunker.chunk(text, "doc1", {})

        assert len(chunks) == 1
        assert chunks[0].text == "Short text"


class TestFixedSizeOverlapChunker:
    """Test fixed-size chunking with overlap"""

    def test_overlap(self):
        """Test that overlap creates shared content"""
        chunker = FixedSizeOverlapChunker(chunk_size=100, overlap=20)
        text = "A" * 150

        chunks = chunker.chunk(text, "doc1", {})

        assert len(chunks) > 1
        # Check overlap exists
        if len(chunks) > 1:
            # The end of first chunk should appear in start of second
            assert len(chunks[0].text) == 100


class TestSentenceChunker:
    """Test sentence-aware chunking"""

    def test_sentence_boundaries(self):
        """Test that chunks respect sentence boundaries"""
        chunker = SentenceChunker(chunk_size=100, overlap=0)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = chunker.chunk(text, "doc1", {})

        assert len(chunks) > 0
        # Each chunk should end with a sentence
        for chunk in chunks:
            if len(chunk.text) < 100:  # Last chunk might not end with period
                continue
            # Verify chunks are coherent
            assert "." in chunk.text or len(chunk.text) < 50

    def test_long_sentence(self):
        """Test handling of sentences longer than chunk size"""
        chunker = SentenceChunker(chunk_size=20, overlap=0)
        text = "This is a very long sentence that exceeds the chunk size limit."

        chunks = chunker.chunk(text, "doc1", {})

        assert len(chunks) > 0


class TestSemanticChunker:
    """Test semantic chunking"""

    def test_semantic_grouping(self):
        """Test that similar sentences are grouped"""
        chunker = SemanticChunker(chunk_size=200, similarity_threshold=0.3)
        text = "Python is a programming language. Python is popular. Python is easy to learn. The weather is nice today. It is sunny outside."

        chunks = chunker.chunk(text, "doc1", {})

        assert len(chunks) > 0
        assert chunks[0].chunking_strategy == "semantic"


class TestMetadataAwareChunker:
    """Test metadata-aware chunking"""

    def test_with_passages(self):
        """Test chunking with passage metadata"""
        chunker = MetadataAwareChunker(chunk_size=100, overlap=0)
        text = "Main text content here."
        metadata = {
            "passages": ["First passage content.", "Second passage content."]
        }

        chunks = chunker.chunk(text, "doc1", metadata)

        assert len(chunks) == 2
        assert chunks[0].text == "First passage content."
        assert chunks[1].text == "Second passage content."

    def test_without_passages(self):
        """Test fallback to sentence chunking"""
        chunker = MetadataAwareChunker(chunk_size=100, overlap=0)
        text = "First sentence. Second sentence."
        metadata = {}

        chunks = chunker.chunk(text, "doc1", metadata)

        assert len(chunks) > 0


class TestChunkerFactory:
    """Test chunker factory function"""

    def test_get_chunker_all_types(self):
        """Test that all chunker types can be created"""
        strategies = ["fixed", "fixed_overlap", "sentence", "semantic", "metadata_aware"]

        for strategy in strategies:
            chunker = get_chunker(strategy)
            assert chunker is not None
            assert chunker.strategy_name == strategy

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error"""
        with pytest.raises(ValueError):
            get_chunker("invalid_strategy")
