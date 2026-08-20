"""Tests for guardrails and grounding"""
import pytest
from app.services.guardrails import get_guardrails_service, GuardrailsService
from app.models.schemas import GeneratedAnswer, RetrievedChunk, Chunk


@pytest.fixture
def guardrails():
    """Get guardrails service"""
    return get_guardrails_service()


class TestInputValidation:
    """Test input validation guardrails"""

    def test_empty_query(self, guardrails):
        """Test empty query is rejected"""
        is_valid, error = guardrails.validate_input("")
        assert not is_valid
        assert "empty" in error.lower() or "cannot" in error.lower()

    def test_whitespace_query(self, guardrails):
        """Test whitespace-only query is rejected"""
        is_valid, error = guardrails.validate_input("   ")
        assert not is_valid

    def test_long_query(self, guardrails):
        """Test excessively long query is rejected"""
        long_query = "x" * 3000
        is_valid, error = guardrails.validate_input(long_query)
        assert not is_valid
        assert "long" in error.lower()

    def test_normal_query(self, guardrails):
        """Test normal query passes validation"""
        is_valid, error = guardrails.validate_input("What is machine learning?")
        assert is_valid
        assert error is None

    def test_hindi_query(self, guardrails):
        """Test Hindi query passes validation"""
        is_valid, error = guardrails.validate_input("भारत की राजधानी क्या है?")
        assert is_valid

    def test_bengali_query(self, guardrails):
        """Test Bengali query passes validation"""
        is_valid, error = guardrails.validate_input("ভারতের রাজধানী কী?")
        assert is_valid


class TestRetrievalQuality:
    """Test retrieval quality checking"""

    def test_no_chunks(self, guardrails):
        """Test that no chunks is insufficient"""
        is_sufficient, reason = guardrails.check_retrieval_quality([], "test query")
        assert not is_sufficient
        assert "no relevant" in reason.lower() or "not found" in reason.lower()

    def test_low_scores(self, guardrails):
        """Test that low scores are insufficient"""
        chunk = Chunk(
            chunk_id="c1",
            document_id="d1",
            text="test text",
            language="en",
            source="test",
            chunking_strategy="fixed",
            position=0,
        )
        retrieved = [RetrievedChunk(chunk=chunk, score=0.01)]
        is_sufficient, _ = guardrails.check_retrieval_quality(retrieved, "test")
        assert not is_sufficient

    def test_high_scores(self, guardrails):
        """Test that high scores are sufficient"""
        chunk = Chunk(
            chunk_id="c1",
            document_id="d1",
            text="Capital of India is New Delhi",
            language="en",
            source="MSMARCO-XI",
            chunking_strategy="fixed",
            position=0,
        )
        retrieved = [RetrievedChunk(chunk=chunk, score=0.9)]
        is_sufficient, _ = guardrails.check_retrieval_quality(retrieved, "capital of india")
        assert is_sufficient


class TestGroundingVerification:
    """Test grounding verification"""

    def test_ungrounded_answer(self, guardrails):
        """Test that LLM-flagged ungrounded answer is rejected"""
        answer = GeneratedAnswer(
            answer="Some random text",
            grounded=False,
            confidence=0.5,
            citations=[]
        )
        chunk = Chunk(
            chunk_id="c1",
            document_id="d1",
            text="test",
            language="en",
            source="test",
            chunking_strategy="fixed",
            position=0,
        )
        is_grounded, conf = guardrails.verify_grounding(answer, [RetrievedChunk(chunk=chunk, score=0.8)], "test query")
        assert not is_grounded

    def test_insufficient_info_answer(self, guardrails):
        """Test that insufficient info answer is rejected"""
        answer = GeneratedAnswer(
            answer="I don't have enough information in the provided dataset to answer that.",
            grounded=True,
            confidence=0.5,
            citations=[]
        )
        is_grounded, conf = guardrails.verify_grounding(answer, [], "test query")
        assert not is_grounded

    def test_grounded_answer(self, guardrails):
        """Test that well-grounded answer passes"""
        answer = GeneratedAnswer(
            answer="The capital of India is New Delhi.",
            grounded=True,
            confidence=0.9,
            citations=["c1"]
        )
        chunk = Chunk(
            chunk_id="c1",
            document_id="d1",
            text="The capital of India is New Delhi. It was declared the capital in 1911.",
            language="en",
            source="MSMARCO-XI",
            chunking_strategy="fixed",
            position=0,
        )
        is_grounded, conf = guardrails.verify_grounding(answer, [RetrievedChunk(chunk=chunk, score=0.9)], "capital of india")
        assert is_grounded


class TestOutputSafety:
    """Test output safety checking"""

    def test_safe_output(self, guardrails):
        """Test that safe output passes"""
        is_safe, _ = guardrails.check_output_safety("The capital of India is New Delhi.")
        assert is_safe

    def test_pii_in_output(self, guardrails):
        """Test that PII in output is caught"""
        is_safe, _ = guardrails.check_output_safety("My SSN is 123-45-6789")
        assert not is_safe
