"""Tests for API endpoints"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.models.schemas import QueryResponse, TranscribeResponse, LatencyBreakdown


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check(self, client):
        """Test health endpoint returns valid response"""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data


class TestMetricsEndpoint:
    """Test metrics endpoint"""

    def test_metrics(self, client):
        """Test metrics endpoint returns valid response"""
        response = client.get("/api/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "total_queries" in data
        assert "avg_latency_ms" in data


class TestQueryEndpoint:
    """Test query endpoint"""

    @patch("app.main.get_harness")
    def test_query_success(self, mock_harness, client):
        """Test successful query processing"""
        # Mock harness
        mock_instance = MagicMock()
        mock_instance.is_ready.return_value = True
        mock_instance.process_query.return_value = QueryResponse(
            request_id="test-123",
            query="What is the capital of India?",
            language="en",
            answer="New Delhi is the capital of India.",
            grounded=True,
            confidence=0.95,
            citations=[],
            latency=LatencyBreakdown(
                query_embedding_ms=10.0,
                retrieval_ms=20.0,
                reranking_ms=5.0,
                generation_ms=100.0,
                grounding_ms=5.0,
                guardrails_ms=5.0,
                total_rag_ms=145.0,
                total_ms=145.0
            )
        )
        mock_harness.return_value = mock_instance

        response = client.post(
            "/api/query",
            json={"query": "What is the capital of India?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "New Delhi is the capital of India."
        assert data["grounded"] is True

    def test_query_empty(self, client):
        """Test query with empty input"""
        response = client.post(
            "/api/query",
            json={"query": ""}
        )

        assert response.status_code == 422  # Validation error


class TestTranscribeEndpoint:
    """Test transcription endpoint"""

    @patch("app.services.stt.get_stt_service")
    def test_transcribe_mock(self, mock_stt, client):
        """Test transcription with mock"""
        mock_instance = MagicMock()
        mock_instance.transcribe.return_value = TranscribeResponse(
            request_id="test-456",
            transcription="What is the capital of India?",
            language="en",
            confidence=0.95,
            latency_ms=100.0
        )
        mock_stt.return_value = mock_instance

        # Create a test audio file
        import io
        audio_content = b"fake audio content"

        response = client.post(
            "/api/transcribe",
            files={"audio": ("test.wav", io.BytesIO(audio_content), "audio/wav")},
            params={"language": "en"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "transcription" in data


class TestRootEndpoint:
    """Test root endpoint"""

    def test_root(self, client):
        """Test root endpoint"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Voice-Enabled RAG System"