"""Speech-to-Text service using Sarvam AI"""
import httpx
import structlog
from typing import Optional, Dict, Any
from app.config import get_settings
from app.models.schemas import TranscribeResponse
import uuid
import time

logger = structlog.get_logger()


class STTService:
    """Speech-to-text service with Sarvam integration"""

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.sarvam_api_key
        self.stt_url = self.settings.sarvam_stt_url
        self.mock_mode = self.settings.mock_stt or not self.api_key

        if self.mock_mode:
            logger.info("STT service running in MOCK mode (no API key)")

    async def transcribe(
        self,
        audio_data: bytes,
        language: str = "en",
        audio_format: str = "wav"
    ) -> TranscribeResponse:
        """
        Transcribe audio to text using Sarvam STT

        Args:
            audio_data: Raw audio bytes
            language: Language code (en, hi, bn, ta, etc.)
            audio_format: Audio format (wav, mp3, etc.)

        Returns:
            TranscribeResponse with transcription and metadata
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        if self.mock_mode:
            # Mock transcription for development
            return self._mock_transcribe(request_id, language, start_time)

        try:
            # Map language codes to Sarvam format
            sarvam_lang = self._map_language(language)

            headers = {
                "api-subscription-key": self.api_key,
            }

            files = {
                "file": (f"audio.{audio_format}", audio_data, f"audio/{audio_format}")
            }

            data = {
                "language_code": sarvam_lang,
                "model": "saarika:v1",
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.stt_url,
                    headers=headers,
                    files=files,
                    data=data
                )
                response.raise_for_status()

            result = response.json()
            transcription = result.get("transcript", "")
            confidence = result.get("confidence", 0.9)

            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                "stt_transcription_complete",
                request_id=request_id,
                language=language,
                latency_ms=latency_ms,
                transcription_length=len(transcription)
            )

            return TranscribeResponse(
                request_id=request_id,
                transcription=transcription,
                language=language,
                confidence=confidence,
                latency_ms=latency_ms
            )

        except httpx.HTTPError as e:
            logger.error(
                "stt_api_error",
                request_id=request_id,
                error=str(e)
            )
            # Fallback to mock on error
            return self._mock_transcribe(request_id, language, start_time)

        except Exception as e:
            logger.error(
                "stt_unexpected_error",
                request_id=request_id,
                error=str(e)
            )
            return self._mock_transcribe(request_id, language, start_time)

    def _mock_transcribe(
        self,
        request_id: str,
        language: str,
        start_time: float
    ) -> TranscribeResponse:
        """Generate mock transcription for development"""
        mock_transcriptions = {
            "en": "What is the capital of India?",
            "hi": "भारत की राजधानी क्या है?",
            "bn": "ভারতের রাজধানী কী?",
            "ta": "இந்தியாவின் தலைநகரம் என்ன?",
            "te": "భారతదేశం యొక్క రాజధాని ఏమిటి?",
        }

        latency_ms = (time.time() - start_time) * 1000

        return TranscribeResponse(
            request_id=request_id,
            transcription=mock_transcriptions.get(language, mock_transcriptions["en"]),
            language=language,
            confidence=0.95,
            latency_ms=latency_ms
        )

    def _map_language(self, lang_code: str) -> str:
        """Map language codes to Sarvam format"""
        mapping = {
            "en": "en-IN",
            "hi": "hi-IN",
            "bn": "bn-IN",
            "ta": "ta-IN",
            "te": "te-IN",
            "mr": "mr-IN",
            "gu": "gu-IN",
            "kn": "kn-IN",
            "ml": "ml-IN",
            "pa": "pa-IN",
            "or": "od-IN",
            "as": "as-IN",
            "ur": "ur-IN",
        }
        return mapping.get(lang_code, "en-IN")


# Singleton instance
_stt_service: Optional[STTService] = None


def get_stt_service() -> STTService:
    """Get STT service singleton"""
    global _stt_service
    if _stt_service is None:
        _stt_service = STTService()
    return _stt_service
