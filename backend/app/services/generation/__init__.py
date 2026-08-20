"""LLM-based answer generation service"""
import os
from typing import Optional, Dict, Any, List
import structlog
from app.config import get_settings
from app.models.schemas import GeneratedAnswer, RetrievedChunk
import time
import json
import re

logger = structlog.get_logger()


class GenerationService:
    """LLM-based answer generation with structured output"""

    SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based ONLY on the provided context.

CRITICAL RULES:
1. Answer ONLY using information from the provided context.
2. If the context does not contain enough information, respond with: {"answer": "I don't have enough information in the provided dataset to answer that.", "grounded": false, "confidence": 0.0, "citations": []}
3. Cite your sources by including chunk IDs in the citations list.
4. Be accurate and concise.
5. Respond in JSON format only.

Response format (JSON):
{
  "answer": "Your answer here",
  "grounded": true,
  "confidence": 0.0-1.0,
  "citations": ["chunk_id_1", "chunk_id_2"]
}"""

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.llm_api_key or os.getenv("OPENAI_API_KEY")
        self.provider = self.settings.llm_provider
        self.model = self.settings.llm_model
        self.mock_mode = self.settings.mock_llm or not self.api_key

        if self.mock_mode:
            logger.info("Generation service running in MOCK mode (no API key)")

    def generate(
        self,
        query: str,
        context_chunks: List[RetrievedChunk],
        language: str = "en"
    ) -> tuple[GeneratedAnswer, float]:
        """
        Generate answer from query and context

        Args:
            query: User question
            context_chunks: Retrieved context chunks
            language: Response language

        Returns:
            Tuple of (GeneratedAnswer, latency_ms)
        """
        start_time = time.time()

        if self.mock_mode:
            return self._mock_generate(query, context_chunks, start_time)

        try:
            # Build context string
            context = self._build_context(context_chunks)

            prompt = f"""Context:
{context}

Question: {query}

Instructions:
1. Answer the question using ONLY the provided context
2. If the context is insufficient, set grounded to false
3. Include relevant chunk IDs in citations
4. Respond in JSON format only

JSON Response:"""

            if self.provider == "openai":
                answer = self._call_openai(prompt)
            elif self.provider == "anthropic":
                answer = self._call_anthropic(prompt)
            else:
                logger.warning("Unknown provider, using mock", provider=self.provider)
                return self._mock_generate(query, context_chunks, start_time)

            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                "generation_complete",
                grounded=answer.grounded,
                confidence=answer.confidence,
                latency_ms=latency_ms
            )

            return answer, latency_ms

        except Exception as e:
            logger.error("generation_error", error=str(e))
            return self._mock_generate(query, context_chunks, start_time)

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        """Build context string from chunks"""
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[{chunk.chunk.chunk_id}] {chunk.chunk.text}"
            )
        return "\n\n".join(context_parts)

    def _call_openai(self, prompt: str) -> GeneratedAnswer:
        """Call OpenAI API"""
        import openai

        client = openai.OpenAI(api_key=self.api_key)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        return self._parse_response(content)

    def _call_anthropic(self, prompt: str) -> GeneratedAnswer:
        """Call Anthropic API"""
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)

        response = client.messages.create(
            model=self.model or "claude-sonnet-5-20250514",
            max_tokens=1000,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text
        return self._parse_response(content)

    def _parse_response(self, content: str) -> GeneratedAnswer:
        """Parse LLM response into structured format"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                return GeneratedAnswer(
                    answer=data.get("answer", ""),
                    grounded=data.get("grounded", True),
                    confidence=float(data.get("confidence", 0.8)),
                    citations=data.get("citations", []),
                    raw_response=content
                )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse JSON response", error=str(e))

        # Fallback: return raw content as answer
        return GeneratedAnswer(
            answer=content,
            grounded=True,
            confidence=0.7,
            citations=[],
            raw_response=content
        )

    def _mock_generate(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        start_time: float
    ) -> tuple[GeneratedAnswer, float]:
        """Generate mock answer for development - extracts info from retrieved chunks"""
        latency_ms = (time.time() - start_time) * 1000

        # Check if top chunk has very low relevance score (indicating no relevant context)
        top_score = chunks[0].score if chunks else 0.0

        if not chunks or top_score <= 0.1:
            answer = "I don't have enough information in the provided dataset to answer that."
            citations = []
            grounded = False
            confidence = 0.0
            return GeneratedAnswer(
                answer=answer,
                grounded=grounded,
                confidence=confidence,
                citations=citations
            ), latency_ms

        # Extract relevant information from chunks for a better mock answer
        query_lower = query.lower()
        query_keywords = set(query_lower.split())

        # Find chunks that contain query keywords
        relevant_texts = []
        for chunk in chunks[:3]:
            text = chunk.chunk.text
            text_lower = text.lower()
            # Check if any query keyword appears in the chunk
            if any(kw in text_lower for kw in query_keywords if len(kw) > 3):
                relevant_texts.append(text)
            # Also include chunks with high scores
            elif chunk.score > 0.5:
                relevant_texts.append(text)

        # Build answer from relevant texts
        if relevant_texts:
            # Combine unique sentences from relevant texts
            sentences = []
            for text in relevant_texts:
                for sent in text.split('.'):
                    sent = sent.strip()
                    if sent and len(sent) > 10:
                        sentences.append(sent)

            # Deduplicate and take first few
            unique_sentences = list(dict.fromkeys(sentences))
            answer = '. '.join(unique_sentences[:3]) + '.'
            if not answer.endswith('.'):
                answer += '.'
        else:
            answer = "I don't have enough information in the provided dataset to answer that."
            grounded = False
            confidence = 0.0
            citations = []
            return GeneratedAnswer(
                answer=answer,
                grounded=grounded,
                confidence=confidence,
                citations=citations
            ), latency_ms

        citations = [c.chunk.chunk_id for c in chunks[:3]]
        grounded = True
        confidence = min(0.95, max(0.6, top_score))

        return GeneratedAnswer(
            answer=answer,
            grounded=grounded,
            confidence=confidence,
            citations=citations
        ), latency_ms


# Singleton instance
_generation_service: Optional[GenerationService] = None


def get_generation_service() -> GenerationService:
    """Get generation service singleton"""
    global _generation_service
    if _generation_service is None:
        _generation_service = GenerationService()
    return _generation_service
