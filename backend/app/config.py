"""Configuration management using Pydantic settings"""
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings from environment variables"""

    # API Keys
    sarvam_api_key: Optional[str] = None
    llm_api_key: Optional[str] = None

    # LLM Configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

    # Embedding Model
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Retrieval Configuration
    top_k: int = 50
    rerank_top_k: int = 5
    relevance_threshold: float = 0.3

    # Chunking Configuration
    default_chunk_strategy: str = "semantic"
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Benchmarking
    enable_benchmarking: bool = True

    # Paths
    data_dir: str = "./data"
    index_dir: str = "./indexes"
    reports_dir: str = "./reports"

    # API URLs
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"

    # Mock Mode (for development without API keys)
    mock_stt: bool = False
    mock_llm: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
