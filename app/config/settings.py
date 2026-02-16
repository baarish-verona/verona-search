"""Application settings using Pydantic Settings."""

import logging
import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=(".env", f".env.{os.getenv('APP_ENV', 'development')}"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Environment
    app_env: str = "development"  # development, staging, production

    # Application
    app_name: str = "Verona AI Search"
    debug: bool = False

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "matrimonial_profiles"

    # OpenAI (for query parsing and embeddings)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    # Embedding
    embedding_device: str = "cpu"  # cpu, cuda, mps
    colbert_batch_size: int = 20

    # Search defaults
    default_search_limit: int = 100
    max_search_limit: int = 200
    prefetch_limit: int = 1000
    score_threshold: float = 0.0

    # CloudFront (for photo URLs)
    cloud_front_url: str = "https://d34thlcszyehjn.cloudfront.net"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()

    # Log all settings on first load
    logger.info("=" * 50)
    logger.info("CONFIGURATION LOADED")
    logger.info("=" * 50)
    logger.info(f"APP_ENV: {settings.app_env}")
    logger.info(f"APP_NAME: {settings.app_name}")
    logger.info(f"DEBUG: {settings.debug}")
    logger.info(f"QDRANT_HOST: {settings.qdrant_host}")
    logger.info(f"QDRANT_PORT: {settings.qdrant_port}")
    logger.info(f"QDRANT_COLLECTION: {settings.qdrant_collection}")
    logger.info(f"OPENAI_API_KEY: {'***' + settings.openai_api_key[-4:] if settings.openai_api_key else 'NOT SET'}")
    logger.info(f"OPENAI_MODEL: {settings.openai_model}")
    logger.info(f"EMBEDDING_DEVICE: {settings.embedding_device}")
    logger.info(f"COLBERT_BATCH_SIZE: {settings.colbert_batch_size}")
    logger.info(f"DEFAULT_SEARCH_LIMIT: {settings.default_search_limit}")
    logger.info(f"MAX_SEARCH_LIMIT: {settings.max_search_limit}")
    logger.info(f"PREFETCH_LIMIT: {settings.prefetch_limit}")
    logger.info(f"SCORE_THRESHOLD: {settings.score_threshold}")
    logger.info(f"CLOUD_FRONT_URL: {settings.cloud_front_url}")
    logger.info("=" * 50)

    return settings
