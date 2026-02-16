"""FastAPI dependencies for dependency injection."""

from functools import lru_cache
from typing import Optional

from ..config import get_settings
from ..services import SearchService, QueryParser, IngestService
from ..vector_store import QdrantVectorStore


@lru_cache
def get_vector_store() -> QdrantVectorStore:
    """Get cached vector store instance."""
    settings = get_settings()
    return QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection,
    )


@lru_cache
def get_search_service() -> SearchService:
    """Get cached search service instance."""
    vector_store = get_vector_store()
    return SearchService(vector_store=vector_store)


@lru_cache
def get_ingest_service() -> IngestService:
    """Get cached ingest service instance."""
    vector_store = get_vector_store()
    return IngestService(vector_store=vector_store)


def get_query_parser() -> Optional[QueryParser]:
    """
    Get query parser instance.

    Returns None if OpenAI API key is not configured.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    return QueryParser()


def warmup_services() -> None:
    """
    Warmup services on startup.

    Pre-loads embedding models to avoid cold start latency.
    """
    search_service = get_search_service()
    search_service.warmup_providers()
