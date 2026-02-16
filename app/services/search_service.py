"""Search service orchestrating the search flow with OpenAI + ColBERT."""

import logging
import time
from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..config.embedding_specs import VECTOR_CONFIG, get_required_providers
from ..embeddings import EmbeddingProviderFactory
from ..mappers import QueryMapper
from ..vector_store import QdrantVectorStore
from .filter_analysis import FilterAnalysisService

logger = logging.getLogger(__name__)


class SearchService:
    """
    Search service using OpenAI embeddings + ColBERT.

    Orchestrates:
    - Query parsing and validation
    - Embedding generation for semantic queries
    - Search execution with dynamic query building
    - Filter impact analysis
    - Result formatting
    """

    def __init__(self, vector_store: QdrantVectorStore):
        """
        Initialize search service.

        Args:
            vector_store: Qdrant vector store instance
        """
        self.vector_store = vector_store
        self.filter_analysis_service = FilterAnalysisService(vector_store)
        settings = get_settings()
        self.device = settings.embedding_device

    def search(
        self,
        parsed_queries: Optional[Dict[str, str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
        score_threshold: float = 0.0,
        include_filter_analysis: bool = True,
        skip_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute search with parsed queries and filters.

        Args:
            parsed_queries: Dict with education_query, profession_query, etc.
            filters: Filter dict
            limit: Max results
            offset: Result offset
            score_threshold: Minimum score threshold
            include_filter_analysis: Whether to include filter impact analysis
            skip_ids: Profile IDs to exclude from results

        Returns:
            Search results with metadata and filter analysis
        """
        start_time = time.time()

        # Extract semantic queries
        semantic_queries = QueryMapper.extract_semantic_queries(parsed_queries or {})

        # Normalize filters
        normalized_filters = QueryMapper.normalize_filters(filters or {})

        # Check for empty search
        if QueryMapper.is_empty_search(semantic_queries, normalized_filters):
            return {
                "results": [],
                "total_count": 0,
                "query_mode": "empty",
                "vectors_used": [],
                "filters_applied": {},
                "search_time_ms": 0,
                "embedding_model": "openai+colbert",
                "filter_analysis": None,
                "error": "Provide either a semantic query or filters",
            }

        # Generate embeddings for semantic queries
        dense_vectors, colbert_vectors = self._generate_embeddings(semantic_queries)

        # Execute search
        search_result = self.vector_store.search(
            dense_vectors=dense_vectors,
            colbert_vectors=colbert_vectors,
            filters=normalized_filters,
            limit=limit,
            offset=offset,
            score_threshold=score_threshold,
            skip_ids=skip_ids,
        )

        # Compute filter analysis if filters are applied
        filter_analysis = None
        if include_filter_analysis and normalized_filters:
            filter_analysis = self._compute_filter_analysis(
                normalized_filters,
                search_result["total_count"]
            )

        # Calculate search time
        search_time_ms = (time.time() - start_time) * 1000

        return {
            "results": search_result["results"],
            "total_count": search_result["total_count"],
            "query_mode": search_result["query_mode"],
            "vectors_used": search_result["vectors_used"],
            "filters_applied": search_result["filters_applied"],
            "search_time_ms": round(search_time_ms, 2),
            "embedding_model": "openai+colbert",
            "filter_analysis": filter_analysis,
        }

    def _compute_filter_analysis(
        self,
        filters: Dict[str, Any],
        current_count: int,
    ) -> Dict[str, Any]:
        """
        Compute filter impact analysis.

        Args:
            filters: Applied filters
            current_count: Current result count with all filters

        Returns:
            Filter analysis dict
        """
        analysis = self.filter_analysis_service.analyze_filter_impact(filters)

        # Format impacts for response
        impacts = []
        for impact in analysis["impacts"]:
            impacts.append({
                "filter": impact["filter"],
                "value": impact["value"],
                "count_with": impact["count_with"],
                "count_without": impact["count_without"],
                "removed_count": impact["removed_count"],
                "impact_percentage": impact["impact_percentage"],
            })

        return {
            "impacts": impacts,
            "recommendations": analysis["recommendations"],
            "total_without_filters": analysis["total_without_filters"],
            "current_count": analysis["current_count"],
        }

    def _generate_embeddings(
        self,
        semantic_queries: Dict[str, str]
    ) -> tuple:
        """
        Generate embeddings for semantic queries.

        Args:
            semantic_queries: Dict of {field: query_text}

        Returns:
            Tuple of (dense_vectors, colbert_vectors)
        """
        dense_vectors = {}
        colbert_vectors = {}

        if not semantic_queries:
            return dense_vectors, colbert_vectors

        # Get required providers
        providers_needed = set()
        for field in semantic_queries:
            vector_config = VECTOR_CONFIG.get(field)
            if vector_config:
                providers_needed.add(vector_config["provider"])

        # Load providers
        providers = {}
        for provider_name in providers_needed:
            providers[provider_name] = EmbeddingProviderFactory.get_provider(
                provider_name, device=self.device
            )

        # Generate embeddings
        for field, query_text in semantic_queries.items():
            vector_config = VECTOR_CONFIG.get(field)
            if vector_config:
                provider = providers.get(vector_config["provider"])
                if provider:
                    if vector_config["type"] == "multivector":
                        # ColBERT vectors (vibe_report)
                        colbert_vectors[field] = provider.embed(query_text)
                    else:
                        # Dense vectors (education, profession)
                        dense_vectors[field] = provider.embed(query_text)

        return dense_vectors, colbert_vectors

    def get_providers_status(self) -> Dict[str, bool]:
        """Get status of loaded embedding providers."""
        status = {}
        for provider_name in get_required_providers():
            status[provider_name] = EmbeddingProviderFactory.is_loaded(
                provider_name, self.device
            )
        return status

    def warmup_providers(self) -> None:
        """Pre-load all embedding providers."""
        for provider_name in get_required_providers():
            logger.info(f"Loading provider: {provider_name}")
            provider = EmbeddingProviderFactory.get_provider(
                provider_name, device=self.device
            )
            # Warmup with a test embedding
            provider.embed("warmup text")
            logger.info(f"Provider loaded: {provider_name}")