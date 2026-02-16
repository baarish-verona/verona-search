"""Dynamic query builder for Qdrant with OpenAI + ColBERT."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import models

from ..config.embedding_specs import VECTOR_CONFIG


class QueryMode(Enum):
    """Query mode based on input analysis."""
    FILTER_ONLY = "filter_only"
    SEMANTIC_SEARCH = "semantic_search"


@dataclass
class QueryContext:
    """Context for building a search query."""
    dense_vectors: Dict[str, List[float]]
    colbert_vectors: Dict[str, List[List[float]]]
    filter_obj: Optional[models.Filter]
    limit: int
    offset: int = 0
    score_threshold: float = 0.0


class DynamicQueryBuilder:
    """
    Build Qdrant queries dynamically based on provided inputs.

    Supports:
    - Filter-only mode (scroll with filters)
    - Semantic search with multi-vector prefetch
    - ColBERT late interaction for vibe_report
    - DBSF fusion across all vectors
    """

    @classmethod
    def determine_mode(
        cls,
        dense_vectors: Dict[str, List[float]],
        colbert_vectors: Dict[str, List[List[float]]],
        filter_obj: Optional[models.Filter]
    ) -> QueryMode:
        """
        Determine query mode based on inputs.

        Args:
            dense_vectors: Dict of dense vector embeddings
            colbert_vectors: Dict of ColBERT embeddings
            filter_obj: Qdrant filter object

        Returns:
            QueryMode enum
        """
        has_semantic = bool(dense_vectors) or bool(colbert_vectors)
        if has_semantic:
            return QueryMode.SEMANTIC_SEARCH
        return QueryMode.FILTER_ONLY

    @classmethod
    def build_prefetch_queries(
        cls,
        context: QueryContext
    ) -> Tuple[List[models.Prefetch], List[str]]:
        """
        Build prefetch queries for semantic search.

        Args:
            context: Query context with vectors and settings

        Returns:
            Tuple of (prefetch_list, vectors_used_descriptions)
        """
        prefetches = []
        vectors_used = []

        # Build prefetch for each dense vector (education, profession)
        for field in ["education", "profession"]:
            if field in context.dense_vectors:
                embedding = context.dense_vectors[field]
                prefetches.append(models.Prefetch(
                    query=embedding,
                    using=field,
                    limit=context.limit + context.offset,
                    filter=context.filter_obj,
                ))
                vectors_used.append(f"{field}(openai)")

        # Handle vibe_report ColBERT if present
        if "vibe_report" in context.colbert_vectors:
            prefetches.append(models.Prefetch(
                query=context.colbert_vectors["vibe_report"],
                using="vibe_report",
                limit=context.limit + context.offset,
                filter=context.filter_obj,
            ))
            vectors_used.append("vibe_report(colbert)")

        return prefetches, vectors_used

    @classmethod
    def build_query_request(
        cls,
        context: QueryContext
    ) -> Dict[str, Any]:
        """
        Build complete query request for Qdrant.

        Args:
            context: Query context

        Returns:
            Dict with query parameters for Qdrant client
        """
        mode = cls.determine_mode(
            context.dense_vectors,
            context.colbert_vectors,
            context.filter_obj
        )

        if mode == QueryMode.FILTER_ONLY:
            return cls._build_filter_only_request(context)
        else:
            return cls._build_semantic_request(context)

    @classmethod
    def _build_filter_only_request(cls, context: QueryContext) -> Dict[str, Any]:
        """Build request for filter-only search (scroll)."""
        return {
            "mode": QueryMode.FILTER_ONLY,
            "filter": context.filter_obj,
            "limit": context.limit,
            "offset": context.offset,
            "with_payload": True,
            "with_vectors": False,
        }

    @classmethod
    def _build_semantic_request(cls, context: QueryContext) -> Dict[str, Any]:
        """Build request for semantic search with prefetch."""
        prefetches, vectors_used = cls.build_prefetch_queries(context)

        if not prefetches:
            return cls._build_filter_only_request(context)

        # Get the first available dense vector for the main query
        main_vector = None
        main_using = None

        for field in ["education", "profession"]:
            if field in context.dense_vectors:
                main_using = field
                main_vector = context.dense_vectors[field]
                break

        # Fall back to ColBERT if no dense vectors (vibe_report-only search)
        if main_vector is None and "vibe_report" in context.colbert_vectors:
            main_using = "vibe_report"
            main_vector = context.colbert_vectors["vibe_report"]

        return {
            "mode": QueryMode.SEMANTIC_SEARCH,
            "prefetch": prefetches,
            "query": main_vector,
            "using": main_using,
            "filter": context.filter_obj,
            "limit": context.limit,
            "offset": context.offset,
            "score_threshold": context.score_threshold if context.score_threshold > 0 else None,
            "with_payload": True,
            "with_vectors": False,
            "vectors_used": vectors_used,
        }
