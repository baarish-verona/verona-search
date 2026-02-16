"""Query mapper for search query to Qdrant query conversion."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class QdrantSearchQuery:
    """Represents a Qdrant search query."""
    dense_vectors: Dict[str, List[float]] = field(default_factory=dict)
    colbert_vectors: Dict[str, List[List[float]]] = field(default_factory=dict)
    filter_conditions: Dict[str, Any] = field(default_factory=dict)
    limit: int = 50
    offset: int = 0
    score_threshold: float = 0.0


class QueryMapper:
    """
    Maps parsed search queries to Qdrant search parameters.

    Handles:
    - Semantic query → Vector embedding mapping
    - Filter extraction and validation
    - Empty query handling
    """

    # Semantic query fields → Vector names
    QUERY_TO_VECTOR = {
        "education_query": "education",
        "profession_query": "profession",
        "vibe_report_query": "vibe_report",
    }

    # Filter field mappings (API name → Qdrant payload name, filter_type)
    FILTER_MAPPINGS = {
        # Range filters
        "min_age": ("age", "gte"),
        "max_age": ("age", "lte"),
        "min_height": ("height", "gte"),
        "max_height": ("height", "lte"),
        "min_income": ("income", "gte"),
        "max_income": ("income", "lte"),

        # MatchAny filters (OR condition - all accept arrays)
        "genders": ("gender", "match_any"),
        "religions": ("religion", "match_any"),
        "locations": ("location", "match_any"),
        "marital_statuses": ("marital_status", "match_any"),
        "family_types": ("family_type", "match_any"),
        "food_habits": ("food_habits", "match_any"),
        "smoking": ("smoking", "match_any"),
        "drinking": ("drinking", "match_any"),
        "religiosity": ("religiosity", "match_any"),
        "fitness": ("fitness", "match_any"),
        "intent": ("intent", "match_any"),
    }

    # All categorical filter keys (all support arrays)
    ARRAY_FILTER_KEYS = [
        "genders", "religions", "locations", "marital_statuses",
        "family_types", "food_habits", "smoking", "drinking",
        "religiosity", "fitness", "intent"
    ]

    @classmethod
    def extract_semantic_queries(
        cls,
        parsed: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Extract non-empty semantic queries.

        Args:
            parsed: Parsed query dict from LLM

        Returns:
            Dict of {vector_name: query_text} for non-empty queries
        """
        result = {}
        for query_key, vector_name in cls.QUERY_TO_VECTOR.items():
            query_text = parsed.get(query_key, "")
            if query_text and query_text.strip():
                result[vector_name] = query_text.strip()
        return result

    @classmethod
    def normalize_filters(cls, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and validate filter values.

        Handles:
        - Type coercion (string "25" → int 25)
        - Empty value removal
        - Array normalization (all categorical filters accept arrays)

        Args:
            filters: Raw filter dict

        Returns:
            Normalized filter dict
        """
        if not filters:
            return {}

        normalized = {}

        for key, value in filters.items():
            if value is None or value == "" or value == []:
                continue

            # Handle range filters (convert to int)
            if key in ["min_age", "max_age", "min_height", "max_height",
                       "min_income", "max_income"]:
                try:
                    normalized[key] = int(value)
                except (ValueError, TypeError):
                    continue

            # Handle array filters (all categorical filters)
            elif key in cls.ARRAY_FILTER_KEYS:
                if isinstance(value, str):
                    normalized[key] = [value]
                elif isinstance(value, list):
                    normalized[key] = [v for v in value if v]

            # Pass through other values
            else:
                normalized[key] = value

        return normalized

    @classmethod
    def is_filter_only_search(
        cls,
        semantic_queries: Dict[str, str],
        filters: Dict[str, Any]
    ) -> bool:
        """
        Check if this is a filter-only search (no semantic component).

        Args:
            semantic_queries: Dict of semantic queries
            filters: Filter dict

        Returns:
            True if filter-only search
        """
        has_semantic = bool(semantic_queries)
        has_filters = bool(filters)
        return not has_semantic and has_filters

    @classmethod
    def is_empty_search(
        cls,
        semantic_queries: Dict[str, str],
        filters: Dict[str, Any]
    ) -> bool:
        """
        Check if search has neither semantic nor filter components.

        Args:
            semantic_queries: Dict of semantic queries
            filters: Filter dict

        Returns:
            True if empty search
        """
        return not semantic_queries and not filters

    @classmethod
    def build_search_query(
        cls,
        parsed_queries: Optional[Dict[str, str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
        score_threshold: float = 0.0,
    ) -> QdrantSearchQuery:
        """
        Build QdrantSearchQuery from parsed queries and filters.

        Args:
            parsed_queries: Dict with education_query, profession_query, etc.
            filters: Filter dict
            limit: Max results
            offset: Result offset
            score_threshold: Minimum score

        Returns:
            QdrantSearchQuery object
        """
        semantic_queries = {}
        if parsed_queries:
            semantic_queries = cls.extract_semantic_queries(parsed_queries)

        normalized_filters = cls.normalize_filters(filters or {})

        return QdrantSearchQuery(
            dense_vectors={},  # To be filled by embedding service
            colbert_vectors={},  # To be filled by embedding service
            filter_conditions=normalized_filters,
            limit=limit,
            offset=offset,
            score_threshold=score_threshold,
        )

    @classmethod
    def get_fields_to_embed(cls, parsed_queries: Dict[str, str]) -> Dict[str, str]:
        """
        Get the fields that need embedding from parsed queries.

        Args:
            parsed_queries: Parsed query dict

        Returns:
            Dict of {logical_field: query_text}
        """
        return cls.extract_semantic_queries(parsed_queries)
