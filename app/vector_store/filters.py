"""Filter building utilities for Qdrant queries."""

from typing import Any, Dict, List, Optional

from qdrant_client import models


class FilterBuilder:
    """Build Qdrant filters from request parameters."""

    RANGE_FIELDS = ["age", "height", "income"]

    # All categorical filters support multiple values (match_any)
    # Maps API key → Qdrant payload field
    MATCH_ANY_FIELDS = {
        "gender": "gender",
        "religion": "religion",
        "location": "current_location",  # Payload uses current_location
        "marital_status": "marital_status",
        "family_type": "family_type",
        "food_habit": "food_habits",
        "smoking": "smoking",
        "drinking": "drinking",
        "religiosity": "religiosity",
        "fitness": "fitness",
        "intent": "intent",
        "caste": "caste",
        "open_to_children": "open_to_children",
        "test_lead": "test_lead",
    }

    # All categorical filter keys (all support arrays)
    ARRAY_FILTER_KEYS = [
        "gender", "religion", "location", "marital_status",
        "family_type", "food_habit", "smoking", "drinking",
        "religiosity", "fitness", "intent", "caste", "open_to_children",
        "test_lead"
    ]

    @classmethod
    def build(cls, filters: Dict[str, Any]) -> Optional[models.Filter]:
        """
        Convert filter dict to Qdrant Filter object.

        Args:
            filters: Dict containing filter parameters

        Returns:
            Qdrant Filter object or None if no filters
        """
        if not filters:
            return None

        conditions = []

        # Range filters
        for field in cls.RANGE_FIELDS:
            min_key = f"min_{field}"
            max_key = f"max_{field}"

            min_val = filters.get(min_key)
            max_val = filters.get(max_key)

            if min_val is not None or max_val is not None:
                range_cond = models.FieldCondition(
                    key=field,
                    range=models.Range(
                        gte=min_val,
                        lte=max_val
                    )
                )
                conditions.append(range_cond)

        # Match any filters (all categorical fields support multiple values)
        for api_key, payload_field in cls.MATCH_ANY_FIELDS.items():
            if api_key in filters and filters[api_key]:
                values = filters[api_key]
                if isinstance(values, str):
                    values = [values]
                if values:  # Non-empty list
                    conditions.append(models.FieldCondition(
                        key=payload_field,
                        match=models.MatchAny(any=values)
                    ))

        if not conditions:
            return None

        return models.Filter(must=conditions)

    @classmethod
    def build_single_filter(cls, field: str, value: Any) -> Optional[models.Filter]:
        """
        Build a filter for a single field.

        Args:
            field: Payload field name
            value: Filter value

        Returns:
            Qdrant Filter object
        """
        if value is None:
            return None

        if isinstance(value, (list, tuple)):
            condition = models.FieldCondition(
                key=field,
                match=models.MatchAny(any=list(value))
            )
        else:
            condition = models.FieldCondition(
                key=field,
                match=models.MatchValue(value=value)
            )

        return models.Filter(must=[condition])

    @classmethod
    def combine_filters(cls, *filters: Optional[models.Filter]) -> Optional[models.Filter]:
        """
        Combine multiple filters with AND logic.

        Args:
            *filters: Variable number of filter objects

        Returns:
            Combined Qdrant Filter object or None
        """
        conditions = []
        for f in filters:
            if f is not None and f.must:
                conditions.extend(f.must)

        if not conditions:
            return None

        return models.Filter(must=conditions)

    @classmethod
    def get_filter_fields(cls) -> List[str]:
        """Get list of all supported filter fields."""
        range_keys = []
        for field in cls.RANGE_FIELDS:
            range_keys.extend([f"min_{field}", f"max_{field}"])
        return range_keys + list(cls.MATCH_ANY_FIELDS.keys())

    @classmethod
    def build_default_filters(cls) -> models.Filter:
        """
        Build default filters that should always be applied.

        Returns profiles that are:
        - is_circulateable = True (profile is allowed to be shown)
        - is_paused != True (profile is not paused by user)
        - test_lead != True (exclude test users in controlled production)
        """
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="is_circulateable",
                    match=models.MatchValue(value=True)
                ),
            ],
            must_not=[
                models.FieldCondition(
                    key="is_paused",
                    match=models.MatchValue(value=True)
                ),
                models.FieldCondition(
                    key="test_lead",
                    match=models.MatchValue(value=True)
                ),
            ]
        )

    @classmethod
    def build_with_defaults(cls, filters: Optional[Dict[str, Any]] = None) -> models.Filter:
        """
        Build filters with default filters always applied.

        Args:
            filters: Optional user-provided filters

        Returns:
            Combined filter with defaults + user filters
        """
        default_filter = cls.build_default_filters()
        user_filter = cls.build(filters) if filters else None

        if user_filter is None:
            return default_filter

        # Combine default and user filters
        combined_must = list(default_filter.must or [])
        combined_must_not = list(default_filter.must_not or [])

        if user_filter.must:
            combined_must.extend(user_filter.must)

        return models.Filter(
            must=combined_must if combined_must else None,
            must_not=combined_must_not if combined_must_not else None
        )
