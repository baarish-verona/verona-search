"""Filter analysis service for identifying restrictive filters."""

import logging
from typing import Any, Dict, List, Optional

from ..vector_store import QdrantVectorStore, FilterBuilder

logger = logging.getLogger(__name__)


class FilterAnalysisService:
    """
    Analyzes filter impact on search results.

    Helps identify which filters are most restrictive when searches
    return no or few results.
    """

    def __init__(self, vector_store: QdrantVectorStore):
        """
        Initialize filter analysis service.

        Args:
            vector_store: Qdrant vector store instance
        """
        self.vector_store = vector_store

    def analyze_filter_impact(
        self,
        filters: Dict[str, Any],
        base_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Analyze impact of each filter on result count.

        Args:
            filters: Current filter dict
            base_count: Optional pre-computed base count (without filters)

        Returns:
            Dict with filter impact analysis
        """
        if not filters:
            return {
                "impacts": [],
                "recommendations": [],
                "total_without_filters": base_count or self.vector_store.count(),
            }

        # Get total count without filters
        total = base_count if base_count is not None else self.vector_store.count()

        # Get count with all filters
        current_count = self.vector_store.count(filters)

        # Analyze each filter's impact
        impacts = []
        for filter_key in filters:
            # Create filter dict without this filter
            filters_without = {k: v for k, v in filters.items() if k != filter_key}

            # Get count without this filter
            count_without = self.vector_store.count(filters_without) if filters_without else total

            # Calculate impact
            impact = {
                "filter": filter_key,
                "value": filters[filter_key],
                "count_with": current_count,
                "count_without": count_without,
                "removed_count": count_without - current_count,
                "impact_percentage": round(
                    ((count_without - current_count) / count_without * 100)
                    if count_without > 0 else 0,
                    1
                ),
            }
            impacts.append(impact)

        # Sort by impact (most restrictive first)
        impacts.sort(key=lambda x: x["removed_count"], reverse=True)

        # Generate recommendations
        recommendations = self._generate_recommendations(impacts, current_count, total)

        return {
            "impacts": impacts,
            "recommendations": recommendations,
            "total_without_filters": total,
            "current_count": current_count,
        }

    def _generate_recommendations(
        self,
        impacts: List[Dict[str, Any]],
        current_count: int,
        total: int,
    ) -> List[str]:
        """Generate human-readable recommendations based on impact analysis."""
        recommendations = []

        if current_count == 0 and impacts:
            # No results - suggest removing most restrictive filter
            most_restrictive = impacts[0]
            recommendations.append(
                f"Try removing the '{most_restrictive['filter']}' filter "
                f"(currently set to {most_restrictive['value']}) - "
                f"this would show {most_restrictive['count_without']} profiles"
            )

        elif current_count < 10 and impacts:
            # Few results - suggest relaxing filters
            for impact in impacts[:2]:  # Top 2 most restrictive
                if impact["impact_percentage"] > 50:
                    recommendations.append(
                        f"The '{impact['filter']}' filter is removing "
                        f"{impact['impact_percentage']}% of potential matches"
                    )

        return recommendations

    def get_filter_value_counts(
        self,
        field: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """
        Get count of profiles for each value of a categorical field.

        Args:
            field: Field name (e.g., "gender", "religion")
            filters: Optional base filters to apply

        Returns:
            Dict of {value: count}
        """
        # This would require aggregation queries not directly supported by Qdrant
        # For now, return empty dict - could be implemented with scroll
        logger.warning(f"get_filter_value_counts not fully implemented for field: {field}")
        return {}

    def suggest_filter_expansions(
        self,
        filters: Dict[str, Any],
        min_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Suggest filter modifications to achieve minimum results.

        Args:
            filters: Current filters
            min_results: Minimum desired results

        Returns:
            List of suggested filter modifications
        """
        analysis = self.analyze_filter_impact(filters)

        if analysis["current_count"] >= min_results:
            return []

        suggestions = []
        cumulative_count = analysis["current_count"]

        for impact in analysis["impacts"]:
            if cumulative_count >= min_results:
                break

            if impact["count_without"] > cumulative_count:
                suggestions.append({
                    "action": "remove",
                    "filter": impact["filter"],
                    "expected_count": impact["count_without"],
                })
                cumulative_count = impact["count_without"]

        return suggestions
