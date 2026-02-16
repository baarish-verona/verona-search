"""Request models for API endpoints."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ParseRequest(BaseModel):
    """Request to parse a natural language query."""
    query: str = Field(..., min_length=1, description="Natural language search query")


class SearchRequest(BaseModel):
    """Request to execute a search."""
    # Raw query (will be parsed if no parsed_queries provided)
    query: Optional[str] = Field(None, description="Natural language query to auto-parse")

    # Pre-parsed semantic queries
    parsed_queries: Optional[Dict[str, str]] = Field(
        None,
        description="Pre-parsed semantic queries (education_query, profession_query, etc.)"
    )

    # Hard filters
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Filter conditions (genders, religions, min_age, etc.)"
    )

    # Pagination
    limit: int = Field(100, ge=1, le=200, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Number of results to skip")

    # Score threshold
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum score threshold")

    # IDs to skip
    skip_ids: Optional[List[str]] = Field(None, description="Profile IDs to exclude from results")
