"""Response models for API endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ProcessedPhotoResponse(BaseModel):
    """Processed photo in search results."""
    show_case_id: str
    url: str
    cropped_url: Optional[str] = None


class SearchResultPayload(BaseModel):
    """Typed payload for search results."""
    id: Optional[str] = None
    is_circulateable: bool = False
    is_paused: bool = False
    last_active: Optional[datetime] = None

    # Demographics
    gender: Optional[str] = None
    height: Optional[int] = None
    dob: Optional[str] = None
    current_location: Optional[str] = None
    annual_income: Optional[float] = None

    # Filter fields
    religion: Optional[str] = None
    caste: Optional[str] = None
    fitness: Optional[str] = None
    religiosity: Optional[str] = None
    smoking: Optional[str] = None
    drinking: Optional[str] = None
    family_type: Optional[str] = None
    food_habits: Optional[str] = None
    intent: Optional[str] = None
    open_to_children: Optional[str] = None

    # Text fields
    profession: Optional[str] = None
    education: Optional[str] = None
    vibe_report: Optional[str] = None
    blurb: Optional[str] = None
    profile_hook: Optional[str] = None

    # Lists
    life_style_tags: List[str] = []
    interests: List[str] = []
    photo_collection: List[ProcessedPhotoResponse] = []


class ParseResponse(BaseModel):
    """Response from query parsing."""
    original_query: str
    filters: Dict[str, Any]
    education_query: str
    profession_query: str
    vibe_report_query: str


class SearchResult(BaseModel):
    """Individual search result."""
    id: str
    score: float
    payload: SearchResultPayload


class FilterImpact(BaseModel):
    """Impact of a single filter on results."""
    filter: str
    value: Any
    count_with: int
    count_without: int
    removed_count: int
    impact_percentage: float


class FilterAnalysis(BaseModel):
    """Analysis of filter impacts on search results."""
    impacts: List[FilterImpact]
    recommendations: List[str]
    total_without_filters: int
    current_count: int


class SearchResponse(BaseModel):
    """Response from search."""
    query: Optional[str] = None
    parsed: Optional[Dict[str, str]] = None
    results: List[SearchResult]
    total_count: int
    vectors_used: List[str]
    filters_applied: Dict[str, Any]
    search_time_ms: float
    embedding_model: str
    filter_analysis: Optional[FilterAnalysis] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: float


class CollectionInfoResponse(BaseModel):
    """Collection information response."""
    name: str
    points_count: int
    vectors_count: Optional[int] = None
    status: str
