"""Pydantic models for Verona AI Search."""

from .requests import (
    ParseRequest,
    SearchRequest,
)
from .responses import (
    ParseResponse,
    SearchResult,
    SearchResultPayload,
    ProcessedPhotoResponse,
    SearchResponse,
    HealthResponse,
    CollectionInfoResponse,
    FilterAnalysis,
    FilterImpact,
)
from .object import User, PartnerPreference, EducationDetails, ProfessionalJourneyDetails
from .ingest import IngestUserProfile

__all__ = [
    # Requests
    "ParseRequest",
    "SearchRequest",
    # Responses
    "ParseResponse",
    "SearchResult",
    "SearchResultPayload",
    "ProcessedPhotoResponse",
    "SearchResponse",
    "HealthResponse",
    "CollectionInfoResponse",
    "FilterAnalysis",
    "FilterImpact",
    # Domain
    "User",
    "PartnerPreference",
    "EducationDetails",
    "ProfessionalJourneyDetails",
    # Ingest
    "IngestUserProfile",
]
