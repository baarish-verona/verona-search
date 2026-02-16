"""API route definitions."""

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..models import (
    ParseRequest,
    ParseResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    CollectionInfoResponse,
    IngestUserProfile,
)
from ..models.responses import FilterAnalysis, FilterImpact, SearchResultPayload
from ..services import SearchService, QueryParser, IngestService
from ..vector_store import QdrantVectorStore
from .dependencies import get_search_service, get_query_parser, get_vector_store, get_ingest_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/collection/info", response_model=CollectionInfoResponse, tags=["collection"])
async def collection_info(
    vector_store: QdrantVectorStore = Depends(get_vector_store),
):
    """Get collection information."""
    try:
        info = vector_store.collection_info()
        return CollectionInfoResponse(**info)
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/parse", response_model=ParseResponse, tags=["search"])
async def parse_query(
    request: ParseRequest,
    query_parser: Optional[QueryParser] = Depends(get_query_parser),
):
    """
    Parse natural language query into structured format.

    Uses GPT-4o-mini to extract filters and semantic queries.
    """
    if not query_parser:
        raise HTTPException(
            status_code=503,
            detail="Query parsing unavailable - OpenAI API key not configured"
        )

    try:
        parsed = query_parser.parse(request.query)
        return ParseResponse(
            original_query=parsed["original_query"],
            filters=parsed["filters"],
            education_query=parsed["education_query"],
            profession_query=parsed["profession_query"],
            vibe_report_query=parsed["vibe_report_query"],
        )
    except Exception as e:
        logger.error(f"Query parsing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse, tags=["search"])
async def search(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
    query_parser: Optional[QueryParser] = Depends(get_query_parser),
):
    """
    Execute search with semantic queries and/or filters.

    If `query` is provided without `parsed_queries`, will auto-parse using LLM.
    """
    parsed_queries = request.parsed_queries

    # Auto-parse if raw query provided without parsed queries
    if request.query and not parsed_queries:
        if not query_parser:
            raise HTTPException(
                status_code=400,
                detail="Provide parsed_queries or configure OpenAI API key for auto-parsing"
            )
        parsed = query_parser.parse(request.query)
        parsed_queries = {
            "education_query": parsed["education_query"],
            "profession_query": parsed["profession_query"],
            "vibe_report_query": parsed["vibe_report_query"],
        }
        # Merge parsed filters with request filters
        if parsed["filters"]:
            request_filters = request.filters or {}
            filters = {**parsed["filters"], **request_filters}
        else:
            filters = request.filters
    else:
        filters = request.filters

    try:
        result = search_service.search(
            parsed_queries=parsed_queries,
            filters=filters,
            limit=request.limit,
            offset=request.offset,
            score_threshold=request.score_threshold,
            skip_ids=request.skip_ids,
        )

        # Build filter analysis response if present
        filter_analysis = None
        if result.get("filter_analysis"):
            fa = result["filter_analysis"]
            filter_analysis = FilterAnalysis(
                impacts=[FilterImpact(**impact) for impact in fa["impacts"]],
                recommendations=fa["recommendations"],
                total_without_filters=fa["total_without_filters"],
                current_count=fa["current_count"],
            )

        return SearchResponse(
            query=request.query,
            parsed=parsed_queries,
            results=[
                SearchResult(
                    id=r["id"],
                    score=r["score"],
                    payload=SearchResultPayload(**r["payload"])
                )
                for r in result["results"]
            ],
            total_count=result["total_count"],
            vectors_used=result["vectors_used"],
            filters_applied=result["filters_applied"],
            search_time_ms=result["search_time_ms"],
            embedding_model=result["embedding_model"],
            filter_analysis=filter_analysis,
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=SearchResponse, tags=["search"])
async def search_get(
    q: Optional[str] = Query(None, description="Natural language search query"),
    education_query: Optional[str] = Query(None, description="Education semantic query"),
    profession_query: Optional[str] = Query(None, description="Profession semantic query"),
    vibe_report_query: Optional[str] = Query(None, description="Vibe/personality semantic query"),
    genders: Optional[List[str]] = Query(None, description="Filter by gender"),
    religions: Optional[List[str]] = Query(None, description="Filter by religion codes"),
    locations: Optional[List[str]] = Query(None, description="Filter by locations"),
    min_age: Optional[int] = Query(None, description="Minimum age"),
    max_age: Optional[int] = Query(None, description="Maximum age"),
    min_height: Optional[int] = Query(None, description="Minimum height (inches)"),
    max_height: Optional[int] = Query(None, description="Maximum height (inches)"),
    min_income: Optional[int] = Query(None, description="Minimum income (LPA)"),
    max_income: Optional[int] = Query(None, description="Maximum income (LPA)"),
    marital_statuses: Optional[List[str]] = Query(None, description="Marital status codes"),
    food_habits: Optional[List[str]] = Query(None, description="Food habit codes"),
    smoking: Optional[List[str]] = Query(None, description="Smoking codes"),
    drinking: Optional[List[str]] = Query(None, description="Drinking codes"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    offset: int = Query(0, ge=0, description="Result offset"),
    search_service: SearchService = Depends(get_search_service),
    query_parser: Optional[QueryParser] = Depends(get_query_parser),
):
    """
    Search via GET parameters.

    Convenient for simple searches without JSON body.
    """
    # Build parsed queries if any semantic queries provided
    parsed_queries = None
    if any([education_query, profession_query, vibe_report_query]):
        parsed_queries = {
            "education_query": education_query or "",
            "profession_query": profession_query or "",
            "vibe_report_query": vibe_report_query or "",
        }

    # Build filters
    filters = {}
    if genders:
        filters["genders"] = genders
    if religions:
        filters["religions"] = religions
    if locations:
        filters["locations"] = locations
    if min_age is not None:
        filters["min_age"] = min_age
    if max_age is not None:
        filters["max_age"] = max_age
    if min_height is not None:
        filters["min_height"] = min_height
    if max_height is not None:
        filters["max_height"] = max_height
    if min_income is not None:
        filters["min_income"] = min_income
    if max_income is not None:
        filters["max_income"] = max_income
    if marital_statuses:
        filters["marital_statuses"] = marital_statuses
    if food_habits:
        filters["food_habits"] = food_habits
    if smoking:
        filters["smoking"] = smoking
    if drinking:
        filters["drinking"] = drinking

    # Create request and delegate
    request = SearchRequest(
        query=q,
        parsed_queries=parsed_queries,
        filters=filters if filters else None,
        limit=limit,
        offset=offset,
    )

    return await search(request, search_service, query_parser)


@router.post("/ingest", tags=["ingest"])
async def ingest_profile(
    profile: IngestUserProfile,
    ingest_service: IngestService = Depends(get_ingest_service),
):
    """Ingest a single user profile into Qdrant."""
    user = ingest_service.ingest(profile)
    return user


@router.get("/profile/{profile_id}", tags=["profile"])
async def get_profile(
    profile_id: str,
    vector_store: QdrantVectorStore = Depends(get_vector_store),
):
    """Fetch a profile by ID."""
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, profile_id))
    point = vector_store.get_point(point_id)
    if not point:
        raise HTTPException(status_code=404, detail="Profile not found")
    return point
