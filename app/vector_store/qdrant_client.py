"""Qdrant vector store wrapper."""

import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from ..config.embedding_specs import VECTOR_CONFIG
from .filters import FilterBuilder
from .query_builder import DynamicQueryBuilder, QueryContext, QueryMode

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Qdrant vector store wrapper with multi-spec support.

    Handles:
    - Collection creation with multi-vector schema
    - Point upsert with named vectors
    - Search with dynamic query building
    - Filter-only and semantic search modes
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "matrimonial_profiles",
    ):
        """
        Initialize Qdrant client.

        Args:
            host: Qdrant host
            port: Qdrant port
            collection_name: Name of the collection
        """
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)

    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create collection with multi-spec vector schema.

        Args:
            recreate: If True, delete existing collection first

        Returns:
            True if collection was created
        """
        if recreate:
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except UnexpectedResponse:
                pass

        # Check if collection exists
        collections = self.client.get_collections().collections
        if any(c.name == self.collection_name for c in collections):
            logger.info(f"Collection already exists: {self.collection_name}")
            return False

        # Build vectors config from VECTOR_CONFIG
        vectors_config = {}
        for vector_name, config in VECTOR_CONFIG.items():
            if config["type"] == "multivector":
                vectors_config[vector_name] = models.VectorParams(
                    size=config["dim"],
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                )
            else:
                vectors_config[vector_name] = models.VectorParams(
                    size=config["dim"],
                    distance=models.Distance.COSINE,
                )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
        )

        # Create payload indexes for filterable fields
        self._create_payload_indexes()

        logger.info(f"Created collection: {self.collection_name}")
        return True

    def _create_payload_indexes(self) -> None:
        """Create indexes for filterable payload fields."""
        # Integer fields (for range queries)
        for field in ["age", "height", "income"]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.INTEGER,
            )

        # Keyword fields (for match/match_any queries)
        keyword_fields = [
            "id", "gender", "religion", "location",
            "marital_status", "family_type", "food_habits",
            "smoking", "drinking", "religiosity", "fitness", "intent",
            "caste", "open_to_children"
        ]
        for field in keyword_fields:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def upsert_points(
        self,
        points: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Upsert points to collection.

        Args:
            points: List of point dicts with id, vectors, payload
            batch_size: Number of points per batch

        Returns:
            Number of points upserted
        """
        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]

            qdrant_points = []
            for p in batch:
                qdrant_points.append(models.PointStruct(
                    id=p["id"],
                    vector=p["vectors"],
                    payload=p["payload"],
                ))

            self.client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points,
            )
            total += len(batch)

        return total

    def update_vectors(
        self,
        point_id: str,
        vectors: Dict[str, Union[List[float], List[List[float]]]],
    ) -> bool:
        """
        Update specific vectors for a point (partial update).

        Args:
            point_id: Point ID
            vectors: Dict of vector_name -> embedding

        Returns:
            True if successful
        """
        self.client.update_vectors(
            collection_name=self.collection_name,
            points=[
                models.PointVectors(
                    id=point_id,
                    vector=vectors,
                )
            ],
        )
        return True

    def search(
        self,
        dense_vectors: Optional[Dict[str, List[float]]] = None,
        colbert_vectors: Optional[Dict[str, List[List[float]]]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
        score_threshold: float = 0.0,
        skip_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute search with dynamic query building.

        Args:
            dense_vectors: Dict of logical_field -> dense embedding
            colbert_vectors: Dict of logical_field -> colbert embedding
            filters: Filter dict
            limit: Max results
            offset: Result offset
            score_threshold: Minimum score threshold
            skip_ids: Profile IDs to exclude from results

        Returns:
            Dict with results, total_count, query_mode, vectors_used
        """
        dense_vectors = dense_vectors or {}
        colbert_vectors = colbert_vectors or {}

        # Build filter
        filter_obj = FilterBuilder.build(filters) if filters else None

        # Add skip_ids filter if provided
        if skip_ids:
            # Convert profile IDs to point IDs
            point_ids_to_skip = [
                str(uuid.uuid5(uuid.NAMESPACE_DNS, profile_id))
                for profile_id in skip_ids
            ]
            skip_filter = models.Filter(
                must_not=[
                    models.HasIdCondition(has_id=point_ids_to_skip)
                ]
            )
            if filter_obj:
                # Combine with existing filter
                filter_obj = models.Filter(
                    must=[filter_obj, skip_filter]
                )
            else:
                filter_obj = skip_filter

        # Build query context
        context = QueryContext(
            dense_vectors=dense_vectors,
            colbert_vectors=colbert_vectors,
            filter_obj=filter_obj,
            limit=limit,
            offset=offset,
            score_threshold=score_threshold,
        )

        # Build query request
        query_request = DynamicQueryBuilder.build_query_request(context)

        # Execute based on mode
        if query_request["mode"] == QueryMode.FILTER_ONLY:
            return self._execute_filter_only(query_request, filters)
        else:
            return self._execute_semantic_search(query_request, filters)

    def _execute_filter_only(
        self,
        request: Dict[str, Any],
        original_filters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute filter-only search using scroll."""
        results = []
        scroll_filter = request.get("filter")

        # Use scroll for filter-only search
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=request["limit"],
            offset=request.get("offset", 0),
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            results.append({
                "id": record.id,
                "score": 1.0,  # No semantic score for filter-only
                "payload": record.payload,
            })

        # Get total count
        total = self.client.count(
            collection_name=self.collection_name,
            count_filter=scroll_filter,
            exact=True,
        ).count

        return {
            "results": results,
            "total_count": total,
            "query_mode": "filter_only",
            "vectors_used": [],
            "filters_applied": original_filters or {},
        }

    def _execute_semantic_search(
        self,
        request: Dict[str, Any],
        original_filters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute semantic search with prefetch and fusion."""
        # Execute query with prefetch
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=request.get("prefetch"),
            query=request["query"],
            using=request["using"],
            query_filter=request.get("filter"),
            limit=request["limit"],
            offset=request.get("offset", 0),
            score_threshold=request.get("score_threshold"),
            with_payload=True,
            with_vectors=False,
        )

        results = []
        for point in search_results.points:
            results.append({
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
            })

        # Count total (with filter if provided)
        count_filter = request.get("filter")
        total = self.client.count(
            collection_name=self.collection_name,
            count_filter=count_filter,
            exact=True,
        ).count

        return {
            "results": results,
            "total_count": total,
            "query_mode": "semantic_search",
            "vectors_used": request.get("vectors_used", []),
            "filters_applied": original_filters or {},
        }

    def get_point(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Get a single point by ID."""
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
            if points:
                return {
                    "id": points[0].id,
                    "payload": points[0].payload,
                }
            return None
        except Exception:
            return None

    def delete_point(self, point_id: str) -> bool:
        """Delete a point by ID."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[point_id]),
        )
        return True

    def set_payload(
        self,
        point_id: str,
        payload: Dict[str, Any],
    ) -> bool:
        """
        Update payload fields for a point (partial update).

        Args:
            point_id: Point ID
            payload: Dict of fields to update

        Returns:
            True if successful
        """
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=payload,
            points=[point_id],
        )
        return True

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Get total count of points matching filters."""
        filter_obj = FilterBuilder.build(filters) if filters else None
        return self.client.count(
            collection_name=self.collection_name,
            count_filter=filter_obj,
            exact=True,
        ).count

    def collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status.name,
        }
