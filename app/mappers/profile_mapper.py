"""Profile mapper for domain to Qdrant point conversion."""

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class QdrantPointModel:
    """Represents a Qdrant point ready for upsert."""
    id: str
    vectors: Dict[str, Union[List[float], List[List[float]]]]
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Qdrant upsert."""
        return {
            "id": self.id,
            "vectors": self.vectors,
            "payload": self.payload,
        }


class ProfileMapper:
    """
    Maps domain UserProfile to Qdrant Point and vice versa.

    Responsibilities:
    - Generate deterministic point IDs from user_id
    - Separate fields into vectors vs payload
    - Handle missing/null fields gracefully
    - Maintain field type consistency
    """

    # Fields that become vectors (require embedding)
    VECTOR_FIELDS = {
        "education": "education_text",
        "profession": "profession_text",
        "interests": "interests_text",
        "blurb": "blurb",
    }

    # Fields indexed for filtering (stored in payload)
    FILTER_FIELDS = [
        "age", "gender", "height", "income", "religion",
        "location", "marital_status", "family_type", "food_habits",
        "smoking", "drinking", "religiosity", "fitness", "intent"
    ]

    # Fields stored for display (not indexed)
    DISPLAY_FIELDS = [
        "user_id", "name", "education_text", "profession_text",
        "interests_text", "blurb"
    ]

    @classmethod
    def generate_point_id(cls, user_id: str) -> str:
        """
        Generate deterministic UUID from user_id.
        Ensures same user always maps to same point ID.

        Args:
            user_id: User identifier

        Returns:
            UUID-formatted string
        """
        hash_bytes = hashlib.md5(user_id.encode()).hexdigest()
        return f"{hash_bytes[:8]}-{hash_bytes[8:12]}-{hash_bytes[12:16]}-{hash_bytes[16:20]}-{hash_bytes[20:32]}"

    @classmethod
    def to_qdrant_point(
        cls,
        profile: Dict[str, Any],
        vectors: Dict[str, Union[List[float], List[List[float]]]]
    ) -> QdrantPointModel:
        """
        Convert domain profile + embeddings to Qdrant point.

        Args:
            profile: Domain user profile dict
            vectors: Pre-computed embeddings {vector_name: embedding}

        Returns:
            QdrantPointModel ready for upsert

        Raises:
            ValueError: If profile missing user_id
        """
        user_id = profile.get("user_id")
        if not user_id:
            raise ValueError("Profile must have user_id")

        point_id = cls.generate_point_id(user_id)

        # Build payload (filter fields + display fields)
        payload = {}

        # Add filter fields
        for field_name in cls.FILTER_FIELDS:
            value = profile.get(field_name)
            if value is not None:
                payload[field_name] = value

        # Add display fields
        for field_name in cls.DISPLAY_FIELDS:
            value = profile.get(field_name)
            if value is not None:
                payload[field_name] = value

        return QdrantPointModel(
            id=point_id,
            vectors=vectors,
            payload=payload
        )

    @classmethod
    def to_domain_profile(cls, qdrant_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Qdrant search result to domain profile.

        Args:
            qdrant_point: Qdrant point with id, score, payload

        Returns:
            Domain profile dict with search metadata
        """
        payload = qdrant_point.get("payload", {})

        return {
            "id": qdrant_point.get("id"),
            "score": qdrant_point.get("score"),
            **payload
        }

    @classmethod
    def get_text_for_embedding(cls, profile: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract text fields that need embedding.

        Args:
            profile: Domain user profile

        Returns:
            Dict mapping vector_name -> text_content
        """
        result = {}
        for vector_name, field_name in cls.VECTOR_FIELDS.items():
            text = profile.get(field_name, "")
            result[vector_name] = text if text else ""
        return result

    @classmethod
    def validate_profile(cls, profile: Dict[str, Any]) -> List[str]:
        """
        Validate profile has required fields.

        Args:
            profile: Profile dict to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not profile.get("user_id"):
            errors.append("Missing required field: user_id")

        # Check at least one text field has content
        text_fields = cls.get_text_for_embedding(profile)
        if not any(text_fields.values()):
            errors.append("Profile must have at least one text field for embedding")

        return errors

    @classmethod
    def batch_to_qdrant_points(
        cls,
        profiles: List[Dict[str, Any]],
        vectors_batch: List[Dict[str, Union[List[float], List[List[float]]]]]
    ) -> List[QdrantPointModel]:
        """
        Convert batch of profiles to Qdrant points.

        Args:
            profiles: List of profile dicts
            vectors_batch: List of vector dicts (one per profile)

        Returns:
            List of QdrantPointModel
        """
        if len(profiles) != len(vectors_batch):
            raise ValueError("Profiles and vectors must have same length")

        points = []
        for profile, vectors in zip(profiles, vectors_batch):
            point = cls.to_qdrant_point(profile, vectors)
            points.append(point)

        return points
