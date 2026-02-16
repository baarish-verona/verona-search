"""Ingest service for profile ingestion into Qdrant."""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..config import get_settings

logger = logging.getLogger(__name__)
from ..embeddings import EmbeddingProviderFactory
from ..models.ingest import IngestUserProfile
from ..models.object import User
from ..vector_store import QdrantVectorStore
from .vibe_service import VibeService


class IngestService:
    """Service for ingesting profiles into Qdrant."""

    def __init__(self, vector_store: QdrantVectorStore):
        self.vector_store = vector_store
        self.device = get_settings().embedding_device
        self._vibe_service: Optional[VibeService] = None

    @property
    def vibe_service(self) -> VibeService:
        """Lazy-load vibe service."""
        if self._vibe_service is None:
            self._vibe_service = VibeService()
        return self._vibe_service

    def _get_point_id(self, profile_id: str) -> str:
        """Generate deterministic point ID from profile id."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, profile_id))

    def _extract_photo_urls(self, user: User) -> List[Dict[str, str]]:
        """Extract photo URLs from User's processed photo_collection for vibe generation."""
        if not user.photo_collection:
            return []
        return [
            {"id": photo.show_case_id, "url": photo.url}
            for photo in user.photo_collection
        ]

    def ingest(self, profile: IngestUserProfile) -> User:
        """
        Ingest a single profile into Qdrant with smart updates.

        Logic:
        1. Convert profile to User model
        2. If not circulateable and doesn't exist → skip (don't create new non-circulateable profiles)
        3. If not circulateable and exists → update is_circulateable field
        4. If circulateable: full upsert or smart update based on existence/force flag

        Returns:
            User model with processed data
        """
        logger.info(f"Ingesting profile: {profile.id}")
        point_id = self._get_point_id(profile.id)

        # Convert ingest profile to User model
        user = User.from_ingest_profile(profile)

        # Fetch existing profile
        existing = self.vector_store.get_point(point_id)

        # Handle non-circulateable profiles
        if not user.is_circulateable:
            if existing is None:
                # Don't create new non-circulateable profiles
                logger.info(f"Skipping non-circulateable profile (doesn't exist): {profile.id}")
                return user
            else:
                # Update is_circulateable for existing profile
                logger.info(f"Updating is_circulateable=False for existing profile: {profile.id}")
                self.vector_store.set_payload(point_id, {"is_circulateable": False})
                return user

        # Circulateable profile - proceed with normal ingestion
        if existing is None or profile.force_update_vector_profile:
            # New profile or forced update - full upsert
            self._full_upsert(point_id, user)
        else:
            # Existing profile - smart update
            self._smart_update(point_id, user, existing["payload"])

        return user

    def _full_upsert(
        self,
        point_id: str,
        user: User,
    ) -> None:
        """Perform full upsert with all vectors."""
        openai_provider = EmbeddingProviderFactory.get_provider("openai-small", device=self.device)
        colbert_provider = EmbeddingProviderFactory.get_provider("bge-colbert", device=self.device)

        vectors = {}
        payload = user.model_dump()

        # Generate vectors for text fields (OpenAI)
        if user.education:
            vectors["education"] = openai_provider.embed(user.education)
        if user.profession:
            vectors["profession"] = openai_provider.embed(user.profession)

        # Generate vibe report
        try:
            photo_urls = self._extract_photo_urls(user)
            # Compute hash of input payload (for change detection)
            vibe_input_hash = self.vibe_service.compute_vibe_input_hash(user, photo_urls)
            vibe_map = self.vibe_service.generate_vibe_map(user, photo_urls)
            vibe_report = vibe_map.get("vibeReport")
            if vibe_report:
                payload["vibe_report"] = vibe_report
                payload["vibe_report_hash"] = vibe_input_hash  # Hash of input, not output
                payload["profile_hook"] = vibe_map.get("trumpAdamsSummary")
                # Extract imageTags and flatten to life_style_tags
                image_tags = vibe_map.get("imageTags", [])
                if image_tags:
                    all_tags = []
                    for item in image_tags:
                        tags = item.get("tags", [])
                        if isinstance(tags, list):
                            all_tags.extend(tags)
                    # Remove duplicates while preserving order
                    payload["life_style_tags"] = list(dict.fromkeys(all_tags))
                # Generate vibe_report vector (ColBERT for late interaction)
                vectors["vibe_report"] = colbert_provider.embed(vibe_report)
                logger.info(f"Vibe report generated: {user.id}")
            else:
                logger.warning(f"Vibe map returned no vibeReport: {vibe_map}")
        except Exception as e:
            logger.error(f"Vibe generation failed: {e}", exc_info=True)

        if vectors:
            self.vector_store.upsert_points([{
                "id": point_id,
                "vectors": vectors,
                "payload": payload
            }])

    def _smart_update(
        self,
        point_id: str,
        user: User,
        existing_payload: Dict[str, Any],
    ) -> None:
        """
        Smart update - only update changed fields and vectors.

        Compares hashes to detect changes in:
        - education (education_hash)
        - profession (profession_hash)
        - vibe_report content (based on education, profession, interests, blurb)
        """
        openai_provider = EmbeddingProviderFactory.get_provider("openai-small", device=self.device)

        payload_updates: Dict[str, Any] = {}
        vector_updates: Dict[str, Any] = {}

        # Check education changes
        existing_education_hash = existing_payload.get("education_hash")
        if user.education_hash != existing_education_hash:
            payload_updates["education"] = user.education
            payload_updates["education_hash"] = user.education_hash
            if user.education:
                vector_updates["education"] = openai_provider.embed(user.education)

        # Check profession changes
        existing_profession_hash = existing_payload.get("profession_hash")
        if user.profession_hash != existing_profession_hash:
            payload_updates["profession"] = user.profession
            payload_updates["profession_hash"] = user.profession_hash
            if user.profession:
                vector_updates["profession"] = openai_provider.embed(user.profession)

        # Check if vibe report needs to be generated
        photo_urls = self._extract_photo_urls(user)
        vibe_input_hash = self.vibe_service.compute_vibe_input_hash(user, photo_urls)
        existing_vibe_hash = existing_payload.get("vibe_report_hash")

        # TODO: This will be changed on the basis of a key vibe_report_regenerate_on_hash,
        # that will signify that even if hash doesn't match but vibe_report hash exists, then skip that
        # if vibe_input_hash != existing_vibe_hash:
        if not existing_vibe_hash:
            try:
                colbert_provider = EmbeddingProviderFactory.get_provider("bge-colbert", device=self.device)
                vibe_map = self.vibe_service.generate_vibe_map(user, photo_urls)
                vibe_report = vibe_map.get("vibeReport")
                if vibe_report:
                    payload_updates["vibe_report"] = vibe_report
                    payload_updates["vibe_report_hash"] = vibe_input_hash  # Hash of input, not output
                    payload_updates["profile_hook"] = vibe_map.get("trumpAdamsSummary")
                    # Extract imageTags and flatten to life_style_tags
                    image_tags = vibe_map.get("imageTags", [])
                    if image_tags:
                        all_tags = []
                        for item in image_tags:
                            tags = item.get("tags", [])
                            if isinstance(tags, list):
                                all_tags.extend(tags)
                        # Remove duplicates while preserving order
                        payload_updates["life_style_tags"] = list(dict.fromkeys(all_tags))
                    vector_updates["vibe_report"] = colbert_provider.embed(vibe_report)
                    logger.info(f"Vibe report regenerated: {user.id}")
                else:
                    logger.warning(f"Vibe map returned no vibeReport: {vibe_map}")
            except Exception as e:
                logger.error(f"Vibe generation failed: {e}", exc_info=True)

        # Update other payload fields that might have changed
        payload_updates.update(self._get_payload_diff(user, existing_payload))

        # Handle last_active with 2-hour threshold
        has_other_updates = bool(payload_updates) or bool(vector_updates)
        if self._should_update_last_active(user, existing_payload, has_other_updates):
            payload_updates["last_active"] = user.last_active

        # Apply updates
        if payload_updates:
            self.vector_store.set_payload(point_id, payload_updates)

        if vector_updates:
            self.vector_store.update_vectors(point_id, vector_updates)

    def _get_payload_diff(
        self,
        user: User,
        existing_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get payload fields that have changed (non-vector fields)."""
        diff = {}
        user_dict = user.model_dump()

        # Fields to compare (excluding hash fields, vector text fields, and last_active)
        compare_fields = [
            "is_circulateable", "is_paused",
            "gender", "height", "dob", "age", "current_location", "annual_income",
            "religion", "caste", "fitness", "religiosity", "smoking", "drinking",
            "family_type", "food_habits", "intent", "open_to_children",
            "blurb", "interests", "photo_collection"
        ]

        for field in compare_fields:
            new_value = user_dict.get(field)
            existing_value = existing_payload.get(field)
            if new_value != existing_value:
                diff[field] = new_value

        return diff

    def _should_update_last_active(
        self,
        user: User,
        existing_payload: Dict[str, Any],
        has_other_updates: bool,
    ) -> bool:
        """
        Determine if last_active should be updated.

        Rules:
        - If other fields are being updated, always update last_active if changed
        - If only last_active changed, only update if difference > 2 hours
        """
        if user.last_active is None:
            return False

        existing_last_active = existing_payload.get("last_active")
        if existing_last_active is None:
            return True  # No existing value, always update

        # Parse existing last_active if it's a string
        if isinstance(existing_last_active, str):
            try:
                existing_last_active = datetime.fromisoformat(existing_last_active.replace("Z", "+00:00"))
            except ValueError:
                return True  # Can't parse, update it

        # Ensure both are timezone-aware for comparison
        new_last_active = user.last_active
        if new_last_active.tzinfo is None:
            new_last_active = new_last_active.replace(tzinfo=timezone.utc)
        if existing_last_active.tzinfo is None:
            existing_last_active = existing_last_active.replace(tzinfo=timezone.utc)

        # If other updates exist, update last_active if it changed at all
        if has_other_updates:
            return new_last_active != existing_last_active

        # If only last_active changed, only update if difference > 2 hours
        time_diff = abs(new_last_active - existing_last_active)
        return time_diff > timedelta(hours=2)
