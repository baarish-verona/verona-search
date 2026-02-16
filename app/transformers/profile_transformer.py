"""Transform input profiles into Qdrant-ready payloads."""

from datetime import datetime
from typing import Any, Dict, Optional

from ..models.ingest import IngestUserProfile


class ProfileTransformer:
    """
    Transform input profiles into Qdrant payloads.

    Handles:
    - Status flag computation (is_circulateable)
    - Age calculation from DOB
    - Text field concatenation for embeddings
    - Payload structure normalization
    """

    @classmethod
    def transform(cls, profile: IngestUserProfile) -> Dict[str, Any]:
        """
        Transform an input profile into a Qdrant-ready payload.

        Args:
            profile: Input profile from API request

        Returns:
            Dict with payload fields for Qdrant
        """
        payload = {
            "user_id": profile.user_id,
            "is_circulateable": cls._compute_is_circulateable(profile),
        }

        # Demographics (required fields)
        payload["gender"] = profile.gender
        payload["height"] = profile.height
        payload["location"] = profile.current_location

        # Age from DOB
        age = cls._calculate_age(profile.dob)
        if age is not None:
            payload["age"] = age

        # Filter fields (required)
        payload["religion"] = profile.religion
        payload["caste"] = profile.caste
        payload["fitness"] = profile.fitness
        payload["religiosity"] = profile.religiosity
        payload["smoking"] = profile.smoking
        payload["drinking"] = profile.drinking
        payload["food_habits"] = profile.food_habits
        payload["intent"] = profile.intent
        payload["open_to_children"] = profile.open_to_children

        # Optional filter fields
        if profile.family_type:
            payload["family_type"] = profile.family_type
        if profile.annual_income is not None:
            payload["income"] = profile.annual_income

        # Last active timestamp (app_version_details is required)
        if profile.app_version_details.last_updated_on:
            payload["last_active"] = profile.app_version_details.last_updated_on.isoformat()

        # Text fields for embeddings
        education_text = cls._build_education_text(profile)
        if education_text:
            payload["education_text"] = education_text

        profession_text = cls._build_profession_text(profile)
        if profession_text:
            payload["profession_text"] = profession_text

        interests_text = cls._build_interests_text(profile)
        if interests_text:
            payload["interests_text"] = interests_text

        if profile.blurb:
            payload["blurb"] = profile.blurb

        return payload

    @classmethod
    def _compute_is_circulateable(cls, profile: IngestUserProfile) -> bool:
        """
        Compute whether profile is circulateable.

        Profile is circulateable if:
        - isQL AND isActive AND isVerified AND onboardedOn is not None
        - AND NOT (isSoftDeleted OR isNonServiceable OR isPaused OR testLead)
        """
        is_paused = profile.pause_details.is_paused if profile.pause_details else False

        return (
            profile.is_ql
            and profile.is_active
            and profile.is_verified
            and profile.onboarded_on is not None
            and not profile.is_non_serviceable
            and not profile.is_soft_deleted
            and not is_paused
            and not profile.test_lead
        )

    @classmethod
    def _calculate_age(cls, dob: Optional[str]) -> Optional[int]:
        """
        Calculate age from date of birth string.

        Args:
            dob: Date of birth in YYYY-MM-DD format

        Returns:
            Age in years or None if DOB is invalid
        """
        if not dob:
            return None

        try:
            birth_date = datetime.strptime(dob, "%Y-%m-%d")
            today = datetime.now()
            age = today.year - birth_date.year

            # Adjust if birthday hasn't occurred this year
            if (today.month, today.day) < (birth_date.month, birth_date.day):
                age -= 1

            return age
        except ValueError:
            return None

    @classmethod
    def _build_education_text(cls, profile: IngestUserProfile) -> Optional[str]:
        """
        Build education text for embedding.

        Format: "{degree} from {college}" joined by ";"
        """
        if not profile.education_details:
            return None

        parts = []
        for edu in profile.education_details:
            if edu.degree and edu.college:
                parts.append(f"{edu.degree} from {edu.college}")
            elif edu.degree:
                parts.append(edu.degree)
            elif edu.college:
                parts.append(edu.college)

        return "; ".join(parts) if parts else None

    @classmethod
    def _build_profession_text(cls, profile: IngestUserProfile) -> Optional[str]:
        """
        Build profession text for embedding.

        Format: "{designation} at {company}" joined by ";"
        """
        if not profile.professional_journey_details:
            return None

        parts = []
        for job in profile.professional_journey_details:
            if job.designation and job.company:
                parts.append(f"{job.designation} at {job.company}")
            elif job.designation:
                parts.append(job.designation)
            elif job.company:
                parts.append(job.company)

        return "; ".join(parts) if parts else None

    @classmethod
    def _build_interests_text(cls, profile: IngestUserProfile) -> Optional[str]:
        """
        Build interests text for embedding.

        Format: interests joined by ","
        """
        if not profile.similar_interests_v2:
            return None

        return ", ".join(profile.similar_interests_v2)

    @classmethod
    def has_embeddable_content(cls, payload: Dict[str, Any]) -> bool:
        """
        Check if payload has any text content for embedding.

        Args:
            payload: Transformed payload dict

        Returns:
            True if at least one text field is present
        """
        text_fields = ["education_text", "profession_text", "interests_text", "blurb"]
        return any(payload.get(field) for field in text_fields)
