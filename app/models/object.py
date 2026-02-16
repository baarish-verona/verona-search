"""Domain object models for Qdrant storage."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from .ingest import IngestUserProfile


class PartnerPreference(BaseModel):
    """Partner preference - placeholder for future use."""
    pass


class EducationDetails(BaseModel):
    """Education details - used only for conversion."""
    id: str
    college: Optional[str] = None
    degree: Optional[str] = None


class ProfessionalJourneyDetails(BaseModel):
    """Professional journey details - used only for conversion."""
    id: str
    company: Optional[str] = None
    designation: Optional[str] = None


class ProcessedPhoto(BaseModel):
    """Processed photo with CloudFront URLs."""
    show_case_id: str
    url: str
    cropped_url: Optional[str] = None


class User(BaseModel):
    """Simplified User model for Qdrant storage."""
    id: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    name: Optional[str] = None
    is_circulateable: bool = False
    is_paused: bool = False
    last_active: Optional[datetime] = None
    gender: str
    height: int
    dob: str
    age: Optional[int] = None  # Computed from dob
    current_location: str
    annual_income: Optional[float] = None
    religion: str
    caste: Optional[str] = None
    fitness: Optional[str] = None
    religiosity: Optional[str] = None
    smoking: Optional[str] = None
    drinking: Optional[str] = None
    family_type: Optional[str] = None
    food_habits: Optional[str] = None
    intent: Optional[str] = None
    open_to_children: Optional[str] = None

    # Derived text fields (with vectors)
    profession: Optional[str] = None
    profession_hash: Optional[str] = None
    education: Optional[str] = None
    education_hash: Optional[str] = None
    vibe_report: Optional[str] = None
    vibe_report_hash: Optional[str] = None

    # Content fields
    blurb: Optional[str] = None
    profile_hook: Optional[str] = None
    life_style_tags: List[str] = []
    interests: List[str] = []
    photo_collection: List[ProcessedPhoto] = []

    @staticmethod
    def _md5_hash(text: Optional[str]) -> Optional[str]:
        """Compute MD5 hash of text, returns None if text is None."""
        if not text:
            return None
        return hashlib.md5(text.encode()).hexdigest()

    @staticmethod
    def _compute_age(dob: str) -> Optional[int]:
        """Compute age from date of birth string (YYYY-MM-DD format)."""
        try:
            birth_date = datetime.strptime(dob, "%Y-%m-%d")
            today = datetime.now()
            age = today.year - birth_date.year
            # Adjust if birthday hasn't occurred yet this year
            if (today.month, today.day) < (birth_date.month, birth_date.day):
                age -= 1
            return age
        except (ValueError, TypeError):
            return None

    @classmethod
    def from_ingest_profile(cls, profile: "IngestUserProfile") -> "User":
        """Create a User from an IngestUserProfile."""
        from .ingest import IngestUserProfile  # noqa: F811

        # Compute is_circulateable
        is_paused = profile.pause_details.is_paused if profile.pause_details else False
        is_circulateable = (
            profile.is_ql
            and profile.is_active
            and profile.is_verified
            and profile.onboarded_on is not None
            and not profile.is_non_serviceable
            and not profile.is_soft_deleted
            and not is_paused
            and not profile.test_lead
        )

        # Get last_active from app_version_details
        last_active = profile.app_version_details.last_updated_on

        # Build profession string from professional journey details
        profession = cls._build_profession(profile)

        # Build education string from education details
        education = cls._build_education(profile)

        # Build photo collection with CloudFront URLs
        photo_collection = cls._build_photo_collection(profile)

        # Build name if not provided
        name = profile.name
        if not name and (profile.first_name or profile.last_name):
            name = f"{profile.first_name or ''} {profile.last_name or ''}".strip()

        return cls(
            id=profile.id,
            first_name=profile.first_name,
            last_name=profile.last_name,
            name=name,
            is_circulateable=is_circulateable,
            is_paused=is_paused,
            last_active=last_active,
            # Demographics
            gender=profile.gender,
            height=profile.height,
            dob=profile.dob,
            age=cls._compute_age(profile.dob),
            current_location=profile.current_location,
            annual_income=profile.annual_income,
            # Filter fields
            religion=profile.religion,
            caste=profile.caste,
            fitness=profile.fitness,
            religiosity=profile.religiosity,
            smoking=profile.smoking,
            drinking=profile.drinking,
            family_type=profile.family_type,
            food_habits=profile.food_habits,
            intent=profile.intent,
            open_to_children=profile.open_to_children,
            # Derived fields with hashes
            profession=profession,
            profession_hash=cls._md5_hash(profession),
            education=education,
            education_hash=cls._md5_hash(education),
            vibe_report=None,  # Not in IngestUserProfile yet
            vibe_report_hash=None,
            # Content fields
            interests=profile.similar_interests_v2 or [],
            blurb=profile.blurb,
            photo_collection=photo_collection,
        )

    @classmethod
    def _build_profession(cls, profile: "IngestUserProfile") -> Optional[str]:
        """
        Build profession string from professional journey details.

        Uses highlighted professional detail if specified, otherwise uses the last element.
        Format: "{designation} at {company}"

        Note: Uses *Other fields if they exist, otherwise falls back to regular fields.
        """
        if not profile.professional_journey_details:
            return None

        # Find the highlighted detail or use the last one
        selected_detail = None
        if profile.highlighted_professional_detail_id:
            for detail in profile.professional_journey_details:
                if detail.id == profile.highlighted_professional_detail_id:
                    selected_detail = detail
                    break

        if not selected_detail:
            selected_detail = profile.professional_journey_details[-1]

        # Use *Other fields if they exist, otherwise fall back to regular fields
        designation = selected_detail.designation_other or selected_detail.designation
        company = selected_detail.company_other or selected_detail.company

        # Build the string
        parts = []
        if designation:
            parts.append(designation)
        if company:
            if parts:
                parts.append(f"at {company}")
            else:
                parts.append(company)

        return " ".join(parts) if parts else None

    @classmethod
    def _build_education(cls, profile: "IngestUserProfile") -> Optional[str]:
        """
        Build education string from education details.

        Format: "{degree} from {college}; {degree2} from {college2}"

        Note: Uses *Other fields if they exist, otherwise falls back to regular fields.
        """
        if not profile.education_details:
            return None

        parts = []
        for edu in profile.education_details:
            # Use *Other fields if they exist, otherwise fall back to regular fields
            degree = edu.degree_other or edu.degree
            college = edu.college_other or edu.college

            if degree and college:
                parts.append(f"{degree} from {college}")
            elif degree:
                parts.append(degree)
            elif college:
                parts.append(college)

        return "; ".join(parts) if parts else None

    @classmethod
    def _build_photo_collection(cls, profile: "IngestUserProfile") -> List[ProcessedPhoto]:
        """
        Build photo collection with CloudFront URLs.

        Logic:
        - Filter out removed photos
        - Only include photos in showCaseProfileIds
        - Match by showCaseId OR mediaId (when mediaType is IMAGE_JPEG)
        - Build URLs using CloudFront
        """
        from ..config import get_settings

        settings = get_settings()
        cloud_front_url = settings.cloud_front_url

        if not profile.photo_collection or not profile.show_case_profile_ids:
            return []

        # Filter out removed photos
        photos = [p for p in profile.photo_collection if not p.is_removed]

        processed_photos = []
        for showcase_id in profile.show_case_profile_ids:
            # Try to match by showCaseId first
            matched_photo = next(
                (p for p in photos if p.show_case_id == showcase_id),
                None
            )

            # If not found, try to match by mediaId where mediaType is IMAGE_JPEG
            if not matched_photo:
                matched_photo = next(
                    (p for p in photos
                     if p.media_id == showcase_id and p.media_type == "IMAGE_JPEG"),
                    None
                )

            if matched_photo and matched_photo.key:
                # Use cropped_key if it exists, otherwise fall back to key
                photo_key = matched_photo.cropped_key or matched_photo.key
                url = f"{cloud_front_url}/{photo_key}"

                processed_photos.append(ProcessedPhoto(
                    show_case_id=showcase_id,
                    url=url,
                ))

        return processed_photos
