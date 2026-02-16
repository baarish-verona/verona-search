"""Pydantic models for profile ingestion."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field




class EducationDetail(BaseModel):
    """Education details from input profile."""

    id: Optional[str] = None
    college: Optional[str] = None
    college_other: Optional[str] = Field(None, alias="collegeOther")
    degree: Optional[str] = None
    degree_other: Optional[str] = Field(None, alias="degreeOther")
    degree_type: Optional[str] = Field(None, alias="degreeType")
    tier: Optional[int] = None
    is_verified: Optional[bool] = Field(None, alias="isVerified")

    class Config:
        populate_by_name = True


class ProfessionalJourneyDetail(BaseModel):
    """Professional journey details from input profile."""

    id: Optional[str] = None
    company: Optional[str] = None
    company_other: Optional[str] = Field(None, alias="companyOther")
    designation: Optional[str] = None
    designation_other: Optional[str] = Field(None, alias="designationOther")
    is_verified: Optional[bool] = Field(None, alias="isVerified")

    class Config:
        populate_by_name = True


class AppVersionDetails(BaseModel):
    """App version details containing last active timestamp."""

    last_updated_on: Optional[datetime] = Field(None, alias="lastUpdatedOn")


class PhotoDoc(BaseModel):
    """Photo document from input profile."""

    key: Optional[str] = None
    cropped_key: Optional[str] = Field(None, alias="croppedKey")
    is_removed: Optional[bool] = Field(None, alias="isRemoved")
    show_case_id: Optional[str] = Field(None, alias="showCaseId")
    media_id: Optional[str] = Field(None, alias="mediaId")
    media_type: Optional[str] = Field(None, alias="mediaType")
    is_cropped: Optional[bool] = Field(None, alias="isCropped")

    class Config:
        populate_by_name = True


class PauseDetails(BaseModel):
    """Pause details for profile."""

    is_paused: bool = Field(False, alias="isPaused")


class IngestUserProfile(BaseModel):
    """
    Input profile model for ingestion.

    Accepts camelCase input via alias mappings for compatibility with
    TypeScript/JSON sources.
    """

    # Required identifier (MongoDB _id)
    id: str = Field(..., alias="_id")

    # Name fields
    first_name: Optional[str] = Field(None, alias="firstName")
    last_name: Optional[str] = Field(None, alias="lastName")
    name: Optional[str] = None

    # Status flags for is_circulateable
    is_ql: bool = Field(False, alias="isQL")
    is_active: bool = Field(False, alias="isActive")
    is_verified: bool = Field(False, alias="isVerified")
    is_non_serviceable: bool = Field(False, alias="isNonServiceable")
    is_soft_deleted: bool = Field(False, alias="isSoftDeleted")
    pause_details: Optional[PauseDetails] = Field(None, alias="pauseDetails")
    onboarded_on: Optional[datetime] = Field(None, alias="onboardedOn")
    test_lead: Optional[bool] = Field(None, alias="testLead")

    # Other flags
    force_update_vector_profile: bool = Field(False, alias="forceUpdate")
    has_installed_app: bool = Field(False, alias="hasInstalledApp")
    is_about_you_form_filled: bool = Field(False, alias="isAboutYouFormFilled")
    is_photo_collection_submitted: bool = Field(False, alias="isPhotoCollectionSubmitted")

    # Demographics
    gender: str
    height: int
    dob: str
    current_location: str = Field(..., alias="currentLocation")
    annual_income: Optional[float] = Field(None, alias="annualIncome")

    # Filter fields
    religion: str
    caste: Optional[str] = None
    fitness: Optional[str] = None
    religiosity: Optional[str] = None
    smoking: Optional[str] = None
    drinking: Optional[str] = None
    food_habits: Optional[str] = Field(None, alias="foodHabits")
    intent: Optional[str] = None
    open_to_children: Optional[str] = Field(None, alias="openToChildren")
    family_type: Optional[str] = Field(None, alias="familyType")

    # App details
    app_version_details: AppVersionDetails = Field(..., alias="appVersionDetails")

    # Text fields for embedding
    education_details: Optional[List[EducationDetail]] = Field(None, alias="educationDetails")
    professional_journey_details: Optional[List[ProfessionalJourneyDetail]] = Field(
        None, alias="professionalJourneyDetails"
    )
    highlighted_professional_detail_id: Optional[str] = Field(
        None, alias="highlightedProfessionalDetailId"
    )
    similar_interests_v2: Optional[List[str]] = Field(None, alias="similarInterestsV2")
    blurb: Optional[str] = None
    photo_collection: Optional[List[PhotoDoc]] = Field(None, alias="photoCollection")
    show_case_profile_ids: Optional[List[str]] = Field(None, alias="showCaseProfileIds")

    class Config:
        populate_by_name = True