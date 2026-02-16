#!/usr/bin/env python3
"""
Transform raw user data into embedding-ready profiles.

This script takes raw user data (e.g., from a database export) and
transforms it into the format expected by the ingestion script.

Usage:
    python -m scripts.transform_profiles --input data/users.json --output data/profiles.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def transform_profile(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Transform a single raw profile to embedding-ready format.

    Handles:
    - Field renaming
    - Text field concatenation
    - Code normalization
    - Missing field handling
    """
    # Extract user ID (required)
    user_id = raw.get("user_id") or raw.get("id") or raw.get("_id")
    if not user_id:
        return None

    profile = {"user_id": str(user_id)}

    # Basic fields
    profile["name"] = raw.get("name") or raw.get("display_name")

    # Demographics
    if "age" in raw:
        try:
            profile["age"] = int(raw["age"])
        except (ValueError, TypeError):
            pass

    profile["gender"] = normalize_gender(raw.get("gender"))

    if "height" in raw:
        try:
            profile["height"] = int(raw["height"])
        except (ValueError, TypeError):
            pass

    if "income" in raw:
        try:
            profile["income"] = int(raw["income"])
        except (ValueError, TypeError):
            pass

    # Location
    location = raw.get("location") or raw.get("city")
    if location:
        if isinstance(location, list):
            profile["location"] = location
        else:
            profile["location"] = [location]

    # Lifestyle codes
    profile["religion"] = raw.get("religion")
    profile["marital_status"] = raw.get("marital_status")
    profile["family_type"] = raw.get("family_type")
    profile["food_habits"] = raw.get("food_habits") or raw.get("diet")
    profile["smoking"] = raw.get("smoking")
    profile["drinking"] = raw.get("drinking")
    profile["religiosity"] = raw.get("religiosity")
    profile["fitness"] = raw.get("fitness")
    profile["intent"] = raw.get("intent") or raw.get("marriage_intent")

    # Text fields for embedding
    profile["education_text"] = build_education_text(raw)
    profile["profession_text"] = build_profession_text(raw)
    profile["interests_text"] = build_interests_text(raw)
    profile["blurb"] = raw.get("blurb") or raw.get("about_me") or raw.get("bio")

    # Remove None values
    profile = {k: v for k, v in profile.items() if v is not None}

    return profile


def normalize_gender(value: Any) -> Optional[str]:
    """Normalize gender value."""
    if not value:
        return None

    value_lower = str(value).lower().strip()

    if value_lower in ["male", "m", "man"]:
        return "male"
    elif value_lower in ["female", "f", "woman"]:
        return "female"

    return value_lower


def build_education_text(raw: Dict[str, Any]) -> Optional[str]:
    """Build education text from raw fields."""
    parts = []

    # Degree
    degree = raw.get("degree") or raw.get("education_level")
    if degree:
        parts.append(degree)

    # Institution
    institution = raw.get("college") or raw.get("university") or raw.get("institution")
    if institution:
        parts.append(f"from {institution}")

    # Field of study
    field = raw.get("field_of_study") or raw.get("major") or raw.get("specialization")
    if field:
        parts.append(f"in {field}")

    # Check for combined field
    if not parts:
        education = raw.get("education") or raw.get("education_text")
        if education:
            return education

    return " ".join(parts) if parts else None


def build_profession_text(raw: Dict[str, Any]) -> Optional[str]:
    """Build profession text from raw fields."""
    parts = []

    # Job title
    title = raw.get("job_title") or raw.get("designation") or raw.get("role")
    if title:
        parts.append(title)

    # Company
    company = raw.get("company") or raw.get("employer") or raw.get("organization")
    if company:
        parts.append(f"at {company}")

    # Industry
    industry = raw.get("industry") or raw.get("sector")
    if industry:
        parts.append(f"({industry})")

    # Check for combined field
    if not parts:
        profession = raw.get("profession") or raw.get("profession_text") or raw.get("occupation")
        if profession:
            return profession

    return " ".join(parts) if parts else None


def build_interests_text(raw: Dict[str, Any]) -> Optional[str]:
    """Build interests text from raw fields."""
    # Check for combined field
    interests = raw.get("interests") or raw.get("interests_text") or raw.get("hobbies")

    if interests:
        if isinstance(interests, list):
            return ", ".join(interests)
        return interests

    return None


def transform_profiles(raw_profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform list of raw profiles."""
    transformed = []

    for raw in raw_profiles:
        profile = transform_profile(raw)
        if profile:
            # Validate minimum content
            has_text = any([
                profile.get("education_text"),
                profile.get("profession_text"),
                profile.get("interests_text"),
                profile.get("blurb"),
            ])
            if has_text:
                transformed.append(profile)
            else:
                logger.warning(f"Skipping profile {profile['user_id']}: no text content")
        else:
            logger.warning("Skipping profile: no user_id")

    return transformed


def main():
    parser = argparse.ArgumentParser(description="Transform raw profiles")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output JSON file"
    )

    args = parser.parse_args()

    # Load input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    with open(input_path, "r") as f:
        raw_data = json.load(f)

    # Handle both list and dict formats
    if isinstance(raw_data, list):
        raw_profiles = raw_data
    elif isinstance(raw_data, dict):
        raw_profiles = raw_data.get("profiles") or raw_data.get("users") or [raw_data]
    else:
        logger.error("Invalid input format")
        sys.exit(1)

    logger.info(f"Loaded {len(raw_profiles)} raw profiles")

    # Transform
    transformed = transform_profiles(raw_profiles)
    logger.info(f"Transformed {len(transformed)} profiles")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(transformed, f, indent=2)

    logger.info(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
