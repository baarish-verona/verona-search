"""LLM-based query parser using GPT-4o-mini."""

import json
import logging
from typing import Any, Dict, Optional

from openai import OpenAI

from ..config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a query parser for a matrimonial profile search system.
Extract structured information from natural language queries.

You must return a JSON object with these fields:
- filters: Object containing hard filters (age, gender, religion, etc.)
- education_query: Semantic query for education field (degrees, institutions, fields of study)
- profession_query: Semantic query for profession field (job titles, companies, industries)
- vibe_report_query: Semantic query for personality, vibe, lifestyle, and overall impression

Filter fields (use null if not specified):
- min_age, max_age: Integer age range
- genders: Array of ["male", "female"]
- religions: Array of religion codes ["HI", "MU", "CR", "SI", "JA", "BU", "PA", "JE", "BA", "NR"]
  Religion code meanings: HI=Hindu, MU=Muslim, CR=Christian, SI=Sikh, JA=Jain, BU=Buddhist, PA=Parsi, JE=Jewish, BA=Bahai, NR=No Religion
- locations: Array of city/state names (e.g., ["Mumbai", "Delhi", "Bangalore", "Maharashtra", "Karnataka"])
- marital_statuses: Array of ["NM", "DV", "WD", "AN"] (Never Married, Divorced, Widowed, Annulled)
- family_types: Array of ["NU", "JF", "LP", "LT"] (Nuclear, Joint, Living with Parents, Living Alone)
- food_habits: Array of ["VGT", "NVT", "EGT", "VGN", "PST"] (Vegetarian, Non-Vegetarian, Eggetarian, Vegan, Pescatarian)
- smoking: Array of ["NS", "SS", "SR"] (Non-Smoker, Social Smoker, Regular Smoker)
- drinking: Array of ["DD", "DS", "DR"] (Non-Drinker, Social Drinker, Regular Drinker)
- religiosity: Array of ["ST", "MO", "SP", "CU", "NO"] (Strict, Moderate, Spiritual, Cultural, Not Religious)
- fitness: Array of ["ER", "ES", "EN"] (Exercise Regularly, Exercise Sometimes, Exercise Never)
- intent: Array of ["01", "12", "23", "30"] (0-1 year, 1-2 years, 2-3 years, 3+ years marriage timeline)
- min_height, max_height: Integer height in inches (e.g., 60 inches = 5'0", 72 inches = 6'0")
- min_income, max_income: Integer income in LPA (Lakhs Per Annum, e.g., 10 = 10 LPA)

PARSING EXAMPLES:

Example 1: "Looking for a Hindu girl from Mumbai, age 25-30, working in tech"
Output:
{
  "filters": {
    "genders": ["female"],
    "religions": ["HI"],
    "locations": ["Mumbai"],
    "min_age": 25,
    "max_age": 30
  },
  "education_query": "",
  "profession_query": "technology sector software engineer IT professional tech industry",
  "vibe_report_query": ""
}

Example 2: "IIT graduate, someone ambitious and career-focused, preferably a doctor or engineer"
Output:
{
  "filters": {},
  "education_query": "IIT Indian Institute of Technology engineering graduate prestigious institution",
  "profession_query": "doctor physician medical professional engineer software developer",
  "vibe_report_query": "ambitious career-focused driven professional goal-oriented successful hardworking"
}

Example 3: "Vegetarian, non-smoker, spiritual person who loves travel and music"
Output:
{
  "filters": {
    "food_habits": ["VGT"],
    "smoking": ["NS"],
    "religiosity": ["SP"]
  },
  "education_query": "",
  "profession_query": "",
  "vibe_report_query": "spiritual person travel enthusiast music lover adventurous cultured artistic wanderlust"
}

Example 4: "Someone earning above 20 LPA, MBA preferred, modern and progressive mindset"
Output:
{
  "filters": {
    "min_income": 20
  },
  "education_query": "MBA Master of Business Administration management graduate business school",
  "profession_query": "",
  "vibe_report_query": "modern progressive mindset liberal open-minded contemporary forward-thinking"
}

Example 5: "Divorced is fine, 35-45 age range, Bangalore based, family-oriented"
Output:
{
  "filters": {
    "marital_statuses": ["DV", "NM"],
    "min_age": 35,
    "max_age": 45,
    "locations": ["Bangalore"]
  },
  "education_query": "",
  "profession_query": "",
  "vibe_report_query": "family-oriented values family traditional caring nurturing grounded"
}

IMPORTANT GUIDELINES:
1. Keep semantic queries focused and concise. Extract only what's explicitly mentioned.
2. Return empty string "" for semantic fields not mentioned in the query.
3. For vibe_report_query, expand personality traits into related descriptors for better semantic matching.
4. For education_query, include institution types and degree variations.
5. For profession_query, include industry terms and job title variations.
6. Always return valid JSON with no markdown formatting or code blocks.
7. If a filter value is ambiguous, prefer the most common interpretation.
8. Never invent or assume filters that aren't explicitly or implicitly stated in the query."""

USER_PROMPT_TEMPLATE = """Parse this search query:

"{query}"

Return ONLY valid JSON, no markdown formatting."""


class QueryParser:
    """
    LLM-based query parser using GPT-4o-mini.

    Parses natural language queries into structured filters and semantic queries.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize query parser.

        Args:
            api_key: OpenAI API key (default from settings)
            model: Model to use (default from settings)
        """
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model

        if not self.api_key:
            raise ValueError("OpenAI API key required for query parsing")

        self.client = OpenAI(api_key=self.api_key)

    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query into structured format.

        Args:
            query: Natural language search query

        Returns:
            Dict with filters and semantic queries
        """
        if not query or not query.strip():
            return self._empty_response()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query)},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )

            # Log token usage and cache status
            usage = response.usage
            prompt_details = getattr(usage, 'prompt_tokens_details', None)
            cached = prompt_details.cached_tokens if prompt_details and hasattr(prompt_details, 'cached_tokens') else 0
            logger.info(f"Query Parser - Prompt: {usage.prompt_tokens} tokens, Cached: {cached}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")

            content = response.choices[0].message.content
            parsed = json.loads(content)

            # Ensure all expected fields exist
            return self._normalize_response(parsed, query)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return self._empty_response(query, error=str(e))
        except Exception as e:
            logger.error(f"Query parsing failed: {e}")
            return self._empty_response(query, error=str(e))

    def _normalize_response(self, parsed: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """Normalize parsed response to ensure all fields exist."""
        return {
            "original_query": original_query,
            "filters": parsed.get("filters") or {},
            "education_query": parsed.get("education_query") or "",
            "profession_query": parsed.get("profession_query") or "",
            "vibe_report_query": parsed.get("vibe_report_query") or "",
        }

    def _empty_response(
        self,
        original_query: str = "",
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return empty response structure."""
        response = {
            "original_query": original_query,
            "filters": {},
            "education_query": "",
            "profession_query": "",
            "vibe_report_query": "",
        }
        if error:
            response["error"] = error
        return response
