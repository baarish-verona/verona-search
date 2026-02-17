"""LLM-based query parser using GPT-4o-mini."""

import json
import logging
from typing import Any, Dict, Optional

from openai import OpenAI

from ..config import get_settings

logger = logging.getLogger(__name__)

# Strict JSON schema for query parsing output
QUERY_PARSER_SCHEMA = {
    "name": "parsed_query",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "filters": {
                "type": "object",
                "description": "Hard filters for search",
                "properties": {
                    "min_age": {"type": ["integer", "null"], "description": "Minimum age"},
                    "max_age": {"type": ["integer", "null"], "description": "Maximum age"},
                    "min_height": {"type": ["integer", "null"], "description": "Minimum height in INCHES (convert cm to inches if needed: divide by 2.54)"},
                    "max_height": {"type": ["integer", "null"], "description": "Maximum height in INCHES"},
                    "min_income": {"type": ["integer", "null"], "description": "Minimum income in LPA"},
                    "max_income": {"type": ["integer", "null"], "description": "Maximum income in LPA"},
                    "genders": {"type": ["array", "null"], "items": {"type": "string", "enum": ["male", "female"]}},
                    "religions": {"type": ["array", "null"], "items": {"type": "string", "enum": ["HI", "MU", "CR", "SI", "JA", "BU", "PA", "JE", "BA", "NR"]}},
                    "locations": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Location codes like IN_MB (Mumbai), IN_DEL (Delhi), IN_BLR (Bangalore). Use country prefix + city code."},
                    "marital_statuses": {"type": ["array", "null"], "items": {"type": "string", "enum": ["NM", "DV", "WD", "AN"]}},
                    "family_types": {"type": ["array", "null"], "items": {"type": "string", "enum": ["NU", "JF", "LP", "LT"]}},
                    "food_habits": {"type": ["array", "null"], "items": {"type": "string", "enum": ["VGT", "NVT", "EGT", "VGN", "PST"]}},
                    "smoking": {"type": ["array", "null"], "items": {"type": "string", "enum": ["NS", "SS", "SR"]}},
                    "drinking": {"type": ["array", "null"], "items": {"type": "string", "enum": ["DD", "DS", "DR"]}},
                    "religiosity": {"type": ["array", "null"], "items": {"type": "string", "enum": ["ST", "MO", "SP", "CU", "NO"]}},
                    "fitness": {"type": ["array", "null"], "items": {"type": "string", "enum": ["ER", "ES", "EN"]}},
                    "intent": {"type": ["array", "null"], "items": {"type": "string", "enum": ["01", "12", "23", "30"]}}
                },
                "required": ["min_age", "max_age", "min_height", "max_height", "min_income", "max_income", "genders", "religions", "locations", "marital_statuses", "family_types", "food_habits", "smoking", "drinking", "religiosity", "fitness", "intent"],
                "additionalProperties": False
            },
            "education_query": {
                "type": "string",
                "description": "Education keywords exactly as mentioned. No expansion. Empty string if not mentioned."
            },
            "profession_query": {
                "type": "string",
                "description": "Profession keywords exactly as mentioned. No expansion. Empty string if not mentioned."
            },
            "vibe_report_query": {
                "type": "string",
                "description": "Personality traits and hobbies/interests exactly as mentioned. No fluff or expansion. Empty string if not mentioned."
            }
        },
        "required": ["filters", "education_query", "profession_query", "vibe_report_query"],
        "additionalProperties": False
    }
}

SYSTEM_PROMPT = """You are a query parser for a matrimonial profile search system.
Extract structured information from natural language queries.

FILTER CODES:
- religions: HI=Hindu, MU=Muslim, CR=Christian, SI=Sikh, JA=Jain, BU=Buddhist, PA=Parsi, JE=Jewish, BA=Bahai, NR=No Religion
- marital_statuses: NM=Never Married, DV=Divorced, WD=Widowed, AN=Annulled
- family_types: NU=Nuclear, JF=Joint, LP=Living with Parents, LT=Living Alone
- food_habits: VGT=Vegetarian, NVT=Non-Vegetarian, EGT=Eggetarian, VGN=Vegan, PST=Pescatarian
- smoking: NS=Non-Smoker, SS=Social Smoker, SR=Regular Smoker
- drinking: DD=Non-Drinker, DS=Social Drinker, DR=Regular Drinker
- religiosity: ST=Strict, MO=Moderate, SP=Spiritual, CU=Cultural, NO=Not Religious
- fitness: ER=Exercise Regularly, ES=Exercise Sometimes, EN=Exercise Never
- intent: 01=0-1 year, 12=1-2 years, 23=2-3 years, 30=3+ years marriage timeline

LOCATION CODES (use these exact codes):
India: IN_MB=Mumbai, IN_DEL=Delhi, IN_BLR=Bangalore, IN_HYD=Hyderabad, IN_CHE=Chennai, IN_KOL=Kolkata, IN_PUN=Pune, IN_AHM=Ahmedabad, IN_JAI=Jaipur, IN_LKO=Lucknow, IN_GUR=Gurugram, IN_NOI=Noida, IN_CHD=Chandigarh, IN_IND=Indore, IN_NAG=Nagpur, IN_COI=Coimbatore, IN_KOC=Kochi, IN_THI=Thiruvananthapuram, IN_VIZ=Visakhapatnam, IN_VAD=Vadodara, IN_SUR=Surat, IN_LUD=Ludhiana, IN_MYS=Mysore
USA: US_NYC=New York, US_LA=Los Angeles, US_SF=San Francisco, US_CHI=Chicago, US_SEA=Seattle, US_BOS=Boston, US_WAS=Washington DC, US_HOU=Houston, US_DAL=Dallas, US_ATL=Atlanta, US_DEN=Denver, US_PHX=Phoenix, US_SD=San Diego, US_SJ=San Jose, US_AUS=Austin
UK: UK_LON=London, UK_MAN=Manchester, UK_BIR=Birmingham, UK_EDI=Edinburgh, UK_GLA=Glasgow, UK_LEE=Leeds, UK_CAM=Cambridge, UK_OXF=Oxford
Canada: CA_TOR=Toronto, CA_VAN=Vancouver, CA_MON=Montreal, CA_CAL=Calgary, CA_OTT=Ottawa
UAE: AE_DXB=Dubai, AE_AUH=Abu Dhabi
Singapore: SG_SG=Singapore
Australia: AU_SYD=Sydney, AU_MEL=Melbourne, AU_BRI=Brisbane, AU_PER=Perth
Germany: DE_BER=Berlin, DE_MUN=Munich, DE_FRA=Frankfurt
Switzerland: CH_ZUR=Zurich, CH_GN=Geneva

HEIGHT CONVERSION (CRITICAL):
- If height >= 100 (like 150, 160, 170), it's in cm - convert to inches by dividing by 2.54
- 150 cm = 59 inches, 160 cm = 63 inches, 170 cm = 67 inches, 180 cm = 71 inches
- 5'0" = 60 inches, 5'6" = 66 inches, 6'0" = 72 inches

SEMANTIC QUERY RULES:
1. education_query: Extract exactly what's mentioned, no expansion (e.g., "IIT Graduate" → "IIT Graduate")
2. profession_query: Extract exactly what's mentioned, no expansion (e.g., "Software Engineer" → "Software Engineer")
3. vibe_report_query: Extract hobbies/interests/personality exactly as mentioned, no fluff
   - "loves guitar music and hiking" → "guitar music hiking"
   - "ambitious and caring" → "ambitious caring"
   - Keep it straightforward, no added words

EXAMPLES:

Query: "IIT graduate software engineer age 25-32 loves guitar and hiking height atleast 150"
Output:
- filters: {min_age: 25, max_age: 32, min_height: 59}  (150cm = 59 inches)
- education_query: "IIT graduate"
- profession_query: "software engineer"
- vibe_report_query: "guitar hiking"

Query: "Doctor from Mumbai, caring and empathetic person, height 5'6 to 6'"
Output:
- filters: {locations: ["IN_MB"], min_height: 66, max_height: 72}
- education_query: ""
- profession_query: "doctor"
- vibe_report_query: "caring empathetic"

Query: "CA or MBA, vegetarian, modern progressive mindset"
Output:
- filters: {food_habits: ["VGT"]}
- education_query: "CA MBA"
- profession_query: ""
- vibe_report_query: "modern progressive"

Query: "Hindu girl from Delhi, age 28-35, loves travel and photography"
Output:
- filters: {genders: ["female"], religions: ["HI"], locations: ["IN_DEL"], min_age: 28, max_age: 35}
- education_query: ""
- profession_query: ""
- vibe_report_query: "travel photography" """

USER_PROMPT_TEMPLATE = """Parse this search query: "{query}" """


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
                response_format={"type": "json_schema", "json_schema": QUERY_PARSER_SCHEMA},
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
        """Normalize parsed response to ensure all fields exist and filter out nulls."""
        # Filter out null values from filters
        raw_filters = parsed.get("filters") or {}
        filters = {k: v for k, v in raw_filters.items() if v is not None}

        return {
            "original_query": original_query,
            "filters": filters,
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
