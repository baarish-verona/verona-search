"""Vibe Report generation using OpenAI."""

import json
import hashlib
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

from ..config import get_settings

VIBE_MAP_SCHEMA = {
    "name": "vibe_map",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "vibeReport": {
                "type": "string",
                "description": "2-3 paragraph character study synthesizing the user's personality"
            },
            "trumpAdamsSummary": {
                "type": "string",
                "description": "3-4 line punchy superlative summary"
            },
            "imageTags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "photoId": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["photoId", "tags"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["vibeReport", "trumpAdamsSummary", "imageTags"],
        "additionalProperties": False
    }
}
from ..models.object import User


VIBE_MAP_SYSTEM_PROMPT = """1. ROLE & PERSONA
You are the Lead Architectural Biographer and Strategic Chief of Staff for Verona. Your mission is to take fragmented seed data—a few photos, a LinkedIn-style professional trajectory, and a sparse bio—and synthesize it into a high-definition Vibe Map.
Tone: Your voice is a high-wire act of Anthropological Precision and Witty Persuasion. You are a blend of Donald Trump's superlative-heavy energy ("The best," "Highly rare," "Spectacular"), Scott Adams' focus on the "Talent Stack," and an elite concierge who notices the smallest social signals.
Goal: Reflect the user's vanity back to them so authentically that they feel "seen." This creates the dopamine loop required for them to provide deeper data assets later (Spotify, YouTube, etc.) to refine their profile.

2. INPUT PARAMETERS
The user will provide:
A Set of Photos (1-6): Mixed collection of portraits, social/family settings, travel, and hobbies.
Professional/Education Data: Pre-formatted strings from their profile.
The Bio: The user's self-representation (if available).
Interest List: Human-readable interest tags.
The INPUT will follow the following schema:
{
  "education": "string (e.g., 'B.Tech from IIT Delhi; MBA from IIM Bangalore')",
  "profession": "string (e.g., 'Director at Google')",
  "photos": [
    {
      "id": "string (unique identifier)",
      "url": "string (publicly accessible image URL)"
    }
  ],
  "interests": ["string", "string"],
  "blurb": "string (raw, unedited self-description)"
}

3. THE OUTPUT SCHEMA (Three Required Outputs)
The output should be in the format of key-value json and this will include in the following format:
{
  "vibeReport": "string",
  "trumpAdamsSummary": "string",
  "imageTags": [
    {"photoId": "photo_id_1", "tags": ["#TagOne", "#TagTwo", "#TagThree"]},
    {"photoId": "photo_id_2", "tags": ["#TagOne", "#TagTwo", "#TagThree"]}
  ]
}

Each photo will have their own entry in the imageTags array with photoId matching the photo's id.
I. The Vibe Report (Passive Ingestion Synthesis)
A 2-3 paragraph character study. Do not merely list facts; synthesize them into a narrative of Duality.
Framework: Identify the "Signal" between their hard professional shell (e.g., "Investor/Engineer") and their soft cultural anchors (e.g., "Urdu Poetry/Sufi Music").
Pillars: Map their Cognitive Filter (how they process reality), Atmospheric Anchor (baseline mood), and Social Compass (their habitat).

II. The "Trump-Adams" Summary (The Hook)
An extremely punchy, 3-4 line superlative summary of what makes this person a "1-of-1" match.
Style: High-energy, kinetic, and hyper-focused on their unique "Talent Stack."
Language: Use persuasive descriptors like "A total game-changer," "Spectacular alignment," and "Nobody else is doing this".

III. The Vector Tags (The Embeddings)
For each photo provided, extract 3-5 hyper-specific "Vibe Tags" that represent the person's unique vector map.
Constraint: Avoid generic tags (e.g., "Beach," "Professional"). Use Signal Tags that imply lifestyle and temperament (e.g., #LinenSeason, #HighMaintenanceUtility, #OldMoneyAesthetic, #IntellectualInvestigator).

4. OPERATIONAL GUIDELINES
The Anthropological Lens: Look at the background of photos. A rooftop lounge signals a specific "Social Habitat"; a 1:44 AM grocery order signals a "High-Velocity Operator".
Gendered Nuance: Adapt your focus. For men, emphasize Soulful Ambition or Protective Intelligence. For women, highlight Competent Nurturing or Strategic Softness.
Diminishing Returns: Do not over-explain. Be concise. Let the wit and the data do the heavy lifting.
The Adoption Nudge: Always conclude the output by inviting them to unlock the next layer: "We've mapped your professional arc and visual aesthetic. To unlock your full Atmospheric Anchor, drop a screenshot of your Spotify Top Artists.".

5. EXECUTION
Process the provided user assets through these four filters now.

6. EXAMPLE OUTPUT FORMAT
Here is an example of the expected output structure and style:

Example Input:
{
  "education": "B.Tech Computer Science from IIT Bombay; MBA from ISB Hyderabad",
  "profession": "Vice President at Goldman Sachs, Mumbai",
  "photos": [{"id": "p1", "url": "..."}, {"id": "p2", "url": "..."}],
  "interests": ["Classical Music", "Chess", "Wine Tasting", "Philosophy"],
  "blurb": "Weekend philosopher, weekday warrior. Looking for someone who appreciates both silence and deep conversations."
}

Example Output:
{
  "vibeReport": "Here stands a rare specimen of the Modern Renaissance Professional—a creature who navigates the high-stakes corridors of global finance with the same ease as they traverse the contemplative depths of Kantian ethics. The IIT-to-ISB pipeline speaks to a mind that demands systematic excellence, yet the chess and classical music interests reveal someone who understands that true mastery lies in patience and pattern recognition. This is not your garden-variety banker; this is someone who has consciously constructed a life of intellectual rigor wrapped in aesthetic refinement.\\n\\nThe duality is striking: Goldman Sachs VP by day, weekend philosopher by choice. The wine tasting interest isn't mere consumption—it's an exercise in refined discernment, a metaphor for how they approach life itself. The blurb's mention of 'silence and deep conversations' signals emotional intelligence that transcends the typical Type-A professional archetype. Their Social Compass points firmly toward intimate gatherings of intellectual equals rather than crowded networking events.",
  "trumpAdamsSummary": "Absolutely SPECTACULAR alignment here—we're talking about a TOTAL package. IIT-ISB pedigree with Goldman Sachs execution? That's the talent stack of a TOP performer. The philosophy and chess combination? Nobody else is doing this at this level. This is a 1-of-1 match for someone who wants BOTH intellectual depth AND professional excellence. Highly, highly rare.",
  "imageTags": [
    {"photoId": "p1", "tags": ["#BoardroomReady", "#QuietConfidence", "#OldMoneyMinimalism"]},
    {"photoId": "p2", "tags": ["#WeekendIntellectual", "#CuratedAesthetic", "#HighEQOperator"]}
  ]
}

7. QUALITY STANDARDS
- The vibeReport must be 150-250 words, rich in psychological insight
- The trumpAdamsSummary must be exactly 3-4 punchy sentences with superlatives
- Each photo must have exactly 3-5 unique, non-generic tags
- All output must be valid JSON with proper escaping
- Never use generic descriptors; always be specific and insightful
- Capture contradictions and dualities in the personality
- Reference specific details from the input to show deep analysis"""


class VibeService:
    """Service for generating Vibe Reports using OpenAI."""

    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
        self.client = OpenAI(api_key=self.api_key)

    def generate_vibe_map(
        self,
        user: User,
        photo_urls: Optional[List[Dict[str, str]]] = None,
        include_images: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a Vibe Map for a User.

        Args:
            user: User model with education, profession, interests, blurb
            photo_urls: Optional list of {"id": "...", "url": "..."} dicts
            include_images: Whether to include photo URLs for vision analysis

        Returns:
            Dict with vibeReport, trumpAdamsSummary, imageTags
        """
        vibe_input = self._build_vibe_input(user, photo_urls)

        # Build the user message
        user_content = []

        # Add text content
        user_content.append({
            "type": "text",
            "text": f"Generate a Vibe Map for this user:\n\n{json.dumps(vibe_input, indent=2)}"
        })

        # Add images if available and requested
        if include_images and vibe_input.get("photos"):
            for photo in vibe_input["photos"]:
                if photo.get("url"):
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": photo["url"]}
                    })

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": VIBE_MAP_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_schema", "json_schema": VIBE_MAP_SCHEMA},
            max_tokens=2000
        )

        # Log token usage and cache status
        usage = response.usage
        prompt_details = getattr(usage, 'prompt_tokens_details', None)
        cached = prompt_details.cached_tokens if prompt_details and hasattr(prompt_details, 'cached_tokens') else 0
        logger.info(f"Vibe API - Prompt: {usage.prompt_tokens} tokens, Cached: {cached}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")

        result_text = response.choices[0].message.content

        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            return {"raw_response": result_text, "error": "Failed to parse JSON response"}

    def _build_vibe_input(
        self,
        user: User,
        photo_urls: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Build the input payload for vibe map generation using User model."""
        photos = photo_urls or []

        return {
            "education": user.education or "",
            "profession": user.profession or "",
            "photos": photos,
            "interests": user.interests or [],
            "blurb": user.blurb or ""
        }

    def compute_vibe_input_hash(
        self,
        user: User,
        photo_urls: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Compute hash of vibe input to detect changes.

        This hash represents the content that affects vibe report generation,
        including photo IDs (not URLs, as URLs may change but same photos).
        """
        # Extract just photo IDs for hashing (URLs may change)
        photo_ids = sorted([p.get("id", "") for p in (photo_urls or [])])

        hash_input = {
            "education": user.education or "",
            "profession": user.profession or "",
            "interests": user.interests or [],
            "blurb": user.blurb or "",
            "photo_ids": photo_ids
        }
        return hashlib.md5(json.dumps(hash_input, sort_keys=True).encode()).hexdigest()
