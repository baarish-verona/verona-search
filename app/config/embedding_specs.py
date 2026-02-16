"""
Hardcoded embedding configuration for OpenAI + ColBERT.

Uses OpenAI embeddings for structured fields (education, profession)
and BGE-M3 ColBERT for vibe_report (late interaction).
"""

# Vector name → provider + dimensions + type
VECTOR_CONFIG = {
    # OpenAI vectors for structured fields
    "education": {"provider": "openai-small", "dim": 1536, "type": "dense"},
    "profession": {"provider": "openai-small", "dim": 1536, "type": "dense"},

    # BGE-M3 ColBERT for vibe_report (late interaction)
    "vibe_report": {"provider": "bge-colbert", "dim": 1024, "type": "multivector"},
}

# Logical field → source text field mapping (from profile)
SOURCE_FIELDS = {
    "education": "education_text",
    "profession": "profession_text",
    "vibe_report": "vibe_report",
}


def get_vector_config(vector_name: str) -> dict:
    """Get configuration for a specific vector."""
    if vector_name not in VECTOR_CONFIG:
        raise ValueError(f"Unknown vector: {vector_name}")
    return VECTOR_CONFIG[vector_name]


def get_required_providers() -> set:
    """Get the set of provider names required."""
    providers = set()
    for config in VECTOR_CONFIG.values():
        providers.add(config["provider"])
    return providers