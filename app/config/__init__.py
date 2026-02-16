"""Configuration modules for Verona AI Search."""

from .settings import Settings, get_settings
from .embedding_specs import VECTOR_CONFIG, SOURCE_FIELDS, get_required_providers

__all__ = [
    "Settings",
    "get_settings",
    "VECTOR_CONFIG",
    "SOURCE_FIELDS",
    "get_required_providers",
]
