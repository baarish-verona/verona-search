"""Embedding providers for Verona AI Search."""

from .base import EmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider
from .factory import EmbeddingProviderFactory

# BGEColBERTProvider is imported lazily in factory to avoid transformers dependency issues

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "EmbeddingProviderFactory",
]
