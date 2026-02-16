"""Factory for creating and managing embedding providers."""

from typing import Dict, Optional, Type

from .base import EmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider


class EmbeddingProviderFactory:
    """
    Factory for creating embedding providers (singleton pattern).

    Ensures providers are only instantiated once per configuration.
    """

    _providers: Dict[str, Type[EmbeddingProvider]] = {
        "openai-small": OpenAIEmbeddingProvider,
    }

    _instances: Dict[str, EmbeddingProvider] = {}
    _colbert_loaded: bool = False

    @classmethod
    def _load_colbert(cls) -> None:
        """Lazy load ColBERT provider to avoid transformers import at startup."""
        if not cls._colbert_loaded:
            from .bge_colbert import BGEColBERTProvider
            cls._providers["bge-colbert"] = BGEColBERTProvider
            cls._colbert_loaded = True

    @classmethod
    def get_provider(
        cls,
        provider_type: str,
        device: Optional[str] = None,
        **kwargs
    ) -> EmbeddingProvider:
        """
        Get or create a singleton provider instance.

        Args:
            provider_type: Type of provider (bge-colbert, openai-small)
            device: Device to use (cpu, cuda, mps)
            **kwargs: Additional provider arguments

        Returns:
            EmbeddingProvider instance
        """
        # Lazy load ColBERT if requested
        if provider_type == "bge-colbert":
            cls._load_colbert()

        # Create cache key including device
        cache_key = f"{provider_type}:{device or 'default'}"

        if cache_key not in cls._instances:
            if provider_type not in cls._providers:
                available = list(cls._providers.keys())
                raise ValueError(f"Unknown provider: {provider_type}. Available: {available}")

            provider_kwargs = kwargs.copy()
            if device:
                provider_kwargs["device"] = device

            cls._instances[cache_key] = cls._providers[provider_type](**provider_kwargs)

        return cls._instances[cache_key]

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[EmbeddingProvider]) -> None:
        """
        Register a custom provider (for extensibility).

        Args:
            name: Provider name
            provider_class: Provider class implementing EmbeddingProvider
        """
        cls._providers[name] = provider_class

    @classmethod
    def available_providers(cls) -> list:
        """Get list of available provider names."""
        return list(cls._providers.keys())

    @classmethod
    def clear_instances(cls) -> None:
        """Clear all cached provider instances (useful for testing)."""
        cls._instances.clear()

    @classmethod
    def is_loaded(cls, provider_type: str, device: Optional[str] = None) -> bool:
        """Check if a provider is already loaded."""
        cache_key = f"{provider_type}:{device or 'default'}"
        return cache_key in cls._instances
