"""OpenAI embedding provider for structured fields."""

import logging
from typing import List, Optional

from openai import OpenAI

from ..config import get_settings
from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text-embedding-3-small provider (1536 dimensions)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        device: Optional[str] = None,
    ):
        """
        Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key (default from settings)
            model: Embedding model to use
            device: Ignored (OpenAI runs in cloud)
        """
        settings = get_settings()
        self._api_key = api_key or settings.openai_api_key

        if not self._api_key:
            raise ValueError("OpenAI API key required for embeddings")

        self._client = OpenAI(api_key=self._api_key)
        self._model = model
        self._dimensions = 1536

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def is_late_interaction(self) -> bool:
        return False

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            return [0.0] * self._dimensions

        response = self._client.embeddings.create(
            input=text,
            model=self._model,
        )
        # Log token usage
        logger.info(f"Embeddings API - Total tokens: {response.usage.total_tokens}")
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        # Handle empty strings
        non_empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        # Initialize results with zero vectors
        results = [[0.0] * self._dimensions for _ in range(len(texts))]

        if non_empty_texts:
            response = self._client.embeddings.create(
                input=non_empty_texts,
                model=self._model,
            )
            # Log token usage
            logger.info(f"Embeddings API - Total tokens: {response.usage.total_tokens}")
            for idx, embedding_data in zip(non_empty_indices, response.data):
                results[idx] = embedding_data.embedding

        return results