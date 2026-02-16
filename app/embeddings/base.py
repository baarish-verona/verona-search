"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import List, Union


class EmbeddingProvider(ABC):
    """Base interface for all embedding providers."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the model identifier."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        pass

    @property
    @abstractmethod
    def is_late_interaction(self) -> bool:
        """Return True if this produces multi-vectors (ColBERT-style)."""
        pass

    @abstractmethod
    def embed(self, text: str) -> Union[List[float], List[List[float]]]:
        """
        Generate embedding for a single text.

        Returns:
            - Dense: List[float] of length `dimensions`
            - ColBERT: List[List[float]] of shape [num_tokens, dimensions]
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[Union[List[float], List[List[float]]]]:
        """Generate embeddings for multiple texts."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_id}, dim={self.dimensions})"
