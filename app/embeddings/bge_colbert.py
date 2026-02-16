"""BGE-M3 ColBERT (late interaction) provider for vibe_report field."""

import os
from typing import List

from FlagEmbedding import BGEM3FlagModel

from .base import EmbeddingProvider

# Model ID constant
BGE_M3_MODEL_ID = "BAAI/bge-m3"


class BGEColBERTProvider(EmbeddingProvider):
    """BGE-M3 ColBERT provider for multi-vector late interaction."""

    def __init__(self, device: str = "cpu"):
        """
        Initialize BGE-M3 ColBERT provider.

        Uses HF_HOME environment variable for model cache location.

        Args:
            device: Device to use (cpu, cuda, mps)
        """
        # Ensure cache directory is set (for Docker)
        cache_dir = os.environ.get("HF_HOME", os.environ.get("TRANSFORMERS_CACHE"))

        self._model = BGEM3FlagModel(BGE_M3_MODEL_ID, use_fp16=True, device=device)
        self._dimensions = 1024
        self._device = device

    @property
    def model_id(self) -> str:
        return "BAAI/bge-m3-colbert"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def is_late_interaction(self) -> bool:
        return True

    def embed(self, text: str) -> List[List[float]]:
        """
        Generate ColBERT multi-vector embedding for a single text.

        Returns:
            List[List[float]] of shape [num_tokens, 1024]
        """
        if not text or not text.strip():
            # Return single zero vector for empty text
            return [[0.0] * self._dimensions]

        result = self._model.encode(
            [text],
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True
        )
        return result["colbert_vecs"][0].tolist()

    def embed_batch(self, texts: List[str]) -> List[List[List[float]]]:
        """Generate ColBERT embeddings for multiple texts."""
        if not texts:
            return []

        # Handle empty strings
        non_empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        # Initialize results with single zero vectors
        results = [[[0.0] * self._dimensions] for _ in range(len(texts))]

        if non_empty_texts:
            embeddings = self._model.encode(
                non_empty_texts,
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True
            )
            for idx, embedding in zip(non_empty_indices, embeddings["colbert_vecs"]):
                results[idx] = embedding.tolist()

        return results
