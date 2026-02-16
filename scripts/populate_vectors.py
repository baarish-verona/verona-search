#!/usr/bin/env python3
"""
Regenerate vectors for existing profiles.

This script regenerates embeddings for existing profiles using
OpenAI + ColBERT. Useful for:
- Updating vectors after fixing embedding issues
- Regenerating vectors with new model versions

Usage:
    python -m scripts.populate_vectors
    python -m scripts.populate_vectors --batch-size 100
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.config.embedding_specs import VECTOR_CONFIG, SOURCE_FIELDS, get_required_providers
from app.embeddings import EmbeddingProviderFactory
from app.vector_store import QdrantVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def populate_vectors(
    vector_store: QdrantVectorStore,
    batch_size: int = 100,
) -> int:
    """Regenerate vectors for existing profiles."""
    settings = get_settings()
    device = settings.embedding_device

    # Load required providers
    providers_needed = get_required_providers()

    logger.info(f"Loading providers: {providers_needed}")
    providers = {}
    for provider_name in providers_needed:
        logger.info(f"  Loading {provider_name}...")
        providers[provider_name] = EmbeddingProviderFactory.get_provider(
            provider_name, device=device
        )

    # Scroll through all points and update vectors
    total_updated = 0
    offset = None

    while True:
        # Scroll through points
        records, offset = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not records:
            break

        logger.info(f"Processing batch of {len(records)} profiles...")

        for record in records:
            payload = record.payload or {}
            point_id = record.id

            # Generate vectors for this profile
            new_vectors = {}
            for vector_name, config in VECTOR_CONFIG.items():
                source_field = SOURCE_FIELDS.get(vector_name)
                if not source_field:
                    continue

                text = payload.get(source_field, "")
                if not text:
                    continue

                provider = providers.get(config["provider"])

                if provider:
                    embedding = provider.embed(text)
                    new_vectors[vector_name] = embedding

            # Update point with new vectors
            if new_vectors:
                vector_store.update_vectors(point_id, new_vectors)
                total_updated += 1

        if offset is None:
            break

    return total_updated


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate vectors for existing profiles"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)"
    )

    args = parser.parse_args()

    # Load settings
    settings = get_settings()

    logger.info(f"Regenerating vectors with OpenAI + ColBERT")
    logger.info(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")

    # Initialize vector store
    vector_store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection,
    )

    # Check collection exists
    try:
        info = vector_store.collection_info()
        logger.info(f"Collection: {info['name']} ({info['points_count']} points)")
    except Exception as e:
        logger.error(f"Collection not found: {e}")
        sys.exit(1)

    # Populate vectors
    start_time = time.time()
    try:
        count = populate_vectors(
            vector_store,
            batch_size=args.batch_size,
        )
    except Exception as e:
        logger.error(f"Population failed: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info(f"Regeneration complete: {count} profiles updated in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
