#!/usr/bin/env python3
"""
Profile ingestion script for OpenAI + ColBERT.

Usage:
    python -m scripts.ingest_profiles --file data/profiles.json
    python -m scripts.ingest_profiles --file data/profiles.json --recreate
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.config.embedding_specs import VECTOR_CONFIG, SOURCE_FIELDS, get_required_providers
from app.embeddings import EmbeddingProviderFactory
from app.mappers import ProfileMapper
from app.vector_store import QdrantVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_profiles(file_path: str) -> List[Dict[str, Any]]:
    """Load profiles from JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {file_path}")

    with open(path, "r") as f:
        data = json.load(f)

    # Handle both list and dict with "profiles" key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "profiles" in data:
        return data["profiles"]
    else:
        raise ValueError("Invalid profile file format")


def generate_embeddings_for_profile(
    profile: Dict[str, Any],
    providers: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate embeddings for a single profile."""
    vectors = {}

    for vector_name, config in VECTOR_CONFIG.items():
        # Get source text field
        source_field = SOURCE_FIELDS.get(vector_name)
        if not source_field:
            continue

        text = profile.get(source_field, "")
        if not text:
            continue

        # Get provider
        provider = providers.get(config["provider"])

        if provider:
            embedding = provider.embed(text)
            vectors[vector_name] = embedding

    return vectors


def ingest_profiles(
    profiles: List[Dict[str, Any]],
    vector_store: QdrantVectorStore,
    batch_size: int = 50,
) -> int:
    """Ingest profiles with OpenAI + ColBERT embeddings."""
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

    # Process profiles in batches
    total_ingested = 0
    total_profiles = len(profiles)

    for batch_start in range(0, total_profiles, batch_size):
        batch_end = min(batch_start + batch_size, total_profiles)
        batch = profiles[batch_start:batch_end]

        logger.info(f"Processing batch {batch_start + 1}-{batch_end} of {total_profiles}")

        points = []
        for profile in batch:
            # Validate profile
            errors = ProfileMapper.validate_profile(profile)
            if errors:
                logger.warning(f"Skipping invalid profile: {errors}")
                continue

            # Generate embeddings
            vectors = generate_embeddings_for_profile(profile, providers)

            if not vectors:
                logger.warning(f"No vectors generated for profile {profile.get('user_id')}")
                continue

            # Convert to Qdrant point
            point = ProfileMapper.to_qdrant_point(profile, vectors)
            points.append(point.to_dict())

        # Upsert batch
        if points:
            vector_store.upsert_points(points)
            total_ingested += len(points)
            logger.info(f"  Ingested {len(points)} profiles")

    return total_ingested


def main():
    parser = argparse.ArgumentParser(description="Ingest profiles into Qdrant")
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to profiles JSON file"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection before ingesting"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)"
    )

    args = parser.parse_args()

    # Load settings
    settings = get_settings()

    logger.info(f"Ingesting profiles from: {args.file}")
    logger.info(f"Using OpenAI + ColBERT embeddings")
    logger.info(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")

    # Load profiles
    try:
        profiles = load_profiles(args.file)
        logger.info(f"Loaded {len(profiles)} profiles")
    except Exception as e:
        logger.error(f"Failed to load profiles: {e}")
        sys.exit(1)

    # Initialize vector store
    vector_store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection,
    )

    # Create/recreate collection
    if args.recreate:
        logger.info("Recreating collection...")
        vector_store.create_collection(recreate=True)
    else:
        vector_store.create_collection(recreate=False)

    # Ingest profiles
    start_time = time.time()
    try:
        count = ingest_profiles(
            profiles,
            vector_store,
            batch_size=args.batch_size,
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info(f"Ingestion complete: {count} profiles in {elapsed:.2f}s")
    logger.info(f"Collection info: {vector_store.collection_info()}")


if __name__ == "__main__":
    main()
