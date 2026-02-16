"""
Migration script to add computed 'age' field to existing Qdrant profiles.

Usage:
    python -m scripts.migrate_add_age [--dry-run] [--batch-size 100]
"""

import argparse
import logging
from datetime import datetime
from typing import Optional

from qdrant_client import QdrantClient

from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_age(dob: str) -> Optional[int]:
    """Compute age from date of birth string (YYYY-MM-DD format)."""
    try:
        birth_date = datetime.strptime(dob, "%Y-%m-%d")
        today = datetime.now()
        age = today.year - birth_date.year
        if (today.month, today.day) < (birth_date.month, birth_date.day):
            age -= 1
        return age
    except (ValueError, TypeError):
        return None


def migrate_age_field(dry_run: bool = False, batch_size: int = 100):
    """Add computed age field to all profiles missing it."""
    settings = get_settings()

    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    collection_name = settings.qdrant_collection

    logger.info(f"Connected to Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Dry run: {dry_run}")

    # Get collection info
    info = client.get_collection(collection_name)
    total_points = info.points_count
    logger.info(f"Total profiles in collection: {total_points}")

    # Scroll through all points
    updated_count = 0
    skipped_count = 0
    error_count = 0
    offset = None

    while True:
        # Fetch batch of points
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not records:
            break

        for record in records:
            payload = record.payload or {}
            dob = payload.get("dob")
            existing_age = payload.get("age")

            # Skip if age already exists
            if existing_age is not None:
                skipped_count += 1
                continue

            # Skip if no dob
            if not dob:
                logger.warning(f"No dob for point {record.id}")
                error_count += 1
                continue

            # Compute age
            age = compute_age(dob)
            if age is None:
                logger.warning(f"Could not compute age from dob '{dob}' for point {record.id}")
                error_count += 1
                continue

            # Update payload
            if not dry_run:
                client.set_payload(
                    collection_name=collection_name,
                    payload={"age": age},
                    points=[record.id],
                )

            updated_count += 1

            if updated_count % 500 == 0:
                logger.info(f"Progress: {updated_count} updated, {skipped_count} skipped, {error_count} errors")

        if offset is None:
            break

    logger.info("=" * 50)
    logger.info("Migration complete!")
    logger.info(f"  Updated: {updated_count}")
    logger.info(f"  Skipped (already had age): {skipped_count}")
    logger.info(f"  Errors: {error_count}")
    if dry_run:
        logger.info("  (DRY RUN - no changes made)")


def main():
    parser = argparse.ArgumentParser(description="Add age field to existing Qdrant profiles")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for scrolling")
    args = parser.parse_args()

    migrate_age_field(dry_run=args.dry_run, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
