"""Deterministic submission ID generation for idempotent job submission.

Ray Jobs API supports a ``submission_id`` parameter. Submitting a job with
an existing ``submission_id`` returns the existing job ID instead of creating
a duplicate job. This module generates deterministic IDs from job parameters.
"""

from __future__ import annotations

import hashlib
import json


def generate_submission_id(prefix: str, *components: str) -> str:
    """Generate a deterministic submission ID.

    The ID is derived from a SHA-256 hash of the concatenated components,
    prefixed with ``tributo-{prefix}-``. This ensures:

    - The same inputs always produce the same ID (idempotency).
    - Different inputs produce different IDs (uniqueness).
    - The ID length stays well below Ray's 64-character limit.

    Args:
        prefix: Short prefix indicating the job type, e.g. ``"embed"`` or
            ``"train"``.
        *components: String components that uniquely identify the job.

    Returns:
        A deterministic submission ID string.

    Example:
        >>> generate_submission_id("embed", "bge-small-zh", "s3://in", "s3://out", "64", "4")
        'tributo-embed-a1b2c3d4e5f67890'
    """
    payload = "|".join(components)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"tributo-{prefix}-{digest}"


def generate_training_submission_id(
    run_id: str, attempt_id: str, entrypoint: str
) -> str:
    """Generate the stable Ray identity for one business training attempt."""
    canonical = json.dumps(
        [run_id, attempt_id, entrypoint],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    # Feed the legacy generator one canonical component. Its delimiter-based
    # format remains unchanged for all existing callers, while training
    # identity components cannot collide when they contain ``|``.
    return generate_submission_id("train", canonical)
