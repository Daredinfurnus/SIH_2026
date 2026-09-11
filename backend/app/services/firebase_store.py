"""
Firebase cloud storage bridge for TraumaSense case persistence.

When Firebase is configured (FIREBASE_KEY_PATH set), cases are persisted to
Firestore.  When not configured, this module operates in in-memory fallback
mode — cases are stored only in the route-level in-memory store and lost on
restart.  The rest of the application does not need to branch on this.

This is a STUB implementation.  The Firebase-backed implementation is
planned but not required for the prototype.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory fallback store
# ---------------------------------------------------------------------------

_cases: dict[str, dict[str, Any]] = {}
_ready = False


def cloud_load_all() -> dict[str, dict[str, Any]]:
    """Load all cases from cloud storage.

    Returns:
        dict mapping case_id → case data dict.

        When Firebase is not configured, returns an empty dict.  The
        route-level in-memory store is the primary persistence layer in
        that case.
    """
    global _ready
    if not _ready:
        logger.debug("cloud_load_all: Firebase not configured, returning empty")
        return {}
    # Firebase-backed implementation would go here.
    # For now, return the in-memory store as a best-effort fallback.
    return dict(_cases)


def cloud_save(case_id: str, data: dict[str, Any]) -> None:
    """Save a case to cloud storage.

    Parameters:
        case_id: the case identifier (e.g. CASE-26093-0001).
        data: the serialized case data dict.

    When Firebase is not configured, this is a no-op (the route-level
    in-memory store handles persistence).  When Firebase is configured,
    the case is persisted to Firestore.
    """
    global _ready
    if not _ready:
        logger.debug("cloud_save: Firebase not configured, skipping persistence")
        return
    _cases[case_id] = data


def is_ready() -> bool:
    """Check whether cloud storage is configured and ready.

    Returns:
        True if Firebase is configured and connected, False otherwise.
    """
    return _ready
