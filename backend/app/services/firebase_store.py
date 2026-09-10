"""
Persistent case storage backed by Firebase Firestore.

Provides a drop-in persistence layer for the /api/analyze and /api/cases
endpoints.  When Firebase is reachable, every analyzed case is persisted to
a Firestore collection and loaded back on startup so data survives server
restarts.

When Firebase is unavailable (bad network, missing key, Firestore disabled)
the module degrades gracefully: all public functions become no-ops and the
in-memory _case_store stays the single source of truth for that session.
The API contract does not change — callers still read from _case_store.

Configuration
-------------
Set FIREBASE_KEY_PATH to the path of your service account JSON key, e.g.::

    FIREBASE_KEY_PATH=C:/Users/you/Downloads/traumasence-firebase-adminsdk-XXXX.json

Leave it unset and the app runs without cloud persistence (useful for local
dev without a key, or for running offline when the demo network is flaky).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from app.config import settings

# ---------------------------------------------------------------------------
# Firestore collection name for TraumaSense cases
# ---------------------------------------------------------------------------
COLLECTION = "traumasense_cases"

# ---------------------------------------------------------------------------
# Globals — initialised once on first use
# ---------------------------------------------------------------------------
_firebase_ready: bool = False
_db: Any = None  # firestore.Client when ready
_ready_error: str | None = None


def _load_key_path() -> Path | None:
    """Return the resolved Firebase key path from settings, or None."""
    raw = settings.firebase_key_path
    if not raw or not raw.strip():
        return None
    p = Path(raw.strip())
    if not p.is_absolute():
        # Resolve relative to the working directory (where the server runs).
        p = Path.cwd() / p
    return p if p.exists() else None


def _init_firebase() -> bool:
    """Try to initialise Firebase Admin + Firestore.

    Returns True if everything is reachable, False if we should fall back
    to in-memory storage for this process lifetime.
    """
    global _firebase_ready, _db, _ready_error

    if _firebase_ready:
        return _db is not None

    key_path = _load_key_path()
    if key_path is None:
        _ready_error = "FIREBASE_KEY_PATH not set or file not found"
        logger.info(_ready_error)
        return False

    try:
        import firebase_admin  # noqa: F401 — may already be initialised
        from firebase_admin import credentials
        from firebase_admin import firestore as _firestore

        if not firebase_admin._apps:
            cred = credentials.Certificate(str(key_path))
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK initialised from %s", key_path)

        _db = _firestore.client()
        # Lightweight reachability check — write then delete a probe doc.
        probe = _db.collection("_conn_probe").document("ready")
        probe.set({"ts": firebase_admin.firestore.SERVER_TIMESTAMP})
        probe.delete()
        _firebase_ready = True
        _ready_error = None
        logger.info("Firestore connection verified (collection=%s)", COLLECTION)
        return True

    except Exception as exc:  # noqa: BLE001
        _ready_error = f"Firebase init failed: {exc}"
        logger.warning(_ready_error)
        return False


def is_ready() -> bool:
    """True if Firebase persistence is active for this process."""
    if not _firebase_ready:
        _init_firebase()
    return _firebase_ready


def cloud_save(case_id: str, data: dict[str, Any]) -> bool:
    """Persist a case to Firestore.  Returns True on success, False on fallback.

    The document ID is the case_id, stored in the ``traumasense_cases``
    collection.  Each write also stamps a ``saved_at`` timestamp so we can
    order cases by recency without relying on document creation time.
    """
    if not is_ready():
        return False

    try:
        doc = _db.collection(COLLECTION).document(case_id)
        payload = dict(data)  # do not mutate the caller's dict
        payload["saved_at"] = _now_iso()
        doc.set(payload)
        logger.debug("Saved case %s to Firestore", case_id)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("cloud_save(%s) failed: %s", case_id, exc)
        return False


def cloud_load(case_id: str) -> dict[str, Any] | None:
    """Retrieve a case from Firestore by ID, or None if missing/unavailable."""
    if not is_ready():
        return None

    try:
        doc = _db.collection(COLLECTION).document(case_id).get()
        if not doc.exists:
            return None
        payload = dict(doc.to_dict())
        payload.pop("saved_at", None)  # internal field, not part of the case
        return payload
    except Exception as exc:  # noqa: BLE001
        logger.warning("cloud_load(%s) failed: %s", case_id, exc)
        return None


def cloud_load_all() -> dict[str, dict[str, Any]]:
    """Return every persisted case as {case_id: data}.

    Only cases that actually exist in Firestore are returned.  When
    Firebase is unavailable this returns an empty dict so the caller
    falls back to its in-memory store.
    """
    if not is_ready():
        return {}

    try:
        docs = _db.collection(COLLECTION).stream()
        out: dict[str, dict[str, Any]] = {}
        for doc in docs:
            payload = dict(doc.to_dict())
            payload.pop("saved_at", None)
            out[doc.id] = payload
        logger.info("Loaded %d cases from Firestore", len(out))
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("cloud_load_all() failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
