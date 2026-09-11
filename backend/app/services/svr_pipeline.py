"""
SVR feature pipeline for TraumaSense.

Builds a deterministic feature vector from analysis segments, compatible
with a trained SVR model.  When no trained model is loaded, provides a
deterministic fallback risk score.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.services.scoring_engine import (
    Feature,
    build_features_from_analysis_segment,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical feature schema — MUST match any trained SVR exactly
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "linguistic_arousal",        # stress indicator from NLP (0-1)
    "emotional_arousal",         # emotion classifier arousal (0-1)
    "emotional_suffering",       # negative emotion flag (0-1)
    "acoustic_energy",           # acoustic energy proxy (0-1)
    "speech_disruption",         # acoustic disruption proxy (0-1)
    "physiological_arousal",     # NOT available — always 0
    "context_vulnerability",     # context score (0-1)
    "safety_threat_keyword",     # safety feature (0-1)
    "safety_self_harm_keyword",  # safety feature (0-1)
    "safety_violation_mentioned",# safety feature (0-1)
    "safety_urgent_language",    # safety feature (0-1)
    "safety_protector_present",  # safety feature (0-1)
    "hopelessness",              # distress indicator (0-1)
    "helplessness",              # distress indicator (0-1)
    "loss_of_control",           # distress indicator (0-1)
    "isolation",                 # distress indicator (0-1)
    "coping_difficulty",         # distress indicator (0-1)
    "threat_distress",           # distress indicator (0-1)
    "severity",                  # distress indicator (0-1)
]

_FEATURE_COUNT = len(FEATURE_NAMES)

# ---------------------------------------------------------------------------
# Trained SVR model placeholder
# ---------------------------------------------------------------------------

_trained_svr = None       # sklearn.svm.SVR | None
_svr_scaler = None        # sklearn.preprocessing.StandardScaler | None
_svr_load_error: Optional[str] = None

DEFAULT_SVR_MODEL_PATH = "models/svr_trained.joblib"


def load_svr_model(model_path: Optional[str] = None) -> bool:
    """Load a trained SVR model and its scaler from disk.

    The file may contain either:
      - a dict with keys 'svr' and optionally 'scaler'
      - just the SVR instance

    Returns True if loading succeeded.
    """
    global _trained_svr, _svr_scaler, _svr_load_error
    if model_path is None:
        model_path = DEFAULT_SVR_MODEL_PATH

    try:
        import joblib
        data = joblib.load(model_path)
        if isinstance(data, dict):
            _trained_svr = data.get("svr")
            _svr_scaler = data.get("scaler")
        else:
            _trained_svr = data
            _svr_scaler = None
        _svr_load_error = None
        logger.info("SVR model loaded from %s", model_path)
        return True
    except FileNotFoundError:
        _svr_load_error = f"SVR model file not found: {model_path}"
        logger.warning(_svr_load_error)
        return False
    except Exception as exc:
        _svr_load_error = str(exc)
        logger.warning("SVR model load failed: %s", exc)
        return False


def is_svr_available() -> bool:
    """Check whether a trained SVR model is loaded and ready."""
    return _trained_svr is not None


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features_from_segment(
    seg: Dict[str, Any],
    config: Optional[Any] = None,
) -> List[float]:
    """Extract a feature vector from one analyzed segment.

    Bridges from the existing scoring engine's build_features_from_analysis_segment()
    to the flat FEATURE_NAMES-ordered vector.
    """
    engine_features = build_features_from_analysis_segment(seg, config)

    vector: List[float] = []
    for fname in FEATURE_NAMES:
        f = engine_features.get(fname)
        if f is not None and isinstance(f, Feature) and f.available:
            vector.append(float(f.normalized_value))
        else:
            vector.append(0.0)

    return vector


def extract_features_from_segments(
    segments: List[Dict[str, Any]],
    config: Optional[Any] = None,
) -> List[List[float]]:
    """Extract feature vectors for all segments."""
    return [extract_features_from_segment(seg, config) for seg in segments]


def aggregate_features(
    feature_vectors: List[List[float]],
    method: str = "mean",
) -> List[float]:
    """Aggregate per-segment feature vectors into a single conversation-level vector."""
    if not feature_vectors:
        return [0.0] * _FEATURE_COUNT

    if method == "mean":
        n = len(feature_vectors)
        agg: List[float] = [0.0] * _FEATURE_COUNT
        for vec in feature_vectors:
            for i in range(min(len(vec), _FEATURE_COUNT)):
                agg[i] += vec[i]
        return [v / n for v in agg]

    if method == "max":
        agg = feature_vectors[0][:]
        for vec in feature_vectors[1:]:
            for i in range(min(len(vec), _FEATURE_COUNT)):
                if vec[i] > agg[i]:
                    agg[i] = vec[i]
        return agg

    raise ValueError(f"Unknown aggregation method: {method}")


# ---------------------------------------------------------------------------
# SVR prediction
# ---------------------------------------------------------------------------

def predict_risk_with_svr(
    feature_vector: List[float],
) -> Dict[str, Any]:
    """Run a feature vector through the trained SVR (if available).

    Returns:
        prediction: float SVR output (or None if unavailable)
        available:  bool
        error:      str error description if unavailable
    """
    if not is_svr_available():
        return {
            "prediction": None,
            "available": False,
            "error": _svr_load_error or "No trained SVR model loaded",
        }

    try:
        import numpy as np
        X = np.array([feature_vector], dtype=np.float32)
        if _svr_scaler is not None:
            X = _svr_scaler.transform(X)
        pred = float(_trained_svr.predict(X)[0])
        return {
            "prediction": pred,
            "available": True,
            "error": None,
        }
    except Exception as exc:
        logger.warning("SVR prediction failed: %s", exc)
        return {
            "prediction": None,
            "available": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# End-to-end conversation risk prediction
# ---------------------------------------------------------------------------

def predict_conversation_risk(
    segments: List[Dict[str, Any]],
    config: Optional[Any] = None,
    aggregation: str = "mean",
) -> Dict[str, Any]:
    """End-to-end risk prediction for a conversation.

    Returns:
        svr_prediction: float or None
        svr_available: bool
        feature_vector: List[float] — aggregated feature vector
        fallback_risk: float — deterministic fallback (0-100)
        error: str or None
    """
    feature_vectors = extract_features_from_segments(segments, config)
    aggregated = aggregate_features(feature_vectors, aggregation)

    svr_result = predict_risk_with_svr(aggregated)
    fallback_risk = _deterministic_risk_fallback(aggregated)

    return {
        "svr_prediction": svr_result["prediction"],
        "svr_available": svr_result["available"],
        "feature_vector": aggregated,
        "fallback_risk": fallback_risk,
        "error": svr_result["error"],
    }


# ---------------------------------------------------------------------------
# Deterministic fallback (when no trained SVR is available)
# ---------------------------------------------------------------------------

def _deterministic_risk_fallback(feature_vector: List[float]) -> float:
    """Deterministic risk score from feature vector when SVR is unavailable.

    Weighted combination of stress, distress, safety, and context components.
    Returns a float in 0-100.
    """
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    def g(name: str, default: float = 0.0) -> float:
        if name not in idx:
            return default
        v = feature_vector[idx[name]]
        return float(v) if isinstance(v, (int, float)) else default

    # Stress subscore (0-100)
    stress = (
        g("linguistic_arousal") * 0.40
        + g("emotional_arousal") * 0.30
        + g("acoustic_energy") * 0.20
        + g("speech_disruption") * 0.10
    ) * 100.0

    # Distress subscore (0-100)
    distress = (
        g("emotional_suffering") * 0.30
        + g("hopelessness") * 0.20
        + g("helplessness") * 0.15
        + g("loss_of_control") * 0.10
        + g("isolation") * 0.10
        + g("coping_difficulty") * 0.10
        + g("threat_distress") * 0.05
    ) * 100.0

    # Safety subscore (0-100)
    safety = (
        g("safety_threat_keyword") * 0.30
        + g("safety_self_harm_keyword") * 0.25
        + g("safety_violation_mentioned") * 0.20
        + g("safety_urgent_language") * 0.15
        + g("safety_protector_present") * 0.10
    ) * 100.0

    # Context component
    context = g("context_vulnerability", 0.0) * 100.0

    # SVI-like composite
    svi = (
        stress * 0.25
        + distress * 0.30
        + safety * 0.25
        + context * 0.20
    )

    return float(max(0.0, min(100.0, svi)))
