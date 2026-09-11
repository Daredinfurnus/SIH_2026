"""
Emotion analysis service for TraumaSense.

Combines text-based emotion (IndicBERT) and acoustic emotion (wav2vec2)
into a unified per-segment emotion label with confidence.

Design:
- Text path: IndicBERT embedding similarity against emotion prototypes
  (English + Hindi). Works in the original language — no translation.
- Acoustic path: wav2vec2-base latent features → stress proxy → emotion
  inference. Low-cost acoustic signal that catches tension/urgency even
  when text is sparse.
- Fusion: text emotion wins when confidence is high; acoustic contributes
  when text is short/low-confidence. Final label is the max-confidence
  signal, with acoustic as a tie-breaker / secondary signal.
"""

from __future__ import annotations

from app.config import settings

import logging
from typing import Any

from app.schemas import Emotion

logger = logging.getLogger(__name__)

try:
    from app.services.nlp_service import get_nlp_service
except ImportError:
    get_nlp_service = None  # type: ignore[assignment]

try:
    from app.services.acoustic_service import get_wav2vec2_service
except ImportError:
    get_wav2vec2_service = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Emotion label weights (used to rank text emotion scores)
# ---------------------------------------------------------------------------
_EMOTION_LABEL_WEIGHTS: dict[str, float] = {
    "fear": 1.0,
    "anxiety": 0.85,
    "sadness": 0.80,
    "anger": 0.75,
    "helplessness": 0.70,
    "hopelessness": 0.65,
    "distress": 0.60,
    "safety concern": 0.55,
    "isolation": 0.50,
    "sleep disturbance": 0.20,
}


def _text_emotion_label(nlp_emotion_scores: dict[str, float]) -> Emotion:
    """Map IndicBERT emotion prototype scores to an Emotion enum label."""
    if not nlp_emotion_scores:
        return Emotion.UNCERTAINTY

    weighted: dict[str, float] = {}
    for emotion, score in nlp_emotion_scores.items():
        weight = _EMOTION_LABEL_WEIGHTS.get(emotion, 0.5)
        weighted[emotion] = score * weight

    best_emotion = max(weighted, key=weighted.get)
    best_score = weighted[best_emotion]

    if best_score < 0.15:
        return Emotion.UNCERTAINTY

    # Map prototype emotion names to Emotion enum values
    mapping = {
        "fear": Emotion.FEAR,
        "anxiety": Emotion.ANXIETY,
        "sadness": Emotion.SADNESS,
        "anger": Emotion.ANGER,
        "helplessness": Emotion.HELPLESSNESS,
        "hopelessness": Emotion.HOPELESSNESS,
        "distress": Emotion.DISTRESS,
        "safety concern": Emotion.THREAT,
        "isolation": Emotion.SADNESS,
        "sleep disturbance": Emotion.UNCERTAINTY,
    }

    return mapping.get(best_emotion, Emotion.UNCERTAINTY)


def _acoustic_emotion_label(
    acoustic_stress: int, acoustic_confidence: float
) -> tuple[Emotion, float]:
    """
    Infer an emotion label from acoustic stress + confidence.

    Higher acoustic stress → fear/threat/helplessness range.
    Lower acoustic stress → calm/uncertainty range.
    """
    if acoustic_confidence < 0.35:
        return Emotion.UNCERTAINTY, 0.0

    if acoustic_stress >= 65:
        return Emotion.FEAR, min(0.60, acoustic_confidence)
    elif acoustic_stress >= 45:
        return Emotion.THREAT, min(0.50, acoustic_confidence)
    elif acoustic_stress >= 25:
        return Emotion.ANXIETY, min(0.40, acoustic_confidence)
    else:
        return Emotion.CALM, min(0.35, acoustic_confidence)


def _fuse_emotions(
    text_emotion: Emotion,
    text_confidence: float,
    acoustic_emotion: Emotion,
    acoustic_confidence: float,
) -> Emotion:
    """
    Fuse text and acoustic emotion signals into a single label.

    Rules:
    - If text confidence is high (>= 0.60), prefer text.
    - If acoustic confidence is high (>= 0.50) and text is low,
      prefer acoustic.
    - Otherwise, pick the higher-confidence signal.
    - If both are weak, return Uncertainty.
    """
    if text_confidence >= 0.60:
        return text_emotion

    if acoustic_confidence >= 0.50 and text_confidence < 0.40:
        return acoustic_emotion

    # Pick the higher-confidence signal
    if text_confidence >= acoustic_confidence:
        return text_emotion
    else:
        return acoustic_emotion


class EmotionService:
    """Unified emotion analysis: text (IndicBERT) + acoustic (wav2vec2)."""

    def analyze_segment(
        self,
        text: str,
        audio_path: str | None = None,
        start_sec: float = 0.0,
        end_sec: float = 0.0,
    ) -> dict[str, Any]:
        """
        Analyze a single transcript segment for emotion.

        Returns dict with:
            emotion             (Emotion label string)
            emotion_source      ("text" | "acoustic" | "fused" | "none")
            text_emotion_score  (0.0-1.0, from IndicBERT prototypes)
            text_confidence     (0.0-1.0, derived from score margin)
            acoustic_emotion    (str, inferred from acoustic)
            acoustic_confidence (0.0-1.0)
            acoustic_stress     (int 0-100)
            accent_signals      (dict or None)
            emotion_explanation (list[str], one-line rationale)
        """
        result: dict[str, Any] = {
            "emotion": Emotion.UNCERTAINTY.value,
            "emotion_source": "none",
            "text_emotion_score": 0.0,
            "text_confidence": 0.0,
            "acoustic_emotion": Emotion.UNCERTAINTY.value,
            "acoustic_confidence": 0.0,
            "acoustic_stress": 0,
            "accent_signals": None,
            "emotion_explanation": [],
        }

        if not text or not text.strip():
            result["emotion_explanation"].append("No transcript text to analyze.")
            return result

        # ---- Text-based emotion (IndicBERT) ----
        text_emotion_score = 0.0
        text_confidence = 0.0
        nlp_emotion_scores: dict[str, float] = {}
        nlp_svc = get_nlp_service() if get_nlp_service else None

        if nlp_svc is not None:
            nlp_emotion_scores = nlp_svc.score_emotions(text)
            if nlp_emotion_scores:
                # Top emotion score
                text_emotion_score = max(nlp_emotion_scores.values())
                # Confidence from score distribution
                values = sorted(nlp_emotion_scores.values(), reverse=True)
                top = values[0] if values else 0.0
                second = values[1] if len(values) > 1 else 0.0
                margin = top - second
                text_confidence = max(0.20, min(0.85, 0.30 + margin * 0.5))

                text_emotion = _text_emotion_label(nlp_emotion_scores)
            else:
                text_emotion = Emotion.UNCERTAINTY
        else:
            text_emotion = Emotion.UNCERTAINTY

        result["text_emotion_score"] = round(text_emotion_score, 3)
        result["text_confidence"] = round(text_confidence, 3)

        # ---- Acoustic-based emotion (wav2vec2) ----
        acoustic_stress = 0
        acoustic_confidence = 0.0
        acoustic_emotion = Emotion.UNCERTAINTY.value

        if settings.enable_acoustic and audio_path and start_sec >= 0 and end_sec > start_sec:
            wav_svc = get_wav2vec2_service() if get_wav2vec2_service else None
            if wav_svc is not None:
                try:
                    acoustic = wav_svc.analyze_segment(audio_path, start_sec, end_sec)
                    acoustic_stress = acoustic.get("acoustic_stress_score", 0)
                    acoustic_confidence = acoustic.get("acoustic_confidence", 0.0)
                    acoustic_emotion, _ = _acoustic_emotion_label(
                        acoustic_stress, acoustic_confidence
                    )
                except Exception:
                    logger.warning("Wav2Vec2 acoustic analysis failed for segment", exc_info=True)
            else:
                logger.debug("Wav2Vec2 service unavailable; skipping acoustic emotion")
        else:
            logger.debug("No audio segment provided; skipping acoustic emotion")

        result["acoustic_stress"] = acoustic_stress
        result["acoustic_confidence"] = round(acoustic_confidence, 3)
        result["acoustic_emotion"] = acoustic_emotion.value if isinstance(acoustic_emotion, Emotion) else acoustic_emotion

        # ---- Accent / dialect signals ----
        accent_signals = None
        if nlp_svc is not None:
            try:
                accent_signals = nlp_svc.detect_accent_dialect(text)
            except Exception:
                pass
        result["accent_signals"] = accent_signals

        # ---- Fusion ----
        if text_emotion_score > 0 or acoustic_stress > 0:
            fused_emotion = _fuse_emotions(
                text_emotion,
                text_confidence,
                Emotion(acoustic_emotion) if acoustic_emotion in [e.value for e in Emotion] else Emotion.UNCERTAINTY,
                acoustic_confidence,
            )
            result["emotion"] = fused_emotion.value
            result["emotion_source"] = "fused"
        elif text_emotion_score > 0:
            result["emotion"] = text_emotion.value
            result["emotion_source"] = "text"
        elif acoustic_stress > 0:
            result["emotion"] = acoustic_emotion
            result["emotion_source"] = "acoustic"
        else:
            result["emotion"] = Emotion.UNCERTAINTY.value
            result["emotion_source"] = "none"

        # ---- Explanation ----
        explanation: list[str] = []
        if result["emotion_source"] == "text":
            explanation.append(
                f"Text-based emotion: {result['emotion']} "
                f"(IndicBERT similarity {text_emotion_score:.2f}, confidence {text_confidence:.2f})"
            )
        elif result["emotion_source"] == "acoustic":
            explanation.append(
                f"Acoustic emotion: {result['emotion']} "
                f"(stress proxy {acoustic_stress}/100, confidence {acoustic_confidence:.2f})"
            )
        elif result["emotion_source"] == "fused":
            explanation.append(
                f"Fused emotion: {result['emotion']} "
                f"(text {text_emotion_score:.2f}/{text_confidence:.2f}, "
                f"acoustic {acoustic_stress}/100/{acoustic_confidence:.2f})"
            )
        else:
            explanation.append("Insufficient signal for emotion classification — defaulting to Uncertainty.")

        if accent_signals and accent_signals.get("confidence", 0) > 0.3:
            parts = []
            if accent_signals.get("hindi_dravidian_influence", 0) > 0.2:
                parts.append("possible Dravidian-influenced Hindi")
            if accent_signals.get("hindi_urdū_overlap", 0) > 0.2:
                parts.append("possible Hindustani/Urdū overlap")
            if parts:
                explanation.append(f"Accent signals: {'; '.join(parts)} (confidence {accent_signals['confidence']:.2f})")

        result["emotion_explanation"] = explanation

        return result


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_emotion_svc: EmotionService | None = None


def get_emotion_service() -> EmotionService:
    """Get or create the singleton EmotionService instance."""
    global _emotion_svc
    if _emotion_svc is None:
        _emotion_svc = EmotionService()
    return _emotion_svc
