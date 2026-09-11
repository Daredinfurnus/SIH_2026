"""
Unified scoring engine — the single authoritative source of all scores.

Wires the existing per-segment analysis outputs (stress, distress, emotion,
indicators, safety flags, acoustic features) through the prototype SVI
composite and risk engine, and produces a timeline + overall result that
routes.py maps onto the legacy AnalysisResponse for the frontend.

This is a PROTOTYPE scoring engine.  It is NOT a clinically validated model.
It is designed so a validated model can replace the prototype calculation
later without changing the rest of the system.
"""
from __future__ import annotations

import logging
from typing import Any

from app.schemas import RiskLevel
from app.services.svi_service import SVIService
from app.services.risk_service import RiskService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature bridge — analyzed segment dict → flat feature vector
# ---------------------------------------------------------------------------

def build_features_from_analysis_segment(
    segment: dict[str, Any],
) -> dict[str, Any]:
    """Extract a flat, validated feature vector from an analyzed segment dict.

    This is the bridge between the route-level segment assembly (in routes.py)
    and the scoring pipeline.  The returned features are consumed by the
    pipeline for per-segment and overall scoring.

    Expected input shape (produced by routes.py `_build_analyzed_segment`):
        start_time, duration, text, speaker,
        stress_indicator, distress_indicator, confidence,
        emotion {primary, confidence, arousal},
        indicators[], safety_flag, immediate_safety_flag,
        context_score, acoustic{...}, emotion_explanation[], accent_signals...

    Returns a validated flat dict suitable for scoring.
    """
    emotion = segment.get("emotion", {})
    acoustic = segment.get("acoustic", {})

    return {
        "start_time": float(segment.get("start_time", 0.0)),
        "duration": float(segment.get("duration", 0.0)),
        "text": str(segment.get("text", "")),
        "speaker": str(segment.get("speaker", "caller")),
        # Core scores
        "stress": float(segment.get("stress_indicator", segment.get("stress_score", 0))),
        "distress": float(segment.get("distress_indicator", segment.get("distress_score", 0))),
        "confidence": float(segment.get("confidence", 0.5)),
        # Emotion
        "emotion_primary": (
            emotion.get("primary", "Uncertainty")
            if isinstance(emotion, dict)
            else str(emotion)
        ),
        "emotion_confidence": (
            float(emotion.get("confidence", 0.0))
            if isinstance(emotion, dict)
            else 0.0
        ),
        "arousal": (
            float(emotion.get("arousal", 0.5))
            if isinstance(emotion, dict)
            else 0.5
        ),
        # Indicators / safety
        "indicators": list(segment.get("indicators", [])),
        "safety_flag": bool(segment.get("safety_flag", False)),
        "immediate_safety_flag": bool(segment.get("immediate_safety_flag", False)),
        "context_score": float(segment.get("context_score", 0.0)),
        # Acoustic
        "acoustic_stress": (
            int(acoustic.get("acoustic_stress_score", 0))
            if isinstance(acoustic, dict)
            else 0
        ),
        "acoustic_confidence": (
            float(acoustic.get("acoustic_confidence", 0.0))
            if isinstance(acoustic, dict)
            else 0.0
        ),
        # Extra
        "emotion_explanation": list(segment.get("emotion_explanation", [])),
        "accent_signals": segment.get("accent_signals"),
    }


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------

class ScoringPipeline:
    """Orchestrates per-segment SVI + risk computation into a unified result.

    Flow:
        1. For each segment, compute per-segment SVI via SVIService.
        2. For each segment, derive per-segment risk via RiskService.
        3. Build a timeline record per segment (stress, distress, svi,
           confidence, risk_level).
        4. Compute overall SVI from all segments (confidence-weighted).
        5. Compute overall risk from overall SVI.
        6. Package timeline + overall + explanations.

    The SVI and risk services are the prototype implementations.  They can
    be replaced by validated models later without changing this orchestrator.
    """

    def __init__(self) -> None:
        self._svi = SVIService()
        self._risk = RiskService()

    def analyze_full_conversation(
        self,
        segments: list[dict[str, Any]],
        features_by_segment: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run the full scoring pipeline on all conversation segments.

        Parameters:
            segments: analyzed segment dicts as assembled by routes.py.
            features_by_segment: flat feature vectors from
                build_features_from_analysis_segment(), one per segment.

        Returns:
            dict with timeline[], overall{...}, explanations{...}, indicators[].
        """
        if not segments:
            return {
                "timeline": [],
                "overall": {
                    "stress_score": 0,
                    "distress_score": 0,
                    "svi_score": 0,
                    "risk_level": RiskLevel.LOW,
                    "confidence": 0.0,
                    "safety_score": 0,
                    "safety_severity": "NONE",
                    "safety_immediate_flag": False,
                },
                "explanations": {},
                "indicators": [],
            }

        # ---- Phase 1: per-segment SVI + risk + timeline ----
        timeline: list[dict[str, Any]] = []
        all_indicators: set[str] = set()
        immediate_safety = False

        total_weight = 0.0
        weighted_stress = 0.0
        weighted_distress = 0.0
        weighted_svi = 0.0

        for i, seg in enumerate(segments):
            features = (
                features_by_segment[i]
                if i < len(features_by_segment)
                else build_features_from_analysis_segment(seg)
            )

            stress = int(features.get("stress", 0))
            distress = int(features.get("distress", 0))
            seg_confidence = float(features.get("confidence", 0.5))
            seg_indicators = list(features.get("indicators", []))
            seg_immediate = bool(features.get("immediate_safety_flag", False))

            all_indicators.update(seg_indicators)
            if seg_immediate:
                immediate_safety = True

            # Per-segment SVI
            seg_svi_result = self._svi.compute(
                segments=[seg],
                overall_indicators=seg_indicators,
                immediate_safety=seg_immediate,
            )
            seg_svi = seg_svi_result.get("svi_score", 0)

            # Per-segment risk
            seg_risk = self._risk.assess(
                svi_score=seg_svi,
                overall_stress=stress,
                overall_distress=distress,
                indicators=seg_indicators,
                confidence=seg_confidence,
                immediate_safety=seg_immediate,
                svi_breakdown=seg_svi_result.get("svi_breakdown", {}),
            )

            w = seg_confidence
            total_weight += w
            weighted_stress += stress * w
            weighted_distress += distress * w
            weighted_svi += seg_svi * w

            timeline.append(
                {
                    "timestamp": features.get("start_time", 0.0),
                    "duration": features.get("duration", 0.0),
                    "stress": stress,
                    "distress": distress,
                    "svi": seg_svi,
                    "confidence": min(seg_confidence * 100, 100),
                    "risk_level": seg_risk["risk_level"],
                    "contributors": [],
                    "events": [],
                }
            )

        # ---- Phase 2: overall SVI from all segments ----
        overall_svi_result = self._svi.compute(
            segments=segments,
            overall_indicators=list(all_indicators),
            immediate_safety=immediate_safety,
        )
        overall_svi = overall_svi_result.get("svi_score", 0)
        svi_breakdown = overall_svi_result.get("svi_breakdown", {})

        # ---- Phase 3: overall risk ----
        avg_stress = (
            int(round(weighted_stress / total_weight)) if total_weight > 0 else 0
        )
        avg_distress = (
            int(round(weighted_distress / total_weight)) if total_weight > 0 else 0
        )
        avg_confidence = (
            total_weight / len(segments) if segments else 0.0
        )

        overall_risk = self._risk.assess(
            svi_score=int(round(overall_svi)),
            overall_stress=avg_stress,
            overall_distress=avg_distress,
            indicators=list(all_indicators),
            confidence=avg_confidence,
            immediate_safety=immediate_safety,
            svi_breakdown=svi_breakdown,
        )

        # ---- Phase 4: safety severity ----
        safety_component = svi_breakdown.get("safety_component", 0)
        if immediate_safety:
            safety_severity = "IMMEDIATE_CONCERN"
        elif safety_component >= 65:
            safety_severity = "ELEVATED_CONCERN"
        elif safety_component >= 40:
            safety_severity = "GENERAL_CONCERN"
        else:
            safety_severity = "NONE"

        # ---- Phase 5: explanations ----
        explanations = {
            "stress": {
                "score": avg_stress,
                "confidence": avg_confidence * 100,
            },
            "distress": {
                "score": avg_distress,
                "confidence": avg_confidence * 100,
            },
            "svi": {
                "svi_final": int(round(overall_svi)),
                "svi_breakdown": svi_breakdown,
            },
            "safety": {
                "score": safety_component,
                "severity": safety_severity,
                "immediate_safety_flag": immediate_safety,
                "confidence": avg_confidence * 100,
            },
        }

        return {
            "timeline": timeline,
            "overall": {
                "stress_score": avg_stress,
                "distress_score": avg_distress,
                "svi_score": int(round(overall_svi)),
                "risk_level": overall_risk["risk_level"],
                "confidence": min(avg_confidence, 1.0),
                "safety_score": safety_component,
                "safety_severity": safety_severity,
                "safety_immediate_flag": immediate_safety,
            },
            "explanations": explanations,
            "indicators": sorted(all_indicators),
        }
