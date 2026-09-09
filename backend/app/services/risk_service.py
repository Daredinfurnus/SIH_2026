"""
Risk engine — maps the composite assessment to an assistive risk level.

IMPORTANT: Stress is NOT the same as risk.

Risk is computed from:
  - SVI (composite stress + distress + context + safety)
  - immediate safety flags
  - overall confidence
  - contributing indicators

Prototype thresholds (NOT clinical thresholds):

    0-24   LOW
    25-49  MODERATE
    50-74  HIGH
    75-100 CRITICAL

The engine returns:
    risk_level
    risk_score
    confidence
    explanation           (list of human-readable reasons)
    contributing_indicators
"""
from __future__ import annotations

from typing import Any

from app.schemas import RiskLevel


# ---------------------------------------------------------------------------
# Prototype risk thresholds
# ---------------------------------------------------------------------------

_THRESHOLD_LOW_MAX = 24
_THRESHOLD_MOD_MAX = 49
_THRESHOLD_HIGH_MAX = 74
# 75-100 = CRITICAL


class RiskService:
    """Prototype risk engine for assistive case prioritisation."""

    def assess(
        self,
        svi_score: int,
        overall_stress: int,
        overall_distress: int,
        indicators: list[str],
        confidence: float,
        immediate_safety: bool,
        segment_risks: list[str] | None = None,
        svi_breakdown: dict | None = None,
    ) -> dict[str, Any]:
        """Compute an assistive risk level with explanation.

        Returns:
            risk_level, risk_score, confidence, explanation,
            contributing_indicators, immediate_safety_indicators
        """
        # ---------------------------------------------------------------
        # 1. Base risk score from SVI
        # ---------------------------------------------------------------
        base_risk = int(max(0, min(100, svi_score)))

        # ---------------------------------------------------------------
        # 2. Immediate safety escalation
        # ---------------------------------------------------------------
        if immediate_safety:
            # Conservative escalation: never downgrade below HIGH when an
            # immediate-safety signal is present.
            base_risk = max(base_risk, 60)

        # ---------------------------------------------------------------
        # 3. Confidence adjustment
        # ---------------------------------------------------------------
        # When confidence is low, we do NOT inflate risk — we keep it
        # conservative and let the explanation carry the uncertainty.
        effective_risk = base_risk
        if confidence < 0.5:
            effective_risk = max(0, base_risk - 8)

        # ---------------------------------------------------------------
        # 4. Derive level from thresholds
        # ---------------------------------------------------------------
        if effective_risk <= _THRESHOLD_LOW_MAX:
            level = RiskLevel.LOW
        elif effective_risk <= _THRESHOLD_MOD_MAX:
            level = RiskLevel.MODERATE
        elif effective_risk <= _THRESHOLD_HIGH_MAX:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL

        # ---------------------------------------------------------------
        # 5. Explanation + contributing indicators
        # ---------------------------------------------------------------
        explanation = self._explain(
            level, base_risk, effective_risk, indicators, immediate_safety, confidence,
            svi_breakdown=svi_breakdown,
        )
        contributing = self._contributing_indicators(indicators, level)

        return {
            "risk_level": level,
            "risk_score": effective_risk,
            "confidence": round(confidence, 2),
            "explanation": explanation,
            "contributing_indicators": contributing,
            "immediate_safety_indicators": immediate_safety,
        }

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _explain(
        self,
        level: RiskLevel,
        base_risk: int,
        effective_risk: int,
        indicators: list[str],
        immediate_safety: bool,
        confidence: float,
        svi_breakdown: dict | None = None,
    ) -> list[str]:
        parts: list[str] = []

        bd = svi_breakdown or {}
        stress_c = bd.get("stress_component", 0)
        distress_c = bd.get("distress_component", 0)
        safety_c = bd.get("safety_component", 0)
        context_c = bd.get("context_component", 0)

        if immediate_safety:
            parts.append("Immediate safety indicators detected — trained human review required.")

        if level == RiskLevel.LOW:
            parts.append(f"SVI score {effective_risk}/100 — below the 25-point Moderate threshold.")
            parts.append(f"Stress component {stress_c}, distress component {distress_c} — both in the low range.")
            if context_c > 0:
                parts.append(f"Context indicators contributed {context_c} points.")
            parts.append("No immediate safety concerns detected.")
        elif level == RiskLevel.MODERATE:
            parts.append(f"SVI score {effective_risk}/100 — in the 25-49 Moderate band.")
            parts.append(f"Stress component {stress_c}, distress component {distress_c} — elevated but below the 50-point High threshold.")
            if context_c > 0:
                parts.append(f"Context indicators contributed {context_c} points.")
            parts.append("Conversation shows elevated concern but no immediate safety signals.")
        elif level == RiskLevel.HIGH:
            parts.append(f"SVI score {effective_risk}/100 — at or above the 50-point High threshold.")
            parts.append(f"Distress component {distress_c} is the primary driver, with stress component {stress_c}.")
            if context_c > 0:
                parts.append(f"Context indicators contributed {context_c} points.")
            parts.append("Elevated distress indicators across the conversation.")
            parts.append("Fear-related and threat-related context detected.")
            if any("helplessness" in i.lower() for i in indicators):
                parts.append("Helplessness indicators present.")
            if any("isolation" in i.lower() for i in indicators):
                parts.append("Isolation indicators present.")
        else:  # CRITICAL
            parts.append(f"SVI score {effective_risk}/100 — at or above the 75-point Critical threshold.")
            parts.append(f"Distress component {distress_c} and stress component {stress_c} both elevated.")
            if context_c > 0:
                parts.append(f"Context indicators contributed {context_c} points.")
            parts.append("High distress indicators with multiple stress signals.")
            parts.append("Fear, helplessness, and threat-related context detected.")
            if any("hopelessness" in i.lower() for i in indicators):
                parts.append("Hopelessness indicators present.")
            parts.append("Trained human review required immediately.")

        if confidence < 0.6:
            parts.append(f"Assessment confidence is {int(confidence * 100)}% — interpret with caution and seek additional context.")

        return parts

    def _contributing_indicators(
        self, indicators: list[str], level: RiskLevel
    ) -> list[str]:
        if not indicators:
            return ["No specific indicators identified."]
        # Show the most salient indicators for the level.
        weights = {
            "fear": 5,
            "helplessness": 5,
            "hopelessness": 5,
            "threat-related context": 5,
            "safety concern": 4,
            "immediate safety indicators": 5,
            "distress": 4,
            "isolation": 4,
            "anxiety": 3,
            "sleep disturbance": 2,
            "uncertainty": 2,
            "relief": -1,
            "calm": -1,
        }
        ranked = sorted(indicators, key=lambda i: weights.get(i, 0), reverse=True)
        return ranked[:6]


def compute_risk(
    svi_score: int,
    overall_stress: int,
    overall_distress: int,
    indicators: list[str],
    confidence: float,
    immediate_safety: bool,
    segment_risks: list[str] | None = None,
    svi_breakdown: dict | None = None,
) -> dict[str, Any]:
    """Convenience wrapper preserving legacy call shape."""
    svc = RiskService()
    return svc.assess(
        svi_score, overall_stress, overall_distress, indicators,
        confidence, immediate_safety, segment_risks,
        svi_breakdown=svi_breakdown,
    )
