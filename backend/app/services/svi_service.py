"""
Stress Vulnerability Index (SVI) — prototype composite assessment service.

The SVI is a single, explainable composite score on a 0-100 scale that fuses:
  - stress signals
  - distress signals
  - contextual indicators
  - safety indicators
  - per-segment confidence

It is a PROTOTYPE composite indicator.  It is NOT a clinically validated index.
It is designed so a validated SVI model can replace the prototype calculation
later without changing the rest of the system.

Formula (prototype):
    SVI = weighted combination of per-segment stress, distress, safety flags,
          and indicators, aggregated across the conversation.
"""
from __future__ import annotations

from typing import Any

from app.schemas import RiskLevel


class SVIService:
    """Prototype composite SVI computation."""

    # Weights are deliberately transparent so the calculation can be explained.
    _W_STRESS = 0.30
    _W_DISTRESS = 0.35
    _W_SAFETY = 0.20
    _W_CONTEXT = 0.15

    def compute(
        self,
        segments: list[dict[str, Any]],
        overall_indicators: list[str],
        immediate_safety: bool,
    ) -> dict[str, Any]:
        """Return SVI-derived composite score and breakdown."""
        if not segments:
            return {
                "svi_score": 0,
                "svi_breakdown": {},
                "note": "No segments available for composite assessment.",
            }

        weighted_sum = 0.0
        total_weight = 0.0
        breakdown: dict[str, Any] = {
            "stress_component": 0.0,
            "distress_component": 0.0,
            "safety_component": 0.0,
            "context_component": 0.0,
            "segment_count": len(segments),
            "immediate_safety": immediate_safety,
        }

        stress_vals: list[float] = []
        distress_vals: list[float] = []

        for seg in segments:
            stress = float(seg.get("stress_score", 0))
            distress = float(seg.get("distress_score", 0))
            safety_flag = bool(seg.get("safety_flag", False))
            immediate_flag = bool(seg.get("immediate_safety_flag", False))
            seg_confidence = float(seg.get("confidence", 0.5))

            stress_vals.append(stress)
            distress_vals.append(distress)

            # Per-segment safety component: 0-100 scale, boosted by immediate flag.
            safety_score = 50.0
            if safety_flag:
                safety_score = 65.0
            if immediate_flag:
                safety_score = 90.0
            # Weight by confidence so uncertain segments pull less.
            safety_component = safety_score * seg_confidence

            # Context component: derived from indicator richness.
            indicators = seg.get("indicators", [])
            context_score = min(100.0, len(indicators) * 18.0)

            seg_weight = seg_confidence
            weighted_sum += (
                stress * self._W_STRESS
                + distress * self._W_DISTRESS
                + safety_component * self._W_SAFETY
                + context_score * self._W_CONTEXT
            ) * seg_weight
            total_weight += self._W_STRESS + self._W_DISTRESS + self._W_SAFETY + self._W_CONTEXT

            breakdown["stress_component"] += stress * self._W_STRESS * seg_weight
            breakdown["distress_component"] += distress * self._W_DISTRESS * seg_weight
            breakdown["safety_component"] += safety_component * self._W_SAFETY * seg_weight
            breakdown["context_component"] += context_score * self._W_CONTEXT * seg_weight

        if total_weight <= 0:
            return {
                "svi_score": 0,
                "svi_breakdown": breakdown,
                "note": "Unable to compute SVI from empty analysis.",
            }

        raw_svi = weighted_sum / (total_weight / (self._W_STRESS + self._W_DISTRESS + self._W_SAFETY + self._W_CONTEXT))
        # Clamp to 0-100
        svi_score = max(0, min(100, round(raw_svi)))

        # Immediate safety escalation: if any segment raised an immediate-safety
        # flag, the composite is pulled upward toward the high end to make sure
        # the risk engine sees the signal.
        if immediate_safety:
            svi_score = max(svi_score, 65)

        # Round components to whole numbers for display
        breakdown = {
            **breakdown,
            "stress_component": round(breakdown["stress_component"]),
            "distress_component": round(breakdown["distress_component"]),
            "safety_component": round(breakdown["safety_component"]),
            "context_component": round(breakdown["context_component"]),
        }

        return {
            "svi_score": svi_score,
            "svi_breakdown": breakdown,
        }


def compute_overall_svi(
    segments: list[dict[str, Any]],
    overall_indicators: list[str],
    immediate_safety: bool,
) -> dict[str, Any]:
    """Convenience wrapper matching the legacy function shape."""
    svc = SVIService()
    return svc.compute(segments, overall_indicators, immediate_safety)
