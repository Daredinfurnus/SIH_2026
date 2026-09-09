"""
Recommendation engine — maps assistive risk level to a prototype action suggestion.

These are ASSISTIVE RECOMMENDATIONS ONLY.  The system does not independently
execute them.  Trained human operators retain decision authority.

Prototype mapping:

    LOW       → information / reassurance
    MODERATE  → counselling and relevant support resources
    HIGH      → professional routing and priority human review
    CRITICAL  → immediate human handoff and emergency-support escalation
"""
from __future__ import annotations

from app.schemas import RiskLevel


class RecommendationService:
    """Prototype assistive recommendation mapping."""

    _RECOMMENDATIONS: dict[RiskLevel, str] = {
        RiskLevel.LOW: (
            "Low conversational indicators of stress and distress. "
            "Provide information, reassurance, and routine helpline support. "
            "No immediate escalation indicated by this assessment."
        ),
        RiskLevel.MODERATE: (
            "Moderate stress and distress indicators detected. "
            "Offer counselling support and relevant resources. "
            "Schedule follow-up and continue to monitor the case."
        ),
        RiskLevel.HIGH: (
            "High stress and distress indicators detected. "
            "Professional routing and priority human review recommended. "
            "Consider immediate counsellor handoff and safety planning."
        ),
        RiskLevel.CRITICAL: (
            "Critical distress indicators detected. "
            "Immediate human handoff and emergency-support escalation recommended. "
            "Trained human intervention required without delay."
        ),
    }

    def recommend(
        self,
        risk_level: RiskLevel,
        immediate_safety: bool = False,
        extra_context: str | None = None,
    ) -> str:
        base = self._RECOMMENDATIONS.get(risk_level, self._RECOMMENDATIONS[RiskLevel.MODERATE])

        if immediate_safety:
            base = (
                "Immediate safety indicators detected. "
                + base
                + " Prioritise immediate human review."
            )

        if extra_context:
            base = base + " Context: " + extra_context

        return base


def get_recommendation(
    risk_level: RiskLevel,
    immediate_safety: bool = False,
    extra_context: str | None = None,
) -> str:
    """Convenience wrapper preserving legacy call shape."""
    svc = RecommendationService()
    return svc.recommend(risk_level, immediate_safety, extra_context)
