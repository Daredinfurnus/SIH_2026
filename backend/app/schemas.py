"""
Pydantic schemas for the TraumaSense API contract.

These are the source of truth for request/response shapes.
The frontend API client and the backend both honour these.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ===========================================================================
# Enums
# ===========================================================================

class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Speaker(str, Enum):
    CALLER = "caller"
    OPERATOR = "operator"


class Emotion(str, Enum):
    """Prototype conversational emotion/context labels."""

    FEAR = "Fear"
    ANXIETY = "Anxiety"
    SADNESS = "Sadness"
    ANGER = "Anger"
    UNCERTAINTY = "Uncertainty"
    CALM = "Calm"
    DISTRESS = "Distress"
    HOPELESSNESS = "Hopelessness"
    HELPLESSNESS = "Helplessness"
    THREAT = "Threat"
    MIXED = "Mixed"


# ===========================================================================
# Transcript segment
# ===========================================================================

class TranscriptSegment(BaseModel):
    start: float = Field(..., ge=0, description="Segment start time in seconds")
    end: float = Field(..., ge=0, description="Segment end time in seconds")
    text: str = Field(..., min_length=1)
    speaker: Speaker = Speaker.CALLER
    stress_score: int = Field(..., ge=0, le=100)
    distress_score: int = Field(..., ge=0, le=100)
    emotion: Emotion
    confidence: float = Field(..., ge=0.0, le=1.0)
    indicators: list[str] = Field(default_factory=list)
    svi_score: int = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    risk_explanation: list[str] = Field(default_factory=list)


# ===========================================================================
# Full analysis response
# ===========================================================================

class SviBreakdown(BaseModel):
    stress_component: int = 0
    distress_component: int = 0
    safety_component: int = 0
    context_component: int = 0
    segment_count: int = 0
    immediate_safety: bool = False


class AnalysisResponse(BaseModel):
    case_id: str
    file_name: str
    duration_seconds: float
    transcript: list[TranscriptSegment]
    overall_stress_score: int = Field(..., ge=0, le=100)
    overall_distress_score: int = Field(..., ge=0, le=100)
    overall_svi_score: int = Field(..., ge=0, le=100)
    overall_risk_score: int = Field(..., ge=0, le=100)
    overall_risk_level: RiskLevel
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    overall_indicators: list[str] = Field(default_factory=list)
    risk_explanation: list[str] = Field(default_factory=list)
    svi_breakdown: SviBreakdown = Field(default_factory=SviBreakdown)
    recommendation: str
    mode: str = "demo"
    disclaimer: str = (
        "Assistive risk indicator, not a clinical diagnosis. "
        "High-risk indicators require trained human review."
    )
    immediate_safety_indicators: bool = False
    language: str = "en"
    analyzed_at: str = ""


# ===========================================================================
# Health / error
# ===========================================================================

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "traumasense-api"
    mode: str = "demo"


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None


# ===========================================================================
# Validation / upload metadata
# ===========================================================================

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm"}

MIME_HINTS = {
    ".wav": {"audio/wav", "audio/x-wav", "audio/wave"},
    ".mp3": {"audio/mpeg", "audio/mp3"},
    ".m4a": {"audio/mp4", "audio/x-m4a", "audio/mp4a-latm"},
    ".ogg": {"audio/ogg", "audio/vorbis"},
    ".webm": {"audio/webm", "audio/mp4"},
}


def is_allowed_extension(filename: str) -> bool:
    """Check whether the file extension is in the supported set."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


#Lazy import to avoid circular dependency at module level
import os
