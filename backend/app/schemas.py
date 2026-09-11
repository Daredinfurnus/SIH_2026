"""
Pydantic schemas for the TraumaSense API contract.

These are the source of truth for request/response shapes.
The frontend API client and the backend both honour these.

NEW (scoring engine v2):
  AnalysisResult    — unified engine output (internal + new fields)
  TimelineRecord    — one per-segment point; frontend MUST use this for graphs
  DimensionScore    — score + confidence + evidence_quality + contributors
  SviScore          — composite with components + interactions
  SafetyScore       — score + severity + immediate flag
  ConfidenceBreakdown
  ExplanationSet
  Contributor, SviComponent, InteractionTerm
  SafetySeverity, EvidenceQuality

LEGACY (kept for API backward compat — populated FROM the new engine):
  AnalysisResponse  — what the frontend currently reads
  TranscriptSegment — per-segment in the legacy "transcript" array
  SviBreakdown      — legacy flat SVI breakdown
  ModelStatus       — model availability status
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

import os


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


class SafetySeverity(str, Enum):
    NONE = "NONE"
    GENERAL_CONCERN = "GENERAL_CONCERN"
    ELEVATED_CONCERN = "ELEVATED_CONCERN"
    IMMEDIATE_CONCERN = "IMMEDIATE_CONCERN"


class EvidenceQuality(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    LIMITED = "LIMITED"
    INSUFFICIENT = "INSUFFICIENT"


# ===========================================================================
# NEW — unified scoring engine types
# ===========================================================================

class Contributor(BaseModel):
    """One scored contributor to a dimension (for explanations/debug)."""
    source: str
    normalized_value: float = Field(..., ge=0, le=100)
    base_weight: float = Field(..., ge=0, le=1)
    effective_weight: float = Field(..., ge=0, le=1)
    contribution: float


class SviComponent(BaseModel):
    """One SVI component breakdown."""
    component: str
    value: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=100)
    base_weight: float = Field(..., ge=0, le=1)
    effective_weight: float = Field(..., ge=0, le=1)
    contribution: float


class InteractionTerm(BaseModel):
    """An explicit SVI interaction term."""
    name: str
    formula: str
    value: float
    max_contribution: float


class DimensionScore(BaseModel):
    """A single dimension's score with confidence and evidence quality."""
    score: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=100)
    evidence_quality: str = "SUFFICIENT"
    contributors: list[Contributor] = Field(default_factory=list)
    missing_signals: list[str] = Field(default_factory=list)


class SviScore(BaseModel):
    """SVI composite with components and interactions."""
    svi_base: float = Field(..., ge=0, le=100)
    interaction_total: float = 0.0
    svi_final: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=100)
    components: list[SviComponent] = Field(default_factory=list)
    interactions: list[InteractionTerm] = Field(default_factory=list)


class SafetyScore(BaseModel):
    """Safety dimension with severity."""
    score: float = Field(..., ge=0, le=100)
    severity: SafetySeverity
    immediate_safety_flag: bool = False
    confidence: float = Field(..., ge=0, le=100)
    contributors: list[Contributor] = Field(default_factory=list)


class ConfidenceBreakdown(BaseModel):
    """Confidence dimension with component breakdown."""
    score: float = Field(..., ge=0, le=100)
    components: dict[str, float] = Field(default_factory=dict)
    component_weights: dict[str, float] = Field(default_factory=dict)


class TimelineRecord(BaseModel):
    """One per-segment timeline point — the single source of truth for graphs."""
    timestamp: float
    duration: float
    stress: float = Field(..., ge=0, le=100)
    distress: float = Field(..., ge=0, le=100)
    safety: float = Field(..., ge=0, le=100)
    svi: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    contributors: list[Contributor] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)


class ExplanationSet(BaseModel):
    """Human-readable explanations for all dimensions."""
    stress: DimensionScore
    distress: DimensionScore
    svi: SviScore
    safety: SafetyScore
    confidence: ConfidenceBreakdown


class AnalysisResult(BaseModel):
    """Unified analysis result — the single authoritative output of the scoring engine.

    The route handler maps this to the legacy AnalysisResponse for the frontend.
    """
    # Overall (confidence-weighted mean of timeline)
    overall_stress_score: float = Field(..., ge=0, le=100)
    overall_distress_score: float = Field(..., ge=0, le=100)
    overall_safety_score: float = Field(..., ge=0, le=100)
    overall_svi_score: float = Field(..., ge=0, le=100)
    overall_confidence: float = Field(..., ge=0, le=100)
    overall_risk_level: RiskLevel
    safety_severity: SafetySeverity = SafetySeverity.NONE
    safety_immediate_flag: bool = False
    timeline_length: int = 0

    # Timeline — ONE record per analyzed segment. Frontend must use this, NOT recompute.
    timeline: list[TimelineRecord] = Field(default_factory=list)

    # Explanations
    explanations: ExplanationSet | None = None

    # Metadata for case storage
    case_id: str = ""
    file_name: str = ""
    duration_seconds: float = 0.0
    recommendation: str = ""
    disclaimer: str = ""
    mode: str = "whisper"
    language: str = "en"
    analyzed_at: str = ""
    model_status: dict[str, str] = Field(default_factory=dict)
    detected_language: str = "auto-detected"
    indicators: list[str] = Field(default_factory=list)


# ===========================================================================
# LEGACY — kept for API backward compat; populated FROM AnalysisResult
# ===========================================================================

class SviBreakdown(BaseModel):
    stress_component: int = 0
    distress_component: int = 0
    safety_component: int = 0
    context_component: int = 0
    segment_count: int = 0
    immediate_safety: bool = False


class ModelStatus(BaseModel):
    """Status of each AI model that participated in the analysis."""
    asr: str = "unknown"
    text_emotion: str = "unknown"
    acoustic_emotion: str = "unknown"
    fusion: str = "unknown"


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
    # Extra fields the frontend may read (from emotion service)
    emotion_explanation: list[str] = Field(default_factory=list)
    accent_signals: dict[str, Any] | None = None


class AnalysisResponse(BaseModel):
    """Legacy response shape — populated from AnalysisResult by the route handler.

    The frontend reads: transcript[], overall_stress_score, overall_distress_score,
    overall_svi_score, overall_risk_score, overall_risk_level, overall_confidence,
    overall_indicators, risk_explanation, svi_breakdown, recommendation,
    immediate_safety_indicators, disclaimer, mode, language, analyzed_at,
    model_status, detected_language.
    """
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
    mode: str = "whisper"
    disclaimer: str = (
        "Assistive risk indicator, not a clinical diagnosis. "
        "High-risk indicators require trained human review."
    )
    immediate_safety_indicators: bool = False
    language: str = "en"
    analyzed_at: str = ""
    model_status: ModelStatus = Field(default_factory=ModelStatus)
    detected_language: str = "auto-detected"


# ===========================================================================
# Health / error
# ===========================================================================

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "traumasense-api"
    mode: str = "whisper"


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None


# ===========================================================================
# Validation / upload metadata
# ===========================================================================

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".webm", ".mp4"}

MIME_HINTS = {
    ".wav": {"audio/wav", "audio/x-wav", "audio/wave"},
    ".mp3": {"audio/mpeg", "audio/mp3"},
    ".m4a": {"audio/mp4", "audio/x-m4a", "audio/mp4a-latm"},
    ".aac": {"audio/aac", "audio/aacp", "audio/mp4a-latm", "audio/mp4"},
    ".ogg": {"audio/ogg", "audio/vorbis"},
    ".webm": {"audio/webm", "audio/mp4"},
    ".mp4": {"audio/mp4", "video/mp4"},
}


def is_allowed_extension(filename: str) -> bool:
    """Check whether the file extension is in the supported set."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS
