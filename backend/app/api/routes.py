"""
API routes for TraumaSense.

Endpoints:
  GET  /api/health        — service health check
  GET  /api/demo          — deterministic demo case
  POST /api/analyze       — upload + analyze
  GET  /api/cases/{case_id} — retrieve a previously analyzed case
"""
from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import (
    AnalysisResponse,
    Emotion,
    ErrorResponse,
    HealthResponse,
    RiskLevel,
    Speaker,
    TranscriptSegment,
)
from app.services.analysis_service import AnalysisService
from app.services.audio_service import AudioService
from app.services.recommendation_service import get_recommendation
from app.services.risk_service import compute_risk
from app.services.svi_service import compute_overall_svi
from app.services.speech_to_text import SpeechToTextService
from app.utils.validation import validate_upload

api_router = APIRouter()

# In-memory case store for the MVP (no database required for demo mode).
_case_store: dict[str, dict[str, Any]] = {}

_case_counter = 0


def _next_case_id() -> str:
    global _case_counter
    _case_counter += 1
    return f"CASE-26093-{_case_counter:04d}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# Health
# ===========================================================================

@api_router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="traumasense-api",
        mode=settings.stt_provider,
    )


# ===========================================================================
# Demo
# ===========================================================================

@api_router.get("/demo", response_model=AnalysisResponse)
def get_demo_analysis() -> AnalysisResponse:
    """Return a complete, deterministic demo case for judge demonstration."""
    demo = _build_demo_case()
    _case_store[demo.case_id] = demo.model_dump()
    return demo


# ===========================================================================
# Analyze
# ===========================================================================

@api_router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def upload_and_analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    """Accept an audio file, validate it, run analysis, and return a case."""
    # ---- validate file presence -----------------------------------------
    if not file.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(detail="No file provided.", error_code="NO_FILE").model_dump(),
        )

    # ---- read bytes -----------------------------------------------------
    try:
        contents = await file.read()
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(detail=f"Failed to read file: {exc}", error_code="READ_ERROR").model_dump(),
        )

    file_size = len(contents)
    max_bytes = settings.max_upload_size_bytes

    # ---- validate -------------------------------------------------------
    error = validate_upload(file.filename, file_size, max_bytes)
    if error:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(detail=error, error_code="VALIDATION_ERROR").model_dump(),
        )

    # ---- store temporarily ----------------------------------------------
    audio_svc = AudioService()
    safe_path = audio_svc.store_temp(file.filename, contents)

    try:
        # ---- inspect metadata -------------------------------------------
        meta = audio_svc.inspect(safe_path)
        duration = meta.get("duration_seconds", 0.0) or 0.0

        # ---- STT --------------------------------------------------------
        stt = SpeechToTextService()
        raw_segments = stt.transcribe(safe_path, language="en", real_upload=True)

        if not raw_segments:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    detail="Speech-to-text returned no segments.",
                    error_code="STT_EMPTY",
                ).model_dump(),
            )

        # ---- analysis ---------------------------------------------------
        analysis_svc = AnalysisService()
        analyzed: list[dict[str, Any]] = []
        immediate_safety = False
        all_indicators: set[str] = set()

        for seg in raw_segments:
            text = seg.get("text", "")
            base_emotion = seg.get("emotion")
            result = analysis_svc.analyze_segment(text, base_emotion)

            stress = result["stress_score"]
            distress = result["distress_score"]
            emotion = result["emotion"]
            confidence = result["confidence"]
            indicators = result["indicators"]
            safety_flag = result["safety_flag"]
            immediate_flag = result["immediate_safety_flag"]

            if immediate_flag:
                immediate_safety = True

            for ind in indicators:
                all_indicators.add(ind)

            # Per-segment risk (assistive, not clinical)
            segment_risk = compute_risk(
                svi_score=stress,  # use stress as segment-level proxy before SVI
                overall_stress=stress,
                overall_distress=distress,
                indicators=indicators,
                confidence=confidence,
                immediate_safety=safety_flag,
            )

            analyzed.append(
                {
                    "start": round(float(seg.get("start", 0)), 2),
                    "end": round(float(seg.get("end", 0)), 2),
                    "text": text,
                    "speaker": seg.get("speaker", Speaker.CALLER.value),
                    "stress_score": stress,
                    "distress_score": distress,
                    "emotion": emotion.value if isinstance(emotion, Emotion) else emotion,
                    "confidence": confidence,
                    "indicators": indicators,
                    "safety_flag": safety_flag,
                    "immediate_safety_flag": immediate_flag,
                    "risk_level": segment_risk["risk_level"].value
                    if hasattr(segment_risk["risk_level"], "value")
                    else segment_risk["risk_level"],
                    "risk_explanation": segment_risk["explanation"],
                    "svi_score": 0,  # filled in after SVI computation
                }
            )

        # ---- SVI --------------------------------------------------------
        svi_result = compute_overall_svi(analyzed, list(all_indicators), immediate_safety)
        svi_score = svi_result.get("svi_score", 0)

        # Fill per-segment SVI based on segment stress/distress with safety boost
        for seg in analyzed:
            seg_stress = seg["stress_score"]
            seg_distress = seg["distress_score"]
            seg_safety = 50
            if seg.get("safety_flag"):
                seg_safety = 65
            if seg.get("immediate_safety_flag"):
                seg_safety = 90
            seg_svi = int(
                seg_stress * 0.30
                + seg_distress * 0.35
                + seg_safety * 0.20
                + min(len(seg.get("indicators", [])) * 18, 100) * 0.15
            )
            seg["svi_score"] = max(0, min(100, seg_svi))

        # ---- overall metrics --------------------------------------------
        overall_stress = _mean_int([s["stress_score"] for s in analyzed])
        overall_distress = _mean_int([s["distress_score"] for s in analyzed])
        overall_confidence = _mean_float([s["confidence"] for s in analyzed])

        # ---- risk -------------------------------------------------------
        risk_result = compute_risk(
            svi_score=svi_score,
            overall_stress=overall_stress,
            overall_distress=overall_distress,
            indicators=sorted(all_indicators),
            confidence=overall_confidence,
            immediate_safety=immediate_safety,
        )

        # ---- recommendation ---------------------------------------------
        recommendation = get_recommendation(
            risk_result["risk_level"],
            immediate_safety=risk_result["immediate_safety_indicators"],
        )

        # ---- build response ---------------------------------------------
        case_id = _next_case_id()
        transcript = [
            TranscriptSegment(
                start=s["start"],
                end=s["end"],
                text=s["text"],
                speaker=Speaker(s["speaker"]) if s["speaker"] in [e.value for e in Speaker] else Speaker.CALLER,
                stress_score=int(s["stress_score"]),
                distress_score=int(s["distress_score"]),
                emotion=Emotion(s["emotion"]) if s["emotion"] in [e.value for e in Emotion] else Emotion.UNCERTAINTY,
                confidence=s["confidence"],
                indicators=s["indicators"],
                svi_score=int(s["svi_score"]),
                risk_level=RiskLevel(s["risk_level"]) if s["risk_level"] in [e.value for e in RiskLevel] else RiskLevel.MODERATE,
                risk_explanation=s["risk_explanation"],
            )
            for s in analyzed
        ]

        response = AnalysisResponse(
            case_id=case_id,
            file_name=os.path.basename(file.filename),
            duration_seconds=round(duration, 2) if duration else round(len(contents) / 1_000_000 * 60.0, 2),
            transcript=transcript,
            overall_stress_score=overall_stress,
            overall_distress_score=overall_distress,
            overall_svi_score=svi_score,
            overall_risk_score=risk_result["risk_score"],
            overall_risk_level=risk_result["risk_level"],
            overall_confidence=round(overall_confidence, 2),
            overall_indicators=sorted(all_indicators),
            risk_explanation=risk_result["explanation"],
            recommendation=recommendation,
            mode=stt.provider_name(),
            analyzed_at=_now_iso(),
            immediate_safety_indicators=risk_result["immediate_safety_indicators"],
        )

        # ---- persist in-memory ------------------------------------------
        _case_store[case_id] = response.model_dump()

        return response

    finally:
        # ---- cleanup ----------------------------------------------------
        audio_svc.release(safe_path)


# ===========================================================================
# Case retrieval
# ===========================================================================

@api_router.get("/cases/{case_id}", response_model=AnalysisResponse | None)
def get_case(case_id: str) -> dict[str, Any] | None:
    """Retrieve a previously analyzed case by ID."""
    data = _case_store.get(case_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")
    return data


# ===========================================================================
# Internal helpers
# ===========================================================================

def _mean_int(values: list[int]) -> int:
    if not values:
        return 0
    return round(sum(values) / len(values))


def _mean_float(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


# ===========================================================================
# Deterministic demo case builder
# ===========================================================================

def _build_demo_case() -> AnalysisResponse:
    """Build the deterministic demo case used by GET /api/demo."""
    from app.services.speech_to_text import _build_demo_segments

    # Force demo segments directly
    segments_raw = _build_demo_segments(120.0)

    analysis_svc = AnalysisService()
    analyzed: list[dict[str, Any]] = []
    immediate_safety = False
    all_indicators: set[str] = set()

    for seg in segments_raw:
        # Use pre-assigned demo scores directly — they are calibrated
        # to produce the intended progressive narrative arc.
        text = seg.get("text", "")
        stress = int(seg.get("stress_score", 0))
        distress = int(seg.get("distress_score", 0))
        emotion = seg.get("emotion", "Uncertainty")
        confidence = float(seg.get("confidence", 0.75))
        indicators = list(seg.get("indicators", []))
        safety_flag = bool(seg.get("safety_flag", False))
        immediate_flag = bool(seg.get("immediate_safety_flag", False))

        if immediate_flag:
            immediate_safety = True
        for ind in indicators:
            all_indicators.add(ind)

        segment_risk = compute_risk(
            svi_score=stress,
            overall_stress=stress,
            overall_distress=distress,
            indicators=indicators,
            confidence=confidence,
            immediate_safety=safety_flag,
        )

        analyzed.append(
            {
                "start": round(float(seg.get("start", 0)), 2),
                "end": round(float(seg.get("end", 0)), 2),
                "text": text,
                "speaker": seg.get("speaker", Speaker.CALLER.value),
                "stress_score": stress,
                "distress_score": distress,
                "emotion": emotion.value if isinstance(emotion, Emotion) else emotion,
                "confidence": confidence,
                "indicators": indicators,
                "safety_flag": safety_flag,
                "immediate_safety_flag": immediate_flag,
                "risk_level": segment_risk["risk_level"].value
                if hasattr(segment_risk["risk_level"], "value")
                else segment_risk["risk_level"],
                "risk_explanation": segment_risk["explanation"],
                "svi_score": 0,
            }
        )

    svi_result = compute_overall_svi(analyzed, list(all_indicators), immediate_safety)
    svi_score = svi_result.get("svi_score", 0)

    # Fill per-segment SVI
    for seg in analyzed:
        seg_stress = int(seg["stress_score"])
        seg_distress = int(seg["distress_score"])
        seg_safety = 50
        if seg.get("safety_flag"):
            seg_safety = 65
        if seg.get("immediate_safety_flag"):
            seg_safety = 90
        seg_svi = int(
            seg_stress * 0.30
            + seg_distress * 0.35
            + seg_safety * 0.20
            + min(len(seg.get("indicators", [])) * 18, 100) * 0.15
        )
        seg["svi_score"] = max(0, min(100, seg_svi))
        # Ensure integer scores for Pydantic
        seg["stress_score"] = int(seg["stress_score"])
        seg["distress_score"] = int(seg["distress_score"])

    overall_stress = _mean_int([s["stress_score"] for s in analyzed])
    overall_distress = _mean_int([s["distress_score"] for s in analyzed])
    overall_confidence = _mean_float([s["confidence"] for s in analyzed])

    risk_result = compute_risk(
        svi_score=svi_score,
        overall_stress=overall_stress,
        overall_distress=overall_distress,
        indicators=sorted(all_indicators),
        confidence=overall_confidence,
        immediate_safety=immediate_safety,
    )

    recommendation = get_recommendation(
        risk_result["risk_level"],
        immediate_safety=risk_result["immediate_safety_indicators"],
    )

    transcript = [
        TranscriptSegment(
            start=s["start"],
            end=s["end"],
            text=s["text"],
            speaker=Speaker(s["speaker"]) if s["speaker"] in [e.value for e in Speaker] else Speaker.CALLER,
            stress_score=s["stress_score"],
            distress_score=s["distress_score"],
            emotion=Emotion(s["emotion"]) if s["emotion"] in [e.value for e in Emotion] else Emotion.UNCERTAINTY,
            confidence=s["confidence"],
            indicators=s["indicators"],
            svi_score=s["svi_score"],
            risk_level=RiskLevel(s["risk_level"]) if s["risk_level"] in [e.value for e in RiskLevel] else RiskLevel.MODERATE,
            risk_explanation=s["risk_explanation"],
        )
        for s in analyzed
    ]

    return AnalysisResponse(
        case_id="CASE-26093-0001",
        file_name="demo_call.wav",
        duration_seconds=120.0,
        transcript=transcript,
        overall_stress_score=overall_stress,
        overall_distress_score=overall_distress,
        overall_svi_score=svi_score,
        overall_risk_score=risk_result["risk_score"],
        overall_risk_level=risk_result["risk_level"],
        overall_confidence=round(overall_confidence, 2),
        overall_indicators=sorted(all_indicators),
        risk_explanation=risk_result["explanation"],
        recommendation=recommendation,
        mode="demo",
        analyzed_at=_now_iso(),
        immediate_safety_indicators=risk_result["immediate_safety_indicators"],
    )
