"""
API routes for TraumaSense.

Endpoints:
  GET  /api/health        — service health check
  POST /api/analyze       — upload + analyze
  GET  /api/cases/{case_id} — retrieve a previously analyzed case
"""
from __future__ import annotations

import os
import time
import uuid
import logging
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
from app.services.emotion_service import get_emotion_service
from app.services.firebase_store import cloud_load_all, cloud_save, is_ready
from app.utils.validation import validate_upload

api_router = APIRouter()

# In-memory case store for the MVP (upload-only).
_case_store: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Persistence — load previously saved cases from Firestore on startup.
# When Firebase is unavailable we stay in-memory only; the API contract is
# unchanged either way.
# ---------------------------------------------------------------------------
_loaded_from_firestore = cloud_load_all()
_case_store.update(_loaded_from_firestore)
if _loaded_from_firestore:
    logger.info("Pre-loaded %d cases from Firestore into in-memory store", len(_loaded_from_firestore))
else:
    logger.info("No cases loaded from Firestore (empty or unavailable)")

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
        raw_segments = stt.transcribe(safe_path, language=None, real_upload=True)

        if not raw_segments:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    detail="Speech-to-text returned no segments.",
                    error_code="STT_EMPTY",
                ).model_dump(),
            )

        # ---- analysis ---------------------------------------------------
        ai_provider = settings.ai_provider
        analysis_svc = AnalysisService(ai_provider=ai_provider)
        emotion_svc = get_emotion_service()
        analyzed: list[dict[str, Any]] = []
        immediate_safety = False
        all_indicators: set[str] = set()

        for seg in raw_segments:
            text = seg.get("text", "")
            base_emotion = seg.get("emotion")
            result = analysis_svc.analyze_segment(text, base_emotion)

            stress = result["stress_score"]
            distress = result["distress_score"]
            confidence = result["confidence"]
            indicators = result["indicators"]
            safety_flag = result["safety_flag"]
            immediate_flag = result["immediate_safety_flag"]

            if immediate_flag:
                immediate_safety = True

            for ind in indicators:
                all_indicators.add(ind)

            # ---- Emotion analysis (text + acoustic) ----
            emotion_result = emotion_svc.analyze_segment(
                text=text,
                audio_path=safe_path,
                start_sec=float(seg.get("start", 0)),
                end_sec=float(seg.get("end", 0)),
            )
            emotion_label = emotion_result["emotion"]
            emotion_confidence = max(emotion_result["text_confidence"], emotion_result["acoustic_confidence"])

            # Per-segment risk (assistive, not clinical)
            segment_risk = compute_risk(
                svi_score=stress,  # use stress as segment-level proxy before SVI
                overall_stress=stress,
                overall_distress=distress,
                indicators=indicators,
                confidence=max(confidence, emotion_confidence),
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
                    "emotion": emotion_label,
                    "confidence": max(confidence, emotion_confidence),
                    "indicators": indicators,
                    "safety_flag": safety_flag,
                    "immediate_safety_flag": immediate_flag,
                    "risk_level": segment_risk["risk_level"].value
                    if hasattr(segment_risk["risk_level"], "value")
                    else segment_risk["risk_level"],
                    "risk_explanation": segment_risk["explanation"],
                    "svi_score": 0,  # filled in after SVI computation
                    "emotion_explanation": emotion_result.get("emotion_explanation", []),
                    "accent_signals": emotion_result.get("accent_signals"),
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
                emotion_explanation=s.get("emotion_explanation", []),
                accent_signals=s.get("accent_signals"),
            )
            for s in analyzed
        ]

        response = AnalysisResponse(
            case_id=case_id,
            file_name=os.path.basename(file.filename),
            duration_seconds=duration,
            transcript=transcript,
            overall_stress=overall_stress,
            overall_distress=overall_distress,
            overall_confidence=overall_confidence,
            svi_score=int(svi_score),
            risk_level=risk_result["risk_level"].value
            if hasattr(risk_result["risk_level"], "value")
            else risk_result["risk_level"],
            risk_explanation=risk_result["explanation"],
            recommendation=recommendation,
            mode="upload",
            analyzed_at=_now_iso(),
        )

        _case_store[case_id] = response.model_dump()
        cloud_save(case_id, response.model_dump())
        return response

    finally:
        # Clean up temp file.
        try:
            os.remove(safe_path)
        except OSError:
            pass


# ===========================================================================
# Cases
# ===========================================================================

@api_router.get("/cases/{case_id}", response_model=AnalysisResponse | None)
def get_case(case_id: str) -> dict[str, Any] | None:
    """Retrieve a previously analyzed case by ID."""
    data = _case_store.get(case_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")
    return data


# ===========================================================================
# Helpers
# ===========================================================================

def _mean_int(values: list[int]) -> int:
    if not values:
        return 0
    return round(sum(values) / len(values))


def _mean_float(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)
