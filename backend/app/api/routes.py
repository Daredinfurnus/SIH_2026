"""
API routes for TraumaSense.
Endpoints:
  GET  /api/health        — service health check
  POST /api/analyze       — upload + analyze
  GET  /api/cases/{case_id} — retrieve a previously analyzed case
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import (
    AnalysisResponse,
    Emotion,
    ErrorResponse,
    HealthResponse,
    ModelStatus,
    RiskLevel,
    Speaker,
    TranscriptSegment,
)
from app.services.analysis_service import AnalysisService
from app.services.audio_service import AudioService
from app.services.normalization_service import normalize_audio, cleanup_normalized
from app.services.recommendation_service import get_recommendation
from app.services.risk_service import compute_risk
from app.services.svi_service import compute_overall_svi
from app.services.speech_to_text import SpeechToTextService
from app.services.emotion_service import get_emotion_service
from app.services.nlp_service import get_nlp_service
from app.utils.validation import validate_upload

logger = logging.getLogger(__name__)

api_router = APIRouter()

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
# Analyze
# ===========================================================================

@api_router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def upload_and_analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    """Accept an audio file, validate it, run analysis, and return a case."""
    if not file.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(detail="No file provided.", error_code="NO_FILE").model_dump(),
        )

    try:
        contents = await file.read()
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(detail=f"Failed to read file: {exc}", error_code="READ_ERROR").model_dump(),
        )

    file_size = len(contents)
    max_bytes = settings.max_upload_size_bytes

    error = validate_upload(file.filename, file_size, max_bytes)
    if error:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(detail=error, error_code="VALIDATION_ERROR").model_dump(),
        )

    audio_svc = AudioService()
    safe_path = audio_svc.store_temp(file.filename, contents)

    normalized_path = None

    try:
        meta = audio_svc.inspect(safe_path)
        duration = meta.get("duration_seconds", 0.0) or 0.0

        # ---- normalization ------------------------------------------------------
        norm_result = normalize_audio(safe_path, settings.upload_dir)
        normalized_path = norm_result.get("normalized_path")
        original_format = norm_result.get("original_format", "unknown")
        ffmpeg_ok = norm_result.get("ffmpeg_available", False)
        norm_error = norm_result.get("error")

        if norm_error and not ffmpeg_ok:
            # FFmpeg unavailable is a warning, not a fatal error — try with original
            logger.warning("Audio normalization skipped: %s", norm_error)

        # ---- STT --------------------------------------------------------
        stt = SpeechToTextService()
        raw_segments = stt.transcribe(normalized_path or safe_path, language=None, real_upload=True)

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
            # Use normalized audio path for acoustic analysis if available
            audio_for_acoustic = normalized_path or safe_path
            emotion_result = emotion_svc.analyze_segment(
                text=text,
                audio_path=audio_for_acoustic,
                start_sec=float(seg.get("start", 0)),
                end_sec=float(seg.get("end", 0)),
            )
            emotion_label = emotion_result["emotion"]
            emotion_confidence = max(emotion_result["text_confidence"], emotion_result["acoustic_confidence"])

            segment_risk = compute_risk(
                svi_score=stress,
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
                    "svi_score": 0,
                    "emotion_explanation": emotion_result.get("emotion_explanation", []),
                    "accent_signals": emotion_result.get("accent_signals"),
                }
            )

        # ---- SVI --------------------------------------------------------
        svi_result = compute_overall_svi(analyzed, list(all_indicators), immediate_safety)
        svi_score = svi_result.get("svi_score", 0)

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
        # Determine detected language from first segment
        detected_lang = "en"
        for seg in raw_segments:
            dl = seg.get("detected_language")
            if dl and dl not in ("auto-detected", None):
                detected_lang = dl
                break

        # Model status tracking
        asr_status = "success" if raw_segments else "failed"
        
        # Check NLP status
        nlp_status = "unavailable"
        try:
            nlp_svc = get_nlp_service() if get_nlp_service else None
            if nlp_svc is not None:
                nlp_status = "success"
        except Exception:
            nlp_status = "failed"
        
        # Check acoustic status from emotion results
        acoustic_status = "unavailable"
        has_acoustic = any(
            s.get("emotion_source") in ("acoustic", "fused")
            for s in analyzed
        )
        if has_acoustic:
            acoustic_status = "success"
        elif emotion_svc is not None:
            acoustic_status = "unavailable"  # service exists but no acoustic data

        # Determine fusion mode
        fusion_mode = "none"
        text_only = sum(1 for s in analyzed if s.get("emotion_source") == "text")
        acoustic_only = sum(1 for s in analyzed if s.get("emotion_source") == "acoustic")
        fused = sum(1 for s in analyzed if s.get("emotion_source") == "fused")
        if fused > 0:
            fusion_mode = "multimodal"
        elif text_only > 0 and acoustic_only > 0:
            fusion_mode = "multimodal"
        elif text_only > 0:
            fusion_mode = "text_only"
        elif acoustic_only > 0:
            fusion_mode = "acoustic_only"

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
            svi_breakdown=svi_result.get("svi_breakdown", {}),
            recommendation=recommendation,
            mode=stt.provider_name(),
            analyzed_at=_now_iso(),
            immediate_safety_indicators=risk_result["immediate_safety_indicators"],
            model_status=ModelStatus(
                asr=asr_status,
                text_emotion=nlp_status,
                acoustic_emotion=acoustic_status,
                fusion=fusion_mode,
            ),
            language=detected_lang,
            detected_language=detected_lang,
        )

        _case_store[case_id] = response.model_dump()
        return response

    finally:
        # Clean up normalized file if we created one
        if normalized_path and normalized_path != safe_path:
            cleanup_normalized(normalized_path)
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
