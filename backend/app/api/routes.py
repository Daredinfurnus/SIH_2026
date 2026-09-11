"""
API routes for TraumaSense — wired to the unified scoring engine.

The scoring engine (app.services.scoring_engine) is the SINGLE AUTHORITATIVE
source of all scores.  This module runs the feature-extraction pipeline
(STT → NLP → emotion → indicators), feeds the extracted features into the
engine via build_features_from_analysis_segment(), and maps the engine's
AnalysisResult back to the legacy AnalysisResponse for the frontend.
"""
from __future__ import annotations

import os
import time
import math
import uuid
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

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
    SafetySeverity,
    Speaker,
    SviBreakdown,
    TranscriptSegment,
)
from app.services.analysis_service import AnalysisService
from app.services.audio_service import AudioService
from app.services.normalization_service import normalize_audio, cleanup_normalized
from app.services.recommendation_service import get_recommendation
from app.services.speech_to_text import SpeechToTextService
from app.services.emotion_service import get_emotion_service
from app.services.firebase_store import cloud_load_all, cloud_save, is_ready
from app.services.scoring_engine import (
    ScoringPipeline,
    build_features_from_analysis_segment,
)
from app.services.language_identifier import (
    get_language_identifier,
    route_language,
)
from app.utils.validation import validate_upload

api_router = APIRouter()

_case_store: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
_loaded_from_firestore = cloud_load_all()
_case_store.update(_loaded_from_firestore)
if _loaded_from_firestore:
    logger.info("Pre-loaded %d cases from Firestore into in-memory store",
                len(_loaded_from_firestore))
else:
    logger.info("No cases loaded from Firestore (empty or unavailable)")

_case_counter = 0


def _next_case_id() -> str:
    global _case_counter
    _case_counter += 1
    return f"CASE-26093-{_case_counter:04d}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_risk_explanation(
    risk_level: str,
    safety_severity: str,
    svi_score: float,
    explanations: dict,
    indicators: list[str],
    immediate_safety: bool,
) -> list[str]:
    """Build human-readable risk explanation from engine results."""
    parts: list[str] = []
    if immediate_safety:
        parts.append(
            "Immediate safety indicators detected — trained human review required."
        )
    if safety_severity == "IMMEDIATE_CONCERN":
        parts.append(f"Safety severity: IMMEDIATE CONCERN. SVI: {int(svi_score)}/100.")
    elif safety_severity == "ELEVATED_CONCERN":
        parts.append(f"Safety severity: ELEVATED CONCERN. SVI: {int(svi_score)}/100.")
    elif safety_severity == "GENERAL_CONCERN":
        parts.append(f"Safety concern noted. SVI: {int(svi_score)}/100.")

    # Add dimension explanations
    if explanations:
        stress_exp = explanations.get("stress", {})
        distress_exp = explanations.get("distress", {})
        svi_exp = explanations.get("svi", {})

        if stress_exp.get("score", 0) > 50:
            parts.append(
                f"Elevated stress indicators detected (score: {int(stress_exp['score'])}/100)."
            )
        if distress_exp.get("score", 0) > 50:
            parts.append(
                f"Elevated distress indicators detected (score: {int(distress_exp['score'])}/100)."
            )
        if svi_exp.get("svi_final", 0) > 50:
            parts.append(
                f"Higher vulnerability indicators detected (SVI: {int(svi_exp['svi_final'])}/100)."
            )

    if not parts:
        parts.append(
            f"Overall assessment: {risk_level} risk (SVI: {int(svi_score)}/100). "
            "Assistive indicator — trained human review recommended for clinical interpretation."
        )

    return parts


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
# Analyze — single authoritative pipeline
# ===========================================================================

@api_router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def upload_and_analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    """Upload audio → STT → feature extraction → unified scoring engine → response."""
    if not file.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                detail="No file provided.", error_code="NO_FILE"
            ).model_dump(),
        )

    try:
        contents = await file.read()
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                detail=f"Failed to read file: {exc}", error_code="READ_ERROR"
            ).model_dump(),
        )

    file_size = len(contents)
    max_bytes = settings.max_upload_size_bytes

    error = validate_upload(file.filename, file_size, max_bytes)
    if error:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                detail=error, error_code="VALIDATION_ERROR"
            ).model_dump(),
        )

    audio_svc = AudioService()
    safe_path = audio_svc.store_temp(file.filename, contents)
    normalized_path = None

    try:
        # ---- audio normalization-------------------------------------------
        meta = audio_svc.inspect(safe_path)
        duration = meta.get("duration_seconds", 0.0) or 0.0

        norm_result = normalize_audio(safe_path, settings.upload_dir)
        normalized_path = norm_result.get("normalized_path")
        ffmpeg_ok = norm_result.get("ffmpeg_available", False)
        norm_error = norm_result.get("error")

        if norm_error and not ffmpeg_ok:
            logger.warning("Audio normalization skipped: %s", norm_error)

        # ---- language identification (BEFORE ASR) ----------------------
        #
        # VoxLingua107 is the ONLY authority for language routing.
        #
        # Supported:
        #   en -> Faster-Whisper
        #   hi -> IndicConformer
        #
        # Unsupported:
        #   ur / every other language
        #
        # If LID fails, is unavailable, uncertain, or unsupported:
        # STOP.
        #
        # Whisper MUST NEVER perform language detection or routing.

        audio_path = normalized_path or safe_path

        if not audio_path:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(
                    detail="No valid audio path is available for language identification.",
                    error_code="INVALID_AUDIO",
                ).model_dump(),
            )

        # --------------------------------------------------------------
        # Run authoritative language identification.
        # --------------------------------------------------------------
        try:
            lid = get_language_identifier()
            lid_result = lid.identify(audio_path)

        except Exception:
            logger.exception(
                "Language identification failed for audio=%s",
                audio_path,
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    detail=(
                        "Language identification failed. "
                        "ASR was not executed."
                    ),
                    error_code="LANGUAGE_DETECTION_FAILED",
                ).model_dump(),
            )

        # --------------------------------------------------------------
        # Validate the LID response before using it.
        # --------------------------------------------------------------
        if not isinstance(lid_result, dict):
            logger.error(
                "Invalid LID result type: %s",
                type(lid_result).__name__,
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    detail=(
                        "Language identification returned an invalid result. "
                        "ASR was not executed."
                    ),
                    error_code="LANGUAGE_DETECTION_FAILED",
                ).model_dump(),
            )

        # --------------------------------------------------------------
        # LID model MUST be available.
        #
        # Never run ASR without authoritative LID.
        # --------------------------------------------------------------
        if not bool(lid_result.get("model_available", False)):
            logger.error(
                "Language identification model unavailable."
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    detail=(
                        "Language identification model is unavailable. "
                        "ASR was not executed."
                    ),
                    error_code="LANGUAGE_DETECTION_FAILED",
                ).model_dump(),
            )

        # --------------------------------------------------------------
        # Convert authoritative LID output into an ASR route.
        #
        # route_language() is fail-closed:
        #   en -> en
        #   hi -> hi
        #   everything else -> halt
        # --------------------------------------------------------------
        try:
            routing = route_language(lid_result)

        except Exception:
            logger.exception(
                "Language routing failed. LID result=%r",
                lid_result,
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    detail=(
                        "Language routing failed. "
                        "ASR was not executed."
                    ),
                    error_code="LANGUAGE_DETECTION_FAILED",
                ).model_dump(),
            )

        # --------------------------------------------------------------
        # Routing MUST be a valid dictionary.
        # --------------------------------------------------------------
        if not isinstance(routing, dict):
            logger.error(
                "Invalid language routing result: %r",
                routing,
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    detail=(
                        "Language routing returned an invalid decision. "
                        "ASR was not executed."
                    ),
                    error_code="LANGUAGE_DETECTION_FAILED",
                ).model_dump(),
            )

        # --------------------------------------------------------------
        # Log the authoritative language-routing decision.
        # --------------------------------------------------------------
        logger.info(
            "LANGUAGE ROUTING: lid_language=%s confidence=%.3f route=%s",
            lid_result.get("language"),
            float(lid_result.get("confidence", 0.0)),
            routing.get("route"),
        )


        # --------------------------------------------------------------
        # Halt on:
        #   - low confidence
        #   - uncertain language
        #   - Urdu
        #   - unsupported language
        #   - any other fail-closed condition
        # --------------------------------------------------------------
        if bool(routing.get("halt", True)):
            language = routing.get(
                "language",
                lid_result.get("language", "language_uncertain"),
            )

            if not isinstance(language, str):
                language = "language_uncertain"

            language = language.strip().lower()

            try:
                confidence = float(
                    lid_result.get("confidence", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                confidence = 0.0

            # Treat malformed confidence as uncertain.
            if not math.isfinite(confidence):
                confidence = 0.0

            # ----------------------------------------------------------
            # Low-confidence / uncertain language.
            # ----------------------------------------------------------
            if (
                language == "language_uncertain"
                or confidence < LID_CONFIDENCE_THRESHOLD
            ):
                error_code = "LANGUAGE_CONFIDENCE_TOO_LOW"

            # ----------------------------------------------------------
            # Explicitly unsupported language.
            # This includes Urdu and every language outside en/hi.
            # ----------------------------------------------------------
            else:
                error_code = "UNSUPPORTED_LANGUAGE"

            logger.warning(
                "ASR halted by language routing: "
                "language=%s confidence=%.3f reason=%s",
                language,
                confidence,
                routing.get("reason", "No reason provided."),
            )

            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=ErrorResponse(
                    detail=routing.get(
                        "reason",
                        "Language is unsupported or insufficiently confident.",
                    ),
                    error_code=error_code,
                ).model_dump(),
            )

        # --------------------------------------------------------------
        # The router MUST explicitly return one of:
        #
        #   "en" -> Faster-Whisper
        #   "hi" -> IndicConformer
        #
        # Anything else is an internal routing failure.
        # --------------------------------------------------------------
        forced_lang = routing.get("language")

        if forced_lang not in {"en", "hi"}:
            logger.error(
                "Invalid ASR routing decision: route=%r language=%r",
                routing.get("route"),
                forced_lang,
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    detail=(
                        "Invalid ASR routing decision. "
                        "ASR was not executed."
                    ),
                    error_code="LANGUAGE_DETECTION_FAILED",
                ).model_dump(),
            )

        # --------------------------------------------------------------
        # Make sure the route itself agrees with the language.
        # --------------------------------------------------------------
        if routing.get("route") != forced_lang:
            logger.error(
                "Inconsistent ASR routing decision: %r",
                routing,
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    detail=(
                        "Inconsistent language routing decision. "
                        "ASR was not executed."
                    ),
                    error_code="LANGUAGE_DETECTION_FAILED",
                ).model_dump(),
            )

        # --------------------------------------------------------------
        # speech-to-text
        #
        # IMPORTANT:
        # forced_lang is authoritative.
        #
        # en -> Faster-Whisper(language="en")
        # hi -> IndicConformer(language="hi")
        #
        # There is NO language=None / auto-detect path.
        # --------------------------------------------------------------
        stt = SpeechToTextService()

        try:
            raw_segments = stt.transcribe(
                audio_path,
                language=forced_lang,
            )

        except RuntimeError as exc:
            logger.exception(
                "ASR failed for language=%s",
                forced_lang,
            )

            message = str(exc)
            message_lower = message.lower()

            if "model load failed" in message_lower:
                error_code = "ASR_MODEL_LOAD_FAILED"
            else:
                error_code = "ASR_INFERENCE_FAILED"

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    detail=message or "Speech-to-text inference failed.",
                    error_code=error_code,
                ).model_dump(),
            )

        except ValueError as exc:
            # Invalid language/audio/model input should not be reported
            # as a successful analysis.
            logger.exception(
                "Invalid ASR input for language=%s",
                forced_lang,
            )

            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=ErrorResponse(
                    detail=str(exc) or "Invalid speech-to-text input.",
                    error_code="ASR_INFERENCE_FAILED",
                ).model_dump(),
            )

        except Exception as exc:
            logger.exception(
                "Unexpected ASR failure for language=%s",
                forced_lang,
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    detail=f"Speech-to-text failed: {exc}",
                    error_code="ASR_INFERENCE_FAILED",
                ).model_dump(),
            )

        # --------------------------------------------------------------
        # ASR MUST return actual segments.
        #
        # No placeholder/fake transcript is allowed.
        # --------------------------------------------------------------
        if raw_segments is None:
            logger.error(
                "ASR returned None for language=%s",
                forced_lang,
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    detail="Speech-to-text returned no result.",
                    error_code="EMPTY_TRANSCRIPT",
                ).model_dump(),
            )

        if not isinstance(raw_segments, (list, tuple)):
            logger.error(
                "ASR returned invalid segment type: %s",
                type(raw_segments).__name__,
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    detail=(
                        "Speech-to-text returned an invalid segment result."
                    ),
                    error_code="ASR_INFERENCE_FAILED",
                ).model_dump(),
            )

        if len(raw_segments) == 0:
            logger.warning(
                "ASR returned zero segments for language=%s",
                forced_lang,
            )

            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=ErrorResponse(
                    detail="No speech could be transcribed from the audio.",
                    error_code="EMPTY_TRANSCRIPT",
                ).model_dump(),
            )

        ai_provider = settings.ai_provider
        analysis_svc = AnalysisService(ai_provider=ai_provider)
        emotion_svc = get_emotion_service()

        # ---- feature extraction + scoring (unified engine) ------------------
        segments_for_engine: list[dict[str, Any]] = []
        features_by_segment: list[dict[str, Any]] = []

        immediate_safety = False
        all_indicators: set[str] = set()

        for seg in raw_segments:
            text = seg.get("text", "")
            base_emotion = seg.get("emotion")

            # ---- analysis service (NLP lexical indicators) ------------------
            result = analysis_svc.analyze_segment(text, base_emotion)
            segment_indicators = result.get("indicators", [])
            for ind in segment_indicators:
                all_indicators.add(ind)
            immediate_flag = result.get("immediate_safety_flag", False)
            if immediate_flag:
                immediate_safety = True

            # ---- emotion service (text + acoustic) --------------------------
            audio_for_acoustic = normalized_path or safe_path
            emotion_result = emotion_svc.analyze_segment(
                text=text,
                audio_path=audio_for_acoustic,
                start_sec=float(seg.get("start", 0)),
                end_sec=float(seg.get("end", 0)),
            )
            emotion_label = emotion_result["emotion"]
            emotion_confidence = max(
                emotion_result["text_confidence"],
                emotion_result["acoustic_confidence"],
            )

            # ---- assemble analyzed segment dict (feature source) -----------
            analyzed_seg = {
                "start_time": float(seg.get("start", 0)),
                "duration": float(seg.get("end", 0)) - float(seg.get("start", 0)),
                "text": text,
                "speaker": seg.get("speaker", Speaker.CALLER.value),
                "stress_indicator": result.get("stress_score"),
                "distress_indicator": result.get("distress_score"),
                "confidence": result.get("confidence", 0.5),
                "emotion": {
                    "primary": emotion_label,
                    "confidence": emotion_confidence,
                    "arousal": emotion_result.get("arousal", 0.5),
                },
                "indicators": segment_indicators,
                "safety_flag": result.get("safety_flag", False),
                "immediate_safety_flag": immediate_flag,
                "context_score": result.get("context_score", 0.0),
                "acoustic": emotion_result.get("acoustic", {}),
                "emotion_explanation": emotion_result.get("emotion_explanation", []),
                "accent_signals": emotion_result.get("accent_signals"),
            }
            segments_for_engine.append(analyzed_seg)

            # ---- build features from this segment (bridge to engine) -------
            features = build_features_from_analysis_segment(analyzed_seg)
            features_by_segment.append(features)

        # ---- run the unified scoring pipeline on ALL segments --------------
        pipeline = ScoringPipeline()
        engine_result = pipeline.analyze_full_conversation(
            segments=segments_for_engine,
            features_by_segment=features_by_segment,
        )

        # ---- build AnalysisResponse (legacy shape) from engine result -------
        case_id = _next_case_id()

                # --------------------------------------------------------------
        # Detected language
        #
        # forced_lang is the authoritative language selected by
        # VoxLingua107 before ASR.
        #
        # Never use Whisper's detected_language field.
        # Never use auto-detection.
        # Never default to English.
        # --------------------------------------------------------------
        detected_lang = forced_lang

        if detected_lang not in {"en", "hi"}:
            logger.error(
                "Invalid language state after ASR: %r",
                detected_lang,
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    detail=(
                        "Invalid language state after speech-to-text. "
                        "Analysis was not completed."
                    ),
                    error_code="LANGUAGE_DETECTION_FAILED",
                ).model_dump(),
            )

        # model status
        asr_status = "success" if raw_segments else "failed"

        # Check NLP status — derived from config/ai provider
        nlp_status = "unavailable"
        try:
            from app.services.nlp_service import get_nlp_service as _get_nlp

            _nlp = _get_nlp()
            if _nlp is not None:
                nlp_status = "success"
        except Exception:
            nlp_status = "unavailable"

        # Check acoustic status from emotion results
        acoustic_status = "unavailable"
        has_acoustic = any(
            s.get("emotion_source") in ("acoustic", "fused")
            for s in segments_for_engine
        )
        if has_acoustic:
            acoustic_status = "success"

        fusion_mode = "none"
        text_only = sum(
            1 for s in segments_for_engine
            if s.get("emotion", {}).get("primary")
        )
        if text_only > 0:
            fusion_mode = "text_only"

        # Build transcript segments (legacy shape) from timeline records
        transcript: list[TranscriptSegment] = []
        for i, rec in enumerate(engine_result["timeline"]):
            seg_data = segments_for_engine[i] if i < len(segments_for_engine) else {}
            transcript.append(
                TranscriptSegment(
                    start=rec["timestamp"],
                    end=rec["timestamp"] + rec["duration"],
                    text=seg_data.get("text", ""),
                    speaker=Speaker.CALLER,
                    stress_score=int(round(rec["stress"])),
                    distress_score=int(round(rec["distress"])),
                    emotion=Emotion(
                        seg_data.get("emotion", {}).get("primary", "Uncertainty")
                    )
                    if seg_data.get("emotion", {}).get("primary", "").lower()
                    in [e.value.lower() for e in Emotion]
                    else Emotion.UNCERTAINTY,
                    confidence=rec["confidence"] / 100.0,
                    indicators=seg_data.get("indicators", []),
                    svi_score=int(round(rec["svi"])),
                    risk_level=RiskLevel(rec["risk_level"]),
                    risk_explanation=[],
                    emotion_explanation=seg_data.get("emotion_explanation", []),
                    accent_signals=seg_data.get("accent_signals"),
                )
            )

        # Overall safety score
        overall_safety = engine_result["overall"]["safety_score"]

        # Build risk explanation
        risk_explanation = _build_risk_explanation(
            risk_level=engine_result["overall"]["risk_level"],
            safety_severity=engine_result["overall"]["safety_severity"],
            svi_score=engine_result["overall"]["svi_score"],
            explanations=engine_result.get("explanations", {}),
            indicators=sorted(all_indicators),
            immediate_safety=immediate_safety,
        )

        # Assistive recommendation from risk engine
        recommendation = get_recommendation(
            risk_level=engine_result["overall"]["risk_level"],
            immediate_safety=immediate_safety,
        )

        response = AnalysisResponse(
            case_id=case_id,
            file_name=os.path.basename(file.filename),
            duration_seconds=round(duration, 2) if duration
            else round(len(contents) / 1_000_000 * 60.0, 2),
            transcript=transcript,
            overall_stress_score=int(round(engine_result["overall"]["stress_score"])),
            overall_distress_score=int(round(engine_result["overall"]["distress_score"])),
            overall_svi_score=int(round(engine_result["overall"]["svi_score"])),
            overall_risk_score=int(round(engine_result["overall"]["svi_score"])),
            overall_risk_level=engine_result["overall"]["risk_level"],
            overall_confidence=min(engine_result["overall"]["confidence"] / 100.0, 1.0),
            overall_indicators=sorted(all_indicators),
            risk_explanation=risk_explanation,
            svi_breakdown=SviBreakdown(
                stress_component=int(round(engine_result["overall"]["stress_score"])),
                distress_component=int(round(engine_result["overall"]["distress_score"])),
                safety_component=int(round(overall_safety)),
                context_component=0,
                segment_count=len(engine_result["timeline"]),
                immediate_safety=engine_result["overall"]["safety_immediate_flag"],
            ),
            recommendation=recommendation,
            mode="upload",
            disclaimer=(
                "Assistive risk indicator, not a clinical diagnosis. "
                "High-risk indicators require trained human review."
            ),
            immediate_safety_indicators=engine_result["overall"]["safety_immediate_flag"],
            language=detected_lang,
            analyzed_at=_now_iso(),
            model_status=ModelStatus(
                asr=asr_status,
                text_emotion=nlp_status,
                acoustic_emotion=acoustic_status,
                fusion=fusion_mode,
            ),
            detected_language=detected_lang,
        )

        _case_store[case_id] = response.model_dump()
        cloud_save(case_id, response.model_dump())
        return response

    finally:
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
        raise HTTPException(
            status_code=404, detail=f"Case {case_id} not found."
        )
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
