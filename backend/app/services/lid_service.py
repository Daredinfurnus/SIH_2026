"""
Language Identification service for TraumaSense.

Restricted to the application's two supported languages:
  English  → "en"
  Hindi    → "hi"

All other results (including Urdu, Punjabi, Bengali, etc.) are
rejected as unsupported/uncertain.  The application may internally use
a pretrained spoken-language identification model (e.g. VoxLingua107),
but the application's output space is strictly {"en", "hi"}.

The LID layer is the SINGLE AUTHORITY for language routing.
ASR models (Whisper, IndicConformer) do NOT perform their own
language selection — they receive an explicit language from LID.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {"en", "hi"}
LID_THRESHOLD = 0.50  # minimum confidence to accept a language decision


def supported_languages() -> frozenset[str]:
    """Return the frozenset of supported application language codes."""
    return frozenset(SUPPORTED_LANGUAGES)


def identify_language(
    audio_path: str,
    *,
    model_id: str = "voxlingua107",
) -> Dict[str, Any]:
    """Identify the spoken language in `audio_path`.

    Parameters
    ----------
    audio_path : str
        Path to the audio file to analyze.
    model_id : str
        Which LID backend to use:
          - "voxlingua107"  — SpeechBrain VoxLingua107 (gated; may fail with 401)
          - "whisper"       — Faster-Whisper LID (always available with STT)

    Returns
    -------
    dict with:
        language       : str or None — "en", "hi", or None (unsupported)
        confidence     : float  — 0.0–1.0
        supported      : bool   — True if language in SUPPORTED_LANGUAGES
        method         : str    — "voxlingua107" | "whisper" | "none"
        raw_scores     : dict   — top language scores (debugging only)
    """
    if model_id == "voxlingua107":
        return _identify_voxlingua107(audio_path)
    if model_id == "whisper":
        return _identify_whisper_lid(audio_path)
    logger.warning("Unknown LID model_id: %s; falling back to whisper", model_id)
    return _identify_whisper_lid(audio_path)


# ---------------------------------------------------------------------------
# VoxLingua107 backend
# ---------------------------------------------------------------------------

def _identify_voxlingua107(audio_path: str) -> Dict[str, Any]:
    """Try SpeechBrain VoxLingua107.  Falls back to Whisper LID on any failure."""
    try:
        return _voxlingua107_inner(audio_path)
    except Exception as exc:
        logger.warning(
            "VoxLingua107 LID failed (%s); falling back to Whisper LID", exc
        )
        return _identify_whisper_lid(audio_path)


def _voxlingua107_inner(audio_path: str) -> Dict[str, Any]:
    import torch
    import speechbrain as sb
    from speechbrain.pretrained import LanguageIdentifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading VoxLingua107 language identifier on %s", device)

    # NOTE: speechbrain/lang_id_voxlingua107 is a gated repo.
    # from_hparams may succeed (config download) but the weights download
    # can fail with 401 Unauthorized when the model actually runs.
    # The failure will surface as an exception here — we catch it and
    # fall back to Whisper LID in the caller.
    identifier = LanguageIdentifier.from_hparams(
        source="speechbrain/lang_id_voxlingua107",
        run_opts={"device": device},
    )

    scores = identifier.get_language_scores(audio_path)
    # `scores` is an OrderedDict of lang_code → score
    en_score = float(scores.get("en", 0.0))
    hi_score = float(scores.get("hi", 0.0))

    logger.info(
        "VoxLingua107 raw scores — en=%.3f hi=%.3f (top: %s)",
        en_score, hi_score,
        list(scores.items())[:5],
    )

    return _make_lid_decision(en_score, hi_score, "voxlingua107", scores)


# ---------------------------------------------------------------------------
# Whisper LID backend (fallback)
# ---------------------------------------------------------------------------

def _identify_whisper_lid(audio_path: str) -> Dict[str, Any]:
    """Use Faster-Whisper as a spoken-language identification backend.

    Faster-Whisper can detect the language of an audio segment.  We use it
    strictly as an LID — we do NOT run it as ASR here (that happens later,
    routed by the LID decision).
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.error("Faster-Whisper not installed; cannot use as LID fallback")
        return _fallback_unsupported("whisper_unavailable")

    import numpy as np
    import soundfile as sf

    # Load tiny model for fast LID
    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    # Read audio
    try:
        audio, sr = sf.read(audio_path)
    except Exception:
        import wave
        with wave.open(audio_path, "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            audio = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Take first 10 seconds for LID
    target_len = int(16000 * 10)
    segment = audio[:target_len] if len(audio) > target_len else audio

    if len(segment) < 16000 * 1:  # less than 1 second
        return _fallback_unsupported("whisper_audio_too_short")

    # Resample if needed
    if sr != 16000:
        import librosa
        segment = librosa.resample(segment, orig_sr=sr, target_sr=16000)

    # Run Whisper in language-detection mode
    segments_gen, info = model.transcribe(
        segment,
        language=None,  # auto-detect
        beam_size=1,
        vad_filter=False,
        without_timestamps=True,
    )

    detected_lang = info.language
    lang_prob = info.language_probability

    en_score = lang_prob if detected_lang == "en" else 0.0
    hi_score = lang_prob if detected_lang == "hi" else 0.0

    raw_scores = {"en": en_score, "hi": hi_score, detected_lang: lang_prob}

    logger.info(
        "Whisper LID: detected=%s probability=%.3f en=%.3f hi=%.3f",
        detected_lang, lang_prob, en_score, hi_score,
    )

    return _make_lid_decision(en_score, hi_score, "whisper", dict(raw_scores))


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def _make_lid_decision(
    en_score: float,
    hi_score: float,
    method: str,
    raw_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Apply the decision rule: only en vs hi, with threshold.

    Only english_score and hindi_score are examined.  If the best score is
    below threshold, or en == hi (tied), the result is unsupported/uncertain.
    Everything else (Urdu, Punjabi, Bengali, etc.) is rejected.
    """
    if raw_scores is None:
        raw_scores = {"en": en_score, "hi": hi_score}

    best_score = max(en_score, hi_score)
    best_lang = "en" if en_score >= hi_score else "hi"

    if best_score < LID_THRESHOLD or en_score == hi_score:
        logger.info(
            "LID (method=%s): best_score=%.3f below threshold=%.2f "
            "or tied → unsupported",
            method, best_score, LID_THRESHOLD,
        )
        return {
            "language": None,
            "confidence": 0.0,
            "supported": False,
            "method": method,
            "raw_scores": dict(raw_scores),
        }

    language = best_lang
    confidence = float(best_score)

    logger.info(
        "LID (method=%s): language=%s confidence=%.3f → supported",
        method, language, confidence,
    )

    return {
        "language": language,
        "confidence": confidence,
        "supported": True,
        "method": method,
        "raw_scores": dict(raw_scores),
    }


def _fallback_unsupported(reason: str) -> Dict[str, Any]:
    """Return an unsupported result (e.g. when LID backend is unavailable)."""
    logger.warning("LID fallback unsupported: %s", reason)
    return {
        "language": None,
        "confidence": 0.0,
        "supported": False,
        "method": "none",
        "raw_scores": {},
        "_fallback_reason": reason,
    }
