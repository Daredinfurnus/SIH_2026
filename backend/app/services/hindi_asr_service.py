"""
Hindi ASR service using AI4Bharat IndicConformer.

Provides Hindi-specific speech-to-text transcription.
When the model is unavailable (gated HuggingFace repo, missing
transformers, etc.), falls back to faster-whisper with explicit
language="hi".
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HindiASRService:
    """Hindi ASR: IndicConformer primary, Whisper-hi fallback."""

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._fallback_available: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe Hindi audio.

        Returns a dict with:
            transcript: str — Hindi text in Devanagari script
            confidence: float — 0.0–1.0 estimated confidence
            method: str — "indicconformer" | "whisper_hi" | "unavailable"
            segments: list[dict] — per-segment info (optional)
        """
        # Try IndicConformer first
        result = self._try_indicconformer(audio_path)
        if result["method"] != "unavailable":
            return result

        # Fall back to Whisper-hi
        logger.info("IndicConformer unavailable; falling back to Whisper-hi")
        return self._transcribe_whisper_hi(audio_path)

    # ------------------------------------------------------------------
    # IndicConformer path
    # ------------------------------------------------------------------

    def _try_indicconformer(self, audio_path: str) -> Dict[str, Any]:
        """Try IndicConformer.  Returns unavailable dict on any failure."""
        if self._model is None:
            if not self._load_indicconformer():
                return self._unavailable("indicconformer_load_failed")

        try:
            return self._transcribe_indicconformer(audio_path)
        except Exception as exc:
            logger.warning("IndicConformer transcription failed: %s", exc)
            return self._unavailable("indicconformer_inference_failed")

    def _load_indicconformer(self) -> bool:
        """Load IndicConformer model and processor.

        Returns True if loading succeeded.
        """
        try:
            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
            )
        except ImportError:
            logger.error("transformers not installed; cannot load IndicConformer")
            return False

        MODEL_ID = "ai4bharat/indic-conformer-v3"

        logger.info("Loading IndicConformer v3 from %s", MODEL_ID)
        try:
            self._processor = AutoProcessor.from_pretrained(MODEL_ID)
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID)
            self._model.eval()
            logger.info("IndicConformer v3 loaded successfully")
            return True
        except Exception as exc:
            # This is where gated repo 401s surface
            logger.warning("IndicConformer load failed: %s", exc)
            self._model = None
            self._processor = None
            return False

    def _transcribe_indicconformer(self, audio_path: str) -> Dict[str, Any]:
        """Run inference with IndicConformer."""
        import torch
        import numpy as np
        import soundfile as sf

        # Read audio → 16kHz mono
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        # IndicConformer expects input in a specific format.
        # Process via the processor.
        inputs = self._processor(
            audio, sampling_rate=16000, return_tensors="pt"
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)

        with torch.no_grad():
            outputs = self._model.generate(
                inputs.input_features.to(device)
                if hasattr(inputs, "input_features")
                else inputs["input_features"].to(device),
                max_new_tokens=1024,
            )

        # Decode token IDs → text
        transcript = self._processor.batch_decode(outputs, skip_special_tokens=True)
        text = transcript[0] if transcript else ""

        # IndicConformer returns Hindi text in Devanagari
        confidence = 0.85  # indicative — model is specialized for Hindi

        logger.info(
            "IndicConformer: transcript=%s chars=%d",
            text[:80], len(text),
        )

        return {
            "transcript": text,
            "confidence": confidence,
            "method": "indicconformer",
            "language": "hi",
        }

    # ------------------------------------------------------------------
    # Whisper-hi fallback
    # ------------------------------------------------------------------

    def _transcribe_whisper_hi(self, audio_path: str) -> Dict[str, Any]:
        """Faster-Whisper with explicit language="hi"."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.error("faster-whisper not installed; Hindi ASR fully unavailable")
            return self._unavailable("whisper_not_installed")

        model = WhisperModel("tiny", device="cpu", compute_type="int8")

        segments, info = model.transcribe(
            audio_path,
            language="hi",  # EXPLICIT — forced Hindi
            beam_size=5,
            word_timestamps=False,
        )

        transcript = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        confidence = info.language_probability if info.language == "hi" else 0.5

        logger.info(
            "Whisper-hi: transcript=%s chars=%d lang=%s prob=%.3f",
            transcript[:80], len(transcript), info.language, confidence,
        )

        return {
            "transcript": transcript,
            "confidence": float(confidence),
            "method": "whisper_hi",
            "language": "hi",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unavailable(self, reason: str) -> Dict[str, Any]:
        return {
            "transcript": "",
            "confidence": 0.0,
            "method": "unavailable",
            "language": "hi",
            "_error": reason,
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_hindi_svc: Optional[HindiASRService] = None


def get_hindi_asr_service() -> HindiASRService:
    """Get or create the shared Hindi ASR service instance."""
    global _hindi_svc
    if _hindi_svc is None:
        _hindi_svc = HindiASRService()
    return _hindi_svc


def _TRANSFORMERS_AVAILABLE() -> bool:
    """Check whether transformers is importable (needed for IndicConformer)."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False
