"""
Speech-to-Text service — routes audio to the correct ASR based on LID result.

Architecture (per the SIH 26093 spec):
  LID identifies en or hi
  → en routes to Whisper (language="en")
  → hi routes to HindiASR (IndicConformer, with Whisper-hi fallback)
  → unsupported/uncertain → STOP (no ASR executed)

The service does NOT perform its own language detection.  It receives
an explicit language code from the LID layer and routes accordingly.

When no LID result is provided and language is None, the service returns
placeholder segments (honest fallback — no fake transcription).
"""
from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from faster_whisper import WhisperModel  # noqa: F401
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False

try:
    from app.services.hindi_asr_service import get_hindi_asr_service
    _HINDI_ASR_AVAILABLE = True
except ImportError:
    _HINDI_ASR_AVAILABLE = False


class SpeechToTextService:
    """ASR router — dispatches to Whisper (en) or HindiASR (hi) based on LID."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = (provider or settings.stt_provider).lower()

    def transcribe(
        self,
        file_path: str,
        language: str | None = None,
        real_upload: bool = False,
    ) -> list[dict[str, Any]]:
        """Transcribe audio using the correct ASR for the given language.

        Args:
            file_path:  path to the audio file (PCM WAV, 16kHz preferred)
            language:   explicit language code from LID: "en", "hi", or None
                        None → no LID available → return placeholder segments
            real_upload:True if this is a real user upload (for logging)

        Returns a list of segment dicts, each with:
            start, end, text, speaker, detected_language

        When language is None (no LID), returns placeholder segments
        with an honest "[Speech-to-text not available]" message.
        """
        if language == "en":
            return self._transcribe_en(file_path, real_upload)
        elif language == "hi":
            return self._transcribe_hi(file_path, real_upload)
        else:
            # None, unsupported, or uncertain → no ASR
            logger.info("No valid LID language (got %s) — returning placeholder segments", language)
            return self._build_placeholder_segments(file_path)

    # ------------------------------------------------------------------
    # English → Whisper
    # ------------------------------------------------------------------

    def _transcribe_en(self, file_path: str, real_upload: bool) -> list[dict[str, Any]]:
        """Transcribe English audio using Faster-Whisper with language="en"."""
        if not _WHISPER_AVAILABLE:
            logger.warning("faster-whisper not available — returning placeholders for English")
            return self._build_placeholder_segments(file_path)

        try:
            model = WhisperModel("base", device="cpu", compute_type="int8")
        except Exception as e:
            logger.warning("Whisper model init failed: %s — placeholders", e)
            return self._build_placeholder_segments(file_path)

        segments_out: list[dict[str, Any]] = []
        try:
            segments_gen, info = model.transcribe(
                file_path,
                language="en",           # EXPLICIT — no auto-detection
                beam_size=5,
                word_timestamps=False,
                vad_filter=False,
                condition_on_previous_text=False,
            )

            for seg in segments_gen:
                text = seg.text.strip()
                if not text:
                    continue
                segments_out.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": text,
                    "detected_language": "en",
                })

            # Merge the segments (adaptive 15-22s merging)
            merged = self._merge_segments(segments_out)

            # Add speaker placeholder
            for segment in merged:
                segment["speaker"] = "caller"

            logger.info("Whisper English transcription: %d segments from %s",
                        len(merged), file_path)
            return merged

        except Exception as e:
            logger.warning("Whisper English transcription failed: %s", e)
            return self._build_placeholder_segments(file_path)

    # ------------------------------------------------------------------
    # Hindi → IndicConformer (with Whisper-hi fallback)
    # ------------------------------------------------------------------

    def _transcribe_hi(self, file_path: str, real_upload: bool) -> list[dict[str, Any]]:
        """Transcribe Hindi audio using IndicConformer, falling back to Whisper-hi."""
        if not _HINDI_ASR_AVAILABLE:
            logger.warning("Hindi ASR service unavailable — returning placeholders for Hindi")
            return self._build_placeholder_segments(file_path)

        try:
            hindi_svc = get_hindi_asr_service()
            segments = hindi_svc.transcribe(file_path)

            if not segments:
                logger.warning("Hindi ASR returned no segments — placeholders")
                return self._build_placeholder_segments(file_path)

            # Ensure detected_language is set
            for seg in segments:
                seg.setdefault("detected_language", "hi")
                seg.setdefault("speaker", "caller")

            logger.info("Hindi ASR transcription: %d segments from %s (provider: %s)",
                        len(segments), file_path, hindi_svc.provider_name())
            return segments

        except Exception as e:
            logger.warning("Hindi ASR failed: %s — placeholders", e)
            return self._build_placeholder_segments(file_path)

    # ------------------------------------------------------------------
    # Placeholder segments (honest fallback)
    # ------------------------------------------------------------------

    def _build_placeholder_segments(self, file_path: str) -> list[dict[str, Any]]:
        """Return honest placeholder segments when ASR is unavailable.

        These segments clearly state that transcription is a placeholder,
        not a real transcript.  The frontend should display them as such.
        """
        # Try to get duration for realistic segment sizing
        duration = 120.0  # default fallback
        try:
            from app.services.audio_service import AudioService
            audio = AudioService()
            meta = audio.inspect(file_path)
            dur = meta.get("duration_seconds", 0.0)
            if dur and dur > 0:
                duration = dur
        except Exception:
            pass

        segment_count = max(1, min(8, int(duration / 30.0) + 1))
        seg_duration = duration / segment_count
        segments: list[dict[str, Any]] = []

        for i in range(segment_count):
            start = round(i * seg_duration, 2)
            end = round(min(start + seg_duration, duration), 2)
            segments.append({
                "start": start,
                "end": end,
                "text": (
                    f"[Segment {i + 1} of {segment_count}] "
                    f"Audio recorded {duration:.0f}s. "
                    f"Speech-to-text unavailable — "
                    f"transcript is a placeholder. "
                    f"ASR provider did not return a transcription."
                ),
                "speaker": "caller",
                "detected_language": "unknown",
            })

        return segments

    # ------------------------------------------------------------------
    # Adaptive segment merging (preserved from previous implementation)
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge short Whisper segments into 15-22s chunks, sentence-aware."""
        if not segments:
            return segments

        MIN_SEG = 15.0
        HARD_CAP = 22.0
        SENTENCE_END = {".", "!", "?", ":", ";"}

        def _is_sentence_end(text: str) -> bool:
            return text.rstrip().endswith(tuple(SENTENCE_END))

        def _is_new_paragraph(text: str) -> bool:
            return "\n\n" in text or "\r\n\r\n" in text

        merged: list[dict[str, Any]] = []
        buf_start = segments[0]["start"]
        buf_end = segments[0]["end"]
        buf_text = segments[0]["text"]

        for seg in segments[1:]:
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_text = seg["text"]

            buf_duration = buf_end - buf_start
            buf_text_end = buf_text.rstrip()

            flush = False

            if _is_sentence_end(buf_text):
                flush = True
            elif _is_new_paragraph(buf_text):
                flush = True
            elif buf_duration >= MIN_SEG:
                flush = True
            elif (seg_end - buf_start) >= HARD_CAP:
                flush = True

            if flush:
                merged.append({
                    "start": round(buf_start, 2),
                    "end": round(buf_end, 2),
                    "text": buf_text_end,
                    "detected_language": segments[0].get("detected_language", "en"),
                })
                buf_start = seg_start
                buf_end = seg_end
                buf_text = seg_text
            else:
                buf_end = seg_end
                buf_text = f"{buf_text_end} {seg_text}".strip()

        # Add final buffer
        if buf_text.strip():
            merged.append({
                "start": round(buf_start, 2),
                "end": round(buf_end, 2),
                "text": buf_text.strip(),
                "detected_language": segments[0].get("detected_language", "en"),
            })

        return merged

    def provider_name(self) -> str:
        return self.provider
