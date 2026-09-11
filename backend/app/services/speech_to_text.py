"""
Speech-to-Text service — faster-whisper (base, CPU, int8) for real uploads.
Upload-based pipeline with honest placeholder fallback when ASR is unavailable.
"""

from __future__ import annotations

import os
from typing import Any

from app.config import settings


class SpeechToTextService:
    """Whisper-backed STT provider with honest placeholder fallback."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = (provider or settings.stt_provider).lower()

    def transcribe(
        self, file_path: str, language: str = "en", real_upload: bool = False
    ) -> list[dict[str, Any]]:
        """Return transcript segments. language=None → auto-detect."""
        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError:
            return self._build_placeholder_segments(file_path)
        return self._whisper_transcript(file_path, language, real_upload)

    def provider_name(self) -> str:
        return self.provider

    # ------------------------------------------------------------------
    # Whisper transcription + adaptive merge
    # ------------------------------------------------------------------

    def _whisper_transcript(
        self, file_path: str, language: str | None, real_upload: bool
    ) -> list[dict[str, Any]]:
        """Whisper transcription with English/Hindi-only language selection.

        Strategy:
        1. Auto-detect the language.
        2. If Whisper detects English or Hindi, use that result.
        3. If Whisper detects another language (e.g. Urdu), run both
        English and Hindi and select the stronger decoding using avg_logprob.
        """

        model_size = "base"

        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
            )
        except Exception:
            return self._build_placeholder_segments(file_path)

        # --------------------------------------------------------------
        # Helper: run Whisper transcription
        # --------------------------------------------------------------
        def _run_transcription(
            forced_language: str | None,
        ) -> tuple[list[dict[str, Any]], float, str | None, float]:

            segments_out: list[dict[str, Any]] = []
            logprobs: list[float] = []
            detected_language: str | None = None
            language_probability = 0.0

            try:
                segments_gen, info = model.transcribe(
                    file_path,
                    language=forced_language,
                    beam_size=5,
                    word_timestamps=False,
                    vad_filter=False,
                    condition_on_previous_text=False,
                )

                detected_language = getattr(info, "language", None)
                language_probability = (
                    getattr(info, "language_probability", 0.0) or 0.0
                )

                for seg in segments_gen:
                    text = seg.text.strip()

                    if not text:
                        continue

                    avg_logprob = getattr(seg, "avg_logprob", -10.0)

                    segments_out.append(
                        {
                            "start": round(seg.start, 2),
                            "end": round(seg.end, 2),
                            "text": text,
                            "_avg_logprob": avg_logprob,
                        }
                    )

                    logprobs.append(avg_logprob)

            except Exception:
                return [], -10.0, None, 0.0

            if logprobs:
                mean_logprob = sum(logprobs) / len(logprobs)
            else:
                mean_logprob = -10.0

            return (
                segments_out,
                mean_logprob,
                detected_language,
                language_probability,
            )

        # --------------------------------------------------------------
        # Pass 1: automatic language detection
        # --------------------------------------------------------------
        auto_segments, auto_logprob, detected_lang, detected_lang_prob = (
            _run_transcription(None)
        )

        # --------------------------------------------------------------
        # Decide which transcript to use
        # --------------------------------------------------------------
        if detected_lang in {"en", "hi"}:
            # Whisper detected one of our supported languages.
            best_segments = auto_segments

        else:
            # Whisper detected something outside our supported languages.
            # Example: English audio incorrectly detected as Urdu.
            #
            # Only compare the two languages supported by our prototype.
            en_segments, en_logprob, _, _ = _run_transcription("en")
            hi_segments, hi_logprob, _, _ = _run_transcription("hi")

            if not en_segments and not hi_segments:
                best_segments = auto_segments

            elif not hi_segments:
                best_segments = en_segments
                detected_lang = "en"

            elif not en_segments:
                best_segments = hi_segments
                detected_lang = "hi"

            elif en_logprob >= hi_logprob:
                best_segments = en_segments
                detected_lang = "en"

            else:
                best_segments = hi_segments
                detected_lang = "hi"

        # --------------------------------------------------------------
        # Fallback
        # --------------------------------------------------------------
        if not best_segments:
            return self._build_placeholder_segments(file_path)

        # Remove internal scoring information.
        for seg in best_segments:
            seg.pop("_avg_logprob", None)

        # Safety fallback.
        if detected_lang not in {"en", "hi"}:
            detected_lang = "en"

        # --------------------------------------------------------------
        # Adaptive merge: 15–20s segments, sentence/paragraph aware
        # --------------------------------------------------------------
        MIN_SEG = 15.0
        HARD_CAP = 22.0
        SENTENCE_END = {".", "!", "?", ":", ";"}

        def _is_sentence_end(text: str) -> bool:
            return text.rstrip().endswith(tuple(SENTENCE_END))

        def _is_new_paragraph(text: str) -> bool:
            return "\n\n" in text or "\r\n\r\n" in text

        merged: list[dict[str, Any]] = []

        buf_start = best_segments[0]["start"]
        buf_end = best_segments[0]["end"]
        buf_text = best_segments[0]["text"]

        for seg in best_segments[1:]:
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
                merged.append(
                    {
                        "start": round(buf_start, 2),
                        "end": round(buf_end, 2),
                        "text": buf_text_end,
                    }
                )

                buf_start = seg_start
                buf_end = seg_end
                buf_text = seg_text

            else:
                buf_end = seg_end
                buf_text = f"{buf_text_end} {seg_text}".strip()

        # Add final buffer.
        if buf_text.strip():
            merged.append(
                {
                    "start": round(buf_start, 2),
                    "end": round(buf_end, 2),
                    "text": buf_text.strip(),
                }
            )

        # --------------------------------------------------------------
        # Speaker placeholder
        # --------------------------------------------------------------
        for segment in merged:
            segment["speaker"] = "caller"

        return merged

    # ------------------------------------------------------------------
    # Placeholder segments
    # ------------------------------------------------------------------

    def _build_placeholder_segments(self, file_path: str) -> list[dict[str, Any]]:
        """Honest placeholders when transcription is unavailable."""
        duration = self._available_duration(file_path)
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
                    f"Speech-to-text not available — "
                    f"transcript text is a placeholder. "
                    f"Run with a real ASR provider to transcribe actual speech."
                ),
                "speaker": "caller",
            })
        return segments

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _available_duration(self, file_path: str) -> float:
        from app.services.audio_service import AudioService
        audio = AudioService()
        meta = audio.inspect(file_path)
        dur = meta.get("duration_seconds", 0.0)
        if dur and dur > 0:
            return dur
        try:
            size = os.path.getsize(file_path)
        except OSError:
            size = 0
        if size > 0:
            return max(20.0, min(size / 1_000_000 * 60.0, 600.0))
        return 120.0
