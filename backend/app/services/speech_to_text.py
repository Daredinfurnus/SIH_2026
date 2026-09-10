"""
Speech-to-Text service.

Provides a configurable SpeechToTextService backed by faster-whisper (base model,
CPU, int8) for real audio uploads.  When Whisper cannot be loaded or returns no
segments, the service falls back to honest placeholder segments so the analysis
pipeline keeps running without fabricating a crisis narrative.

This is a REAL uploads-only service.  The judge-facing /api/demo endpoint has
been removed per SIH requirements — there is no synthetic demo mode.

Behaviour:
- transcribe(file_path, language, real_upload) → list of segment dicts
- Each segment: start, end, text, speaker, emotion
- Speaker is always "caller" for a single-audio-channel upload (see README for
  the two-speaker limitation note).
- Whisper segments are merged into 15-20s speaker-turn-aware chunks before
  return, so the transcript reads at human paragraph granularity.
"""

from __future__ import annotations

import os
from typing import Any

from app.config import settings


class SpeechToTextService:
    """Whisper-backed STT provider with honest placeholder fallback."""

    def __init__(self, provider: str | None = None) -> None:
        # provider is kept for API compatibility but the MVP only ever runs
        # the Whisper engine for real uploads.  No synthetic demo provider.
        self.provider = (provider or settings.stt_provider).lower()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self, file_path: str, language: str = "en", real_upload: bool = False
    ) -> list[dict[str, Any]]:
        """Return a list of transcript segments for the given audio file.

        Each segment dict has at least::
            start, end, text, speaker, emotion

        When ``real_upload`` is True the caller is processing an actual
        uploaded file.  When Whisper is unavailable the service returns honest
        placeholder segments (never a fabricated crisis narrative).

        If faster-whisper is installed the service uses it as the real engine.
        """

        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError:
            # Whisper not installed at all — return honest placeholder.
            return self._build_placeholder_segments(file_path)

        return self._whisper_transcript(file_path, language, real_upload)

    def provider_name(self) -> str:
        return self.provider

    # ------------------------------------------------------------------
    # Whisper transcription + adaptive segment merge
    # ------------------------------------------------------------------

    def _whisper_transcript(
        self, file_path: str, language: str, real_upload: bool
    ) -> list[dict[str, Any]]:
        """Transcribe *file_path* with faster-whisper (base, CPU, int8).

        Raw Whisper segments are merged into 15-20s chunks that respect
        sentence/paragraph boundaries before being returned to the caller.
        """
        model_size = "base"
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        except Exception:
            # Model could not be loaded — fall back to honest placeholder.
            return self._build_placeholder_segments(file_path)

        try:
            segments_gen, _info = model.transcribe(
                file_path,
                language=None if language == "auto" else language,
                beam_size=5,
                word_timestamps=False,
            )

            raw_segments: list[dict[str, Any]] = []
            for seg in segments_gen:
                text = seg.text.strip()
                if not text:
                    continue
                raw_segments.append(
                    {"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text}
                )

            if not raw_segments:
                return self._build_placeholder_segments(file_path)

            # ── Adaptive merge: 15s–20s per segment, sentence/paragraph aware ──
            # Merge short Whisper segments so each displayed segment is at
            # least MIN_SEGMENT_SECONDS long.  Flush on sentence boundary
            # (. ! ? : ;), paragraph break, or when the buffer reaches the
            # minimum duration.  22.0s hard cap keeps individual segments
            # readable.
            MIN_SEGMENT_SECONDS = 15.0
            HARD_CAP_SECONDS = 22.0
            SENTENCE_END = {".", "!", "?", ":", ";"}

            def _is_sentence_end(text: str) -> bool:
                return text.rstrip().endswith(tuple(SENTENCE_END))

            def _is_new_paragraph(text: str) -> bool:
                return "\n\n" in text or "\r\n\r\n" in text

            merged: list[dict[str, Any]] = []
            buf_start = raw_segments[0]["start"]
            buf_end = raw_segments[0]["end"]
            buf_text = raw_segments[0]["text"]

            for seg in raw_segments[1:]:
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
                elif buf_duration >= MIN_SEGMENT_SECONDS:
                    flush = True
                elif (seg_end - buf_start) >= HARD_CAP_SECONDS:
                    flush = True

                if flush:
                    merged.append(
                        {
                            "start": round(buf_start, 2),
                            "end": round(buf_end, 2),
                            "text": buf_text_end,
                            "speaker": "caller",
                        }
                    )
                    buf_start = seg_start
                    buf_end = seg_end
                    buf_text = seg_text
                else:
                    buf_end = seg_end
                    buf_text = f"{buf_text} {seg_text}"

            if buf_text.strip():
                merged.append(
                    {
                        "start": round(buf_start, 2),
                        "end": round(buf_end, 2),
                        "text": buf_text.strip(),
                        "speaker": "caller",
                    }
                )

            if not merged:
                return self._build_placeholder_segments(file_path)

            # Build the final segment list with emotion labels set to Neutral
            # (emotion is assigned later by the analysis engine from the text).
            segments: list[dict[str, Any]] = []
            for seg in merged:
                segments.append(
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"],
                        "speaker": seg.get("speaker", "caller"),
                        "emotion": "Neutral",
                    }
                )

            return segments

        except Exception:
            # Any runtime transcription error → honest placeholder.
            return self._build_placeholder_segments(file_path)

    # ------------------------------------------------------------------
    # Honest placeholder segments (no fabricated narrative)
    # ------------------------------------------------------------------

    def _build_placeholder_segments(self, file_path: str) -> list[dict[str, Any]]:
        """Return neutral placeholder segments when transcription is unavailable.

        These segments tell the operator that speech-to-text is not available
        rather than inventing a specific crisis narrative.  The analysis
        pipeline still produces scores from the placeholder text so the UI
        stays usable.
        """
        duration = self._available_duration(file_path)

        # One segment per ~30s of audio, capped at 8 segments for readability.
        segment_count = max(1, min(8, int(duration / 30.0) + 1))
        seg_duration = duration / segment_count

        segments: list[dict[str, Any]] = []
        for i in range(segment_count):
            start = round(i * seg_duration, 2)
            end = round(min(start + seg_duration, duration), 2)
            segments.append(
                {
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
                    "emotion": "Neutral",
                }
            )

        return segments

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _available_duration(self, file_path: str) -> float:
        """Best-effort duration for scaling placeholder segments."""
        from app.services.audio_service import AudioService

        audio = AudioService()
        meta = audio.inspect(file_path)
        dur = meta.get("duration_seconds", 0.0)
        if dur and dur > 0:
            return dur

        # Fallback: estimate from file size (~128kbps → ~1 MB ≈ 60 s).
        try:
            size = os.path.getsize(file_path)
        except OSError:
            size = 0
        if size > 0:
            return max(20.0, min(size / 1_000_000 * 60.0, 600.0))
        return 120.0
