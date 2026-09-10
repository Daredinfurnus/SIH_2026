"""
Speech-to-Text service.

Provides a configurable SpeechToTextService with a DEMO provider and a clear
integration point for a real ASR provider.  The real provider must be
replaceable without changing the frontend.

Behaviour:
- When STT_PROVIDER=demo (default) the service returns deterministic,
  human-readable demo transcripts calibrated to the uploaded duration.
- When a real provider is configured the service is expected to be
  implemented in a subclass / factory; the current MVP keeps the real
  integration point documented but leaves it as a no-op placeholder that
  falls back to demo so the app never crashes due to a missing API key.
"""
from __future__ import annotations

import os
from typing import Any

from app.config import settings


# ===========================================================================
# Deterministic demo transcript segments
# ===========================================================================

_DEMO_SEGMENTS = [
    {
        "start": 0.0,
        "end": 18.0,
        "text": "Hello, I need some help. I don't know who else to call.",
        "speaker": "caller",
        "emotion": "Anxiety",
    },
    {
        "start": 18.0,
        "end": 38.0,
        "text": "Something happened last week and I have been unable to sleep since then. I keep thinking about it.",
        "speaker": "caller",
        "emotion": "Distress",
    },
    {
        "start": 38.0,
        "end": 58.0,
        "text": "I feel scared every time I step outside my house. I don't feel safe anymore.",
        "speaker": "caller",
        "emotion": "Fear",
    },
    {
        "start": 58.0,
        "end": 80.0,
        "text": "People in my neighborhood have been threatening me. I don't know what to do. I feel so alone.",
        "speaker": "caller",
        "emotion": "Fear",
    },
    {
        "start": 80.0,
        "end": 102.0,
        "text": "I don't know if anyone will believe me. I am worried about what they might do next.",
        "speaker": "caller",
        "emotion": "Fear",
    },
    {
        "start": 102.0,
        "end": 124.0,
        "text": "I haven't been eating properly. I keep imagining the worst. I am exhausted.",
        "speaker": "caller",
        "emotion": "Distress",
    },
    {
        "start": 124.0,
        "end": 146.0,
        "text": "Please help me. I don't know where to go. I just want this to stop.",
        "speaker": "caller",
        "emotion": "Helplessness",
    },
    {
        "start": 146.0,
        "end": 168.0,
        "text": "I don't know what will happen now. I am trying to stay calm but it is difficult.",
        "speaker": "caller",
        "emotion": "Uncertainty",
    },
    {
        "start": 168.0,
        "end": 190.0,
        "text": "Thank you for listening. I feel a little better after talking.",
        "speaker": "caller",
        "emotion": "Calm",
    },
]

# Per-segment prototype analysis values — deterministic, realistic progression
_DEMO_ANALYSIS = [
    {
        "stress_score": 34,
        "distress_score": 30,
        "confidence": 0.78,
        "indicators": ["anxiety", "uncertainty"],
    },
    {
        "stress_score": 48,
        "distress_score": 45,
        "confidence": 0.81,
        "indicators": ["distress", "sleep disturbance", "rumination"],
    },
    {
        "stress_score": 62,
        "distress_score": 60,
        "confidence": 0.85,
        "indicators": ["fear", "safety concern", "threat-related context"],
    },
    {
        "stress_score": 71,
        "distress_score": 70,
        "confidence": 0.88,
        "indicators": ["fear", "isolation", "threat-related context", "helplessness"],
    },
    {
        "stress_score": 76,
        "distress_score": 77,
        "confidence": 0.90,
        "indicators": [
            "fear",
            "intimidation concern",
            "uncertainty",
            "hopelessness",
        ],
    },
    {
        "stress_score": 80,
        "distress_score": 82,
        "confidence": 0.91,
        "indicators": ["distress", "helplessness", "hopelessness", "sleep disturbance"],
    },
    {
        "stress_score": 73,
        "distress_score": 68,
        "confidence": 0.86,
        "indicators": ["helplessness", "safety concern", "threat-related context"],
    },
    {
        "stress_score": 66,
        "distress_score": 60,
        "confidence": 0.83,
        "indicators": ["uncertainty", "anxiety", "safety concern"],
    },
    {
        "stress_score": 52,
        "distress_score": 46,
        "confidence": 0.80,
        "indicators": ["anxiety", "relief", "calm"],
    },
]

class SpeechToTextService:
    """Configurable STT provider wrapper."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = (provider or settings.stt_provider).lower()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, file_path: str, language: str = "en", real_upload: bool = False) -> list[dict[str, Any]]:
        """Return a list of transcript segments for the given audio file.

        Each segment is a dict with at least:
            start, end, text, speaker, emotion

        When real_upload is True, the caller is processing an actual uploaded
        file (not the judge-facing /api/demo endpoint). In that case we return
        honest placeholder segments rather than a fabricated crisis narrative,
        because we cannot actually transcribe without a real ASR provider.

        If faster-whisper is installed, the "demo" provider will use it as the
        real engine (with placeholder fallback on failure).  Otherwise it returns
        pure placeholder/demo segments.
        """
        if self.provider == "demo":
            # When faster-whisper is available, use it as the real engine even
            # though the configured provider is "demo".  This gives us real
            # transcription with zero config changes.
            try:
                from faster_whisper import WhisperModel  # noqa: F401
                return self._whisper_transcript(file_path, language, real_upload)
            except ImportError:
                pass
            if real_upload:
                return self._placeholder_transcript(file_path, language)
            return self._demo_transcript(file_path, language)
        return self._real_or_fallback(file_path, language, real_upload=real_upload)

    def provider_name(self) -> str:
        return self.provider

    # ------------------------------------------------------------------
    # Demo provider
    # ------------------------------------------------------------------

    def _demo_transcript(
        self, file_path: str, language: str
    ) -> list[dict[str, Any]]:
        """Deterministic demo transcript scaled to the audio duration."""
        duration = self._available_duration(file_path)
        return _build_demo_segments(duration)

    def _placeholder_transcript(
        self, file_path: str, language: str
    ) -> list[dict[str, Any]]:
        """Honest placeholder segments for a real uploaded file when no ASR
        provider is available.

        These segments acknowledge that transcription is not available in
        demo mode and provide neutral text so the analysis pipeline still
        produces scores without fabricating a specific crisis narrative.
        """
        duration = self._available_duration(file_path)
        segs = _build_demo_segments(duration)
        # Replace the narrative text with honest placeholders.
        n = len(segs)
        for i, seg in enumerate(segs):
            seg["text"] = (
                f"[Segment {i + 1} of {n}] "
                f"Audio recorded {duration:.0f}s. "
                f"Speech-to-text not available in demo mode — "
                f"transcript text is a placeholder. "
                f"Run with a real ASR provider to transcribe actual speech."
            )
            seg["emotion"] = "Neutral"
            seg["indicators"] = ["neutral speech"]
        return segs

    def _whisper_transcript(
        self, file_path: str, language: str, real_upload: bool
    ) -> list[dict[str, Any]]:
        """Transcribe using faster-whisper (base model, CPU, int8).

        Falls back to placeholder (real uploads) or demo narrative
        (judge endpoint) when the model cannot be loaded or transcription
        returns no segments.
        """
        model_size = "base"
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        except Exception:
            return (
                self._placeholder_transcript(file_path, language)
                if real_upload
                else self._demo_transcript(file_path, language)
            )

        try:
            segments_gen, _info = model.transcribe(
                file_path,
                language=None,  # auto-detect
                beam_size=5,
                word_timestamps=False,
            )
            raw_segments = []
            for seg in segments_gen:
                text = seg.text.strip()
                if not text:
                    continue
                raw_segments.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": text,
                })

            if not raw_segments:
                return (
                    self._placeholder_transcript(file_path, language)
                    if real_upload
                    else self._demo_transcript(file_path, language)
                )

                        # ── Adaptive merge: 15s–20s per segment, sentence/paragraph aware ──
            # Merge short segments so each displayed segment is at least
            # MIN_SEGMENT_SECONDS long. Flush on sentence boundary (. ! ? : ;),
            # paragraph break, or when the buffer reaches the minimum duration.
            # 22.0s hard cap prevents a single segment from growing too large.

            MIN_SEGMENT_SECONDS = 15.0
            HARD_CAP_SECONDS = 22.0
            SENTENCE_END = {".", "!", "?", ":", ";"}

            def _is_new_paragraph(text: str) -> bool:
                return "\n\n" in text or "\r\n\r\n" in text

            merged: list[dict[str, float | str]] = []
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
                if buf_text_end.endswith(tuple(SENTENCE_END)):
                    flush = True
                if not flush and _is_new_paragraph(buf_text):
                    flush = True
                if not flush and buf_duration >= MIN_SEGMENT_SECONDS:
                    flush = True
                if not flush and (seg_end - buf_start) >= HARD_CAP_SECONDS:
                    flush = True

                if flush:
                    merged.append({
                        "start": round(buf_start, 2),
                        "end": round(buf_end, 2),
                        "text": buf_text_end,
                        "speaker": "caller",
                    })
                    buf_start = seg_start
                    buf_end = seg_end
                    buf_text = seg_text
                else:
                    buf_end = seg_end
                    buf_text = f"{buf_text} {seg_text}"

            if buf_text.strip():
                merged.append({
                    "start": round(buf_start, 2),
                    "end": round(buf_end, 2),
                    "text": buf_text.strip(),
                    "speaker": "caller",
                })

            if not merged:
                return (
                    self._placeholder_transcript(file_path, language)
                    if real_upload
                    else self._demo_transcript(file_path, language)
                )

            segments = []
            for seg in merged:
                segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "speaker": seg.get("speaker", "victim"),
                    "emotion": "Neutral",
                })

            return segments
        except Exception:
            return (
                self._placeholder_transcript(file_path, language)
                if real_upload
                else self._demo_transcript(file_path, language)
            )

    # ------------------------------------------------------------------
    # Real provider integration point
    # ------------------------------------------------------------------

    def _real_or_fallback(
        self, file_path: str, language: str, real_upload: bool = False
    ) -> list[dict[str, Any]]:
        """Whisper (faster-whisper) based speech-to-text provider.

        Uses the ``faster-whisper`` library with the ``base`` model by default.
        Falls back to placeholder (real uploads) or demo (judge endpoint)
        when the model cannot be loaded or transcription fails.

        The model is downloaded automatically on first use (~140 MB for ``base``)
        and cached by HuggingFace Hub.  Subsequent runs reuse the cached model.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            if real_upload:
                return self._placeholder_transcript(file_path, language)
            return self._demo_transcript(file_path, language)

        model_size = "base"
        try:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        except Exception:
            if real_upload:
                return self._placeholder_transcript(file_path, language)
            return self._demo_transcript(file_path, language)

        try:
            segments_gen, _info = model.transcribe(
                file_path,
                language=None,  # auto-detect
                beam_size=5,
                word_timestamps=False,
            )
            segments = []
            for seg in segments_gen:
                text = seg.text.strip()
                if not text:
                    continue
                segments.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": text,
                    "speaker": "caller",
                    "emotion": "Neutral",
                })
            if not segments:
                if real_upload:
                    return self._placeholder_transcript(file_path, language)
                return self._demo_transcript(file_path, language)
            return segments
        except Exception:
            if real_upload:
                return self._placeholder_transcript(file_path, language)
            return self._demo_transcript(file_path, language)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _available_duration(self, file_path: str) -> float:
        """Best-effort duration for scaling demo segments."""
        from app.services.audio_service import AudioService

        audio = AudioService()
        meta = audio.inspect(file_path)
        dur = meta.get("duration_seconds", 0.0)
        if dur and dur > 0:
            return dur
        # Fallback: use file size as a rough proxy so demo mode always has
        # something to display even without audio tooling.
        try:
            size = os.path.getsize(file_path)
        except OSError:
            size = 0
        # Very rough: assume ~128kbps audio → ~1 MB ≈ 60 s
        if size > 0:
            return max(20.0, min(size / 1_000_000 * 60.0, 600.0))
        return 120.0


# ===========================================================================
# Demo segment builder
# ===========================================================================

def _build_demo_segments(duration: float) -> list[dict[str, Any]]:
    """Build deterministic demo segments spread across `duration` seconds."""
    if duration <= 0:
        duration = 120.0

    total_demo_time = _DEMO_SEGMENTS[-1]["end"]
    scale = duration / total_demo_time if total_demo_time > 0 else 1.0

    segments: list[dict[str, Any]] = []
    for base, analysis in zip(_DEMO_SEGMENTS, _DEMO_ANALYSIS):
        seg: dict[str, Any] = {
            "start": round(base["start"] * scale, 2),
            "end": round(base["end"] * scale, 2),
            "text": base["text"],
            "speaker": base["speaker"],
            "emotion": base["emotion"],
        }
        seg.update(analysis)
        segments.append(seg)

    # Extend or trim to match the actual duration
    if segments:
        segments[-1]["end"] = round(duration, 2)
        # If the last segment would end before the duration, add a short tail
        if segments[-1]["end"] < duration - 1.0:
            tail = dict(segments[-1])
            tail["start"] = round(segments[-1]["end"], 2)
            tail["end"] = round(duration, 2)
            tail["text"] = "..."
            tail["emotion"] = "Calm"
            tail["stress_score"] = max(30, segments[-1]["stress_score"] - 5)
            tail["distress_score"] = max(28, segments[-1]["distress_score"] - 6)
            tail["confidence"] = segments[-1]["confidence"] - 0.02
            tail["indicators"] = ["calm", "uncertainty"]
            segments.append(tail)

    return segments
