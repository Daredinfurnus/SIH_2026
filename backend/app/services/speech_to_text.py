"""
Speech-to-Text service for TraumaSense.

Authoritative language routing is performed by VoxLingua107 BEFORE ASR.

Supported routes:
    en -> Faster-Whisper, explicitly forced to English
    hi -> AI4Bharat IndicConformer, explicitly forced to Hindi

Design guarantees:
    - No Whisper language auto-detection.
    - No English-vs-Hindi decoding comparison.
    - No Urdu fallback.
    - No guessed language.
    - No placeholder/fake transcripts.
    - No silent ASR fallback.
    - Invalid/unsupported languages fail closed.
    - Models are cached process-wide to avoid reloading per request.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = frozenset({"en", "hi"})

WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

INDICCONFORMER_MODEL_ID = (
    "ai4bharat/indic-conformer-600m-multilingual"
)

TARGET_SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Process-wide model cache
# ---------------------------------------------------------------------------
#
# FastAPI creates a SpeechToTextService per request in the current pipeline.
# Therefore model caching must not depend on the service instance.
#
# These models are loaded once per Python process and reused.
# ---------------------------------------------------------------------------

_whisper_model: Any | None = None
_indicconformer_model: Any | None = None

_whisper_lock = threading.Lock()
_indicconformer_lock = threading.Lock()


class SpeechToTextService:
    """
    Dual-ASR service.

    Language routing MUST already have been performed by the LID layer.
    This class does not decide which language is being spoken.
    """

    def __init__(self, provider: str | None = None) -> None:
        # Kept for compatibility with the existing application configuration.
        self.provider = (provider or settings.stt_provider).lower()

    def provider_name(self) -> str:
        return self.provider

    # ======================================================================
    # Public API
    # ======================================================================

    def transcribe(
        self,
        file_path: str,
        language: str,
    ) -> list[dict[str, Any]]:
        """
        Transcribe audio using the language already selected by LID.

        Args:
            file_path: Path to normalized audio.
            language: MUST be "en" or "hi".

        Raises:
            ValueError:
                Unsupported language.

            RuntimeError:
                Model loading, audio preprocessing, or inference failure.
        """

        if not file_path:
            raise ValueError("Audio file path is empty.")

        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported ASR language '{language}'. "
                "Expected one of: en, hi."
            )

        if language == "en":
            return self._transcribe_english(file_path)

        if language == "hi":
            return self._transcribe_hindi(file_path)

        # Defensive fail-closed guard.
        raise ValueError(
            f"Unsupported ASR language '{language}'."
        )

    # ======================================================================
    # English / Faster-Whisper
    # ======================================================================

    @staticmethod
    def _load_whisper() -> Any:
        """
        Load Faster-Whisper once per process.

        Explicitly uses CPU/int8 for the current deployment.
        """

        global _whisper_model

        if _whisper_model is not None:
            return _whisper_model

        with _whisper_lock:
            if _whisper_model is not None:
                return _whisper_model

            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "Faster-Whisper is not installed."
                ) from exc

            try:
                logger.info(
                    "Loading Faster-Whisper model=%s device=%s "
                    "compute_type=%s",
                    WHISPER_MODEL_SIZE,
                    WHISPER_DEVICE,
                    WHISPER_COMPUTE_TYPE,
                )

                _whisper_model = WhisperModel(
                    WHISPER_MODEL_SIZE,
                    device=WHISPER_DEVICE,
                    compute_type=WHISPER_COMPUTE_TYPE,
                )

            except Exception as exc:
                logger.exception(
                    "Faster-Whisper model loading failed."
                )

                raise RuntimeError(
                    f"Faster-Whisper model load failed: {exc}"
                ) from exc

            logger.info("Faster-Whisper loaded successfully.")

            return _whisper_model

    def _transcribe_english(
        self,
        file_path: str,
    ) -> list[dict[str, Any]]:
        """
        Transcribe English.

        IMPORTANT:
        language="en" is always supplied explicitly.
        Whisper is never allowed to choose the language.
        """

        model = self._load_whisper()

        try:
            segments_gen, info = model.transcribe(
                file_path,

                # ------------------------------------------------------
                # AUTHORITATIVE LANGUAGE
                # ------------------------------------------------------
                language="en",

                # Accuracy / stability.
                beam_size=5,
                best_of=5,
                temperature=0.0,

                # Prevent Whisper from carrying hallucinated context
                # between unrelated audio regions.
                condition_on_previous_text=False,

                # Let Whisper skip obvious non-speech.
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 500,
                },

                # We don't need word-level timestamps currently.
                word_timestamps=False,

                # Do not prepend/alter the transcript with special text.
                without_timestamps=False,
            )

            segments: list[dict[str, Any]] = []

            for segment in segments_gen:
                text = (segment.text or "").strip()

                if not text:
                    continue

                start = float(segment.start)
                end = float(segment.end)

                if not np.isfinite(start) or not np.isfinite(end):
                    logger.warning(
                        "Ignoring English ASR segment with invalid "
                        "timestamps: start=%r end=%r",
                        start,
                        end,
                    )
                    continue

                if end <= start:
                    logger.warning(
                        "Ignoring zero/negative English ASR segment: "
                        "start=%s end=%s",
                        start,
                        end,
                    )
                    continue

                segments.append(
                    {
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "text": text,
                        "speaker": "caller",
                        "detected_language": "en",
                    }
                )

        except Exception as exc:
            logger.exception(
                "English ASR inference failed."
            )

            raise RuntimeError(
                f"English ASR inference failed: {exc}"
            ) from exc

        if not segments:
            raise RuntimeError(
                "English ASR completed but produced an empty transcript."
            )

        return self._merge_whisper_segments(segments)

    # ======================================================================
    # Hindi / IndicConformer
    # ======================================================================

    @staticmethod
    def _load_indicconformer() -> Any:
        """
        Load AI4Bharat IndicConformer once per process.

        Requires authenticated Hugging Face access to the gated model.
        """

        global _indicconformer_model

        if _indicconformer_model is not None:
            return _indicconformer_model

        with _indicconformer_lock:
            if _indicconformer_model is not None:
                return _indicconformer_model

            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise RuntimeError(
                    "Transformers is not installed."
                ) from exc

            try:
                logger.info(
                    "Loading IndicConformer: %s",
                    INDICCONFORMER_MODEL_ID,
                )

                _indicconformer_model = AutoModel.from_pretrained(
                    INDICCONFORMER_MODEL_ID,
                    trust_remote_code=True,
                )

            except Exception as exc:
                logger.exception(
                    "IndicConformer model loading failed."
                )

                raise RuntimeError(
                    f"IndicConformer model load failed: {exc}"
                ) from exc

            logger.info(
                "IndicConformer loaded successfully."
            )

            return _indicconformer_model

    @staticmethod
    def _load_audio_for_indicconformer(
        file_path: str,
    ) -> tuple[Any, float]:
        """
        Load audio as:

            float32
            mono
            16 kHz
            shape [1, samples]

        Uses soundfile rather than torchaudio.load(), avoiding the
        TorchCodec requirement introduced by newer torchaudio versions.

        Returns:
            (waveform, duration_seconds)
        """

        import soundfile as sf
        import torch

        try:
            audio, sample_rate = sf.read(
                file_path,
                dtype="float32",
                always_2d=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read audio for Hindi ASR: {exc}"
            ) from exc

        audio = np.asarray(audio, dtype=np.float32)

        # --------------------------------------------------------------
        # Validate raw audio.
        # --------------------------------------------------------------

        if audio.size == 0:
            raise RuntimeError(
                "Hindi ASR received an empty audio file."
            )

        if not np.isfinite(audio).all():
            raise RuntimeError(
                "Hindi ASR audio contains NaN or infinite samples."
            )

        # --------------------------------------------------------------
        # Multi-channel -> mono.
        #
        # soundfile returns [samples, channels] for multichannel audio.
        # --------------------------------------------------------------

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if audio.ndim != 1:
            raise RuntimeError(
                f"Unexpected audio shape: {audio.shape}"
            )

        if sample_rate <= 0:
            raise RuntimeError(
                f"Invalid audio sample rate: {sample_rate}"
            )

        duration_seconds = (
            float(len(audio)) / float(sample_rate)
        )

        if duration_seconds <= 0:
            raise RuntimeError(
                "Audio duration is zero."
            )

        # --------------------------------------------------------------
        # Resample to 16 kHz.
        #
        # Use librosa here rather than torchaudio so this path does not
        # depend on TorchCodec at all.
        # --------------------------------------------------------------

        if sample_rate != TARGET_SAMPLE_RATE:
            try:
                import librosa

                audio = librosa.resample(
                    audio,
                    orig_sr=sample_rate,
                    target_sr=TARGET_SAMPLE_RATE,
                )

                audio = np.asarray(
                    audio,
                    dtype=np.float32,
                )

            except Exception as exc:
                raise RuntimeError(
                    f"Failed to resample Hindi audio from "
                    f"{sample_rate} Hz to {TARGET_SAMPLE_RATE} Hz: "
                    f"{exc}"
                ) from exc

            duration_seconds = (
                float(len(audio))
                / float(TARGET_SAMPLE_RATE)
            )

        # --------------------------------------------------------------
        # Convert to Torch.
        #
        # IndicConformer requires [batch, samples].
        # --------------------------------------------------------------

        waveform = torch.from_numpy(
            np.ascontiguousarray(audio)
        ).float()

        waveform = waveform.unsqueeze(0).contiguous()

        if waveform.ndim != 2:
            raise RuntimeError(
                "IndicConformer waveform must have shape "
                "[batch, samples], got "
                f"{tuple(waveform.shape)}"
            )

        if waveform.shape[0] != 1:
            raise RuntimeError(
                "IndicConformer currently expects exactly one "
                f"audio batch, got {waveform.shape[0]}."
            )

        if waveform.shape[1] == 0:
            raise RuntimeError(
                "IndicConformer received zero audio samples."
            )

        return waveform, duration_seconds

    def _transcribe_hindi(
        self,
        file_path: str,
    ) -> list[dict[str, Any]]:
        """
        Transcribe Hindi using AI4Bharat IndicConformer CTC.

        The language is explicitly fixed to Hindi.
        """

        import torch

        model = self._load_indicconformer()

        waveform, duration_seconds = (
            self._load_audio_for_indicconformer(
                file_path
            )
        )

        logger.info(
            "IndicConformer input: shape=%s sample_rate=%s "
            "duration=%.2fs",
            tuple(waveform.shape),
            TARGET_SAMPLE_RATE,
            duration_seconds,
        )

        try:
            with torch.inference_mode():
                output = model(
                    waveform,
                    "hi",
                    "ctc",
                )

        except Exception as exc:
            logger.exception(
                "Hindi ASR inference failed."
            )

            raise RuntimeError(
                f"Hindi ASR inference failed: {exc}"
            ) from exc

        text = self._extract_indicconformer_text(output)

        if not text:
            raise RuntimeError(
                "Hindi ASR completed but produced an empty transcript."
            )

        return [
            {
                "start": 0.0,
                "end": round(duration_seconds, 2),
                "text": text,
                "speaker": "caller",
                "detected_language": "hi",
            }
        ]

    @staticmethod
    def _extract_indicconformer_text(
        output: Any,
    ) -> str:
        """
        Extract a real transcript from the tested IndicConformer output.

        The current official Transformers remote-code implementation
        returns a string for the CTC call used by this project.

        Unexpected output types are rejected rather than converted to
        arbitrary Python object strings.
        """

        if isinstance(output, str):
            return output.strip()

        if isinstance(output, (list, tuple)):
            parts: list[str] = []

            for item in output:
                if not isinstance(item, str):
                    raise RuntimeError(
                        "IndicConformer returned a sequence containing "
                        f"an unexpected item type: {type(item).__name__}"
                    )

                item = item.strip()

                if item:
                    parts.append(item)

            return " ".join(parts).strip()

        raise RuntimeError(
            "IndicConformer returned an unexpected output type: "
            f"{type(output).__name__}"
        )

    # ======================================================================
    # Whisper segmentation
    # ======================================================================

    @staticmethod
    def _merge_whisper_segments(
        segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Merge Whisper segments for the existing dashboard timeline.

        This does not alter transcript words and does not invent timing.
        """

        if not segments:
            return []

        MIN_SEGMENT_SECONDS = 15.0
        HARD_CAP_SECONDS = 22.0

        SENTENCE_ENDINGS = (
            ".",
            "!",
            "?",
            ":",
            ";",
        )

        merged: list[dict[str, Any]] = []

        buffer_start = segments[0]["start"]
        buffer_end = segments[0]["end"]
        buffer_text = segments[0]["text"]

        for segment in segments[1:]:
            current_duration = (
                buffer_end - buffer_start
            )

            current_text = buffer_text.rstrip()

            sentence_finished = current_text.endswith(
                SENTENCE_ENDINGS
            )

            hard_cap_reached = (
                segment["end"] - buffer_start
            ) >= HARD_CAP_SECONDS

            minimum_reached = (
                current_duration >= MIN_SEGMENT_SECONDS
            )

            if (
                sentence_finished
                or minimum_reached
                or hard_cap_reached
            ):
                merged.append(
                    {
                        "start": round(
                            buffer_start,
                            2,
                        ),
                        "end": round(
                            buffer_end,
                            2,
                        ),
                        "text": current_text,
                        "speaker": "caller",
                        "detected_language": "en",
                    }
                )

                buffer_start = segment["start"]
                buffer_end = segment["end"]
                buffer_text = segment["text"]

            else:
                buffer_end = segment["end"]

                buffer_text = (
                    f"{current_text} "
                    f"{segment['text'].strip()}"
                ).strip()

        if buffer_text.strip():
            merged.append(
                {
                    "start": round(
                        buffer_start,
                        2,
                    ),
                    "end": round(
                        buffer_end,
                        2,
                    ),
                    "text": buffer_text.strip(),
                    "speaker": "caller",
                    "detected_language": "en",
                }
            )

        return merged