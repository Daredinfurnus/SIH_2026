"""
Wav2Vec2 acoustic/emotion indicator service.

Uses facebook/wav2vec2-base (pretrained on 960h LibriSpeech) to extract
acoustic features from audio segments and produce stress/distress/emotion
acoustic indicators.

Graceful degradation: when PyTorch/transformers are not installed, all
public functions return zero/default results. The module itself is always
importable.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

try:
    import torch
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False
    torch = None          # type: ignore[assignment]
    Wav2Vec2Model = None  # type: ignore[assignment]
    Wav2Vec2Processor = None  # type: ignore[assignment]

_WAV2VEC2_MODEL_ID = "facebook/wav2vec2-base"

_processor: "Wav2Vec2Processor | None" = None
_model: "Wav2Vec2Model | None" = None
_model_loaded = False


def _load_wav2vec2() -> None:
    global _model_loaded, _model, _processor
    if _model_loaded:
        return
    if not _ML_AVAILABLE:
        logger.warning("Wav2Vec2: PyTorch/transformers not installed — acoustic service unavailable")
        return
    logger.info("Loading wav2vec2-base (facebook/wav2vec2-base) — one-time cost (~1 GB, ~30 s on CPU)")
    _processor = Wav2Vec2Processor.from_pretrained(_WAV2VEC2_MODEL_ID)
    _model = Wav2Vec2Model.from_pretrained(_WAV2VEC2_MODEL_ID)
    _model.eval()
    _model_loaded = True


def _extract_audio_features(audio_path: str, start_sec: float, end_sec: float) -> dict[str, Any]:
    """Extract acoustic features for the audio segment [start_sec, end_sec].

    Returns dict with acoustic_stress_score, acoustic_confidence, acoustic_features.
    Returns zeros when ML is unavailable.
    """
    if not _ML_AVAILABLE:
        return {
            "acoustic_stress_score": 0,
            "acoustic_confidence": 0.0,
            "acoustic_features": {"error": "ML dependencies not installed"},
        }

    if not _model_loaded:
        _load_wav2vec2()
    if _model is None or _processor is None:
        return {
            "acoustic_stress_score": 0,
            "acoustic_confidence": 0.0,
            "acoustic_features": {"error": "wav2vec2 model not loaded"},
        }

    import numpy as np
    import soundfile as sf

    try:
        audio, sr = sf.read(audio_path)
    except Exception:
        import wave
        try:
            with wave.open(audio_path, "rb") as wf:
                n = wf.getnframes()
                sr = wf.getframerate()
                raw = wf.readframes(n)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            return {
                "acoustic_stress_score": 0,
                "acoustic_confidence": 0.0,
                "acoustic_features": {"error": "could not read audio"},
            }

    if sr <= 0:
        return {
            "acoustic_stress_score": 0,
            "acoustic_confidence": 0.0,
            "acoustic_features": {"error": "invalid sample rate"},
        }

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    seg = audio[start_sample:end_sample]

    if len(seg) < sr * 0.5:
        return {
            "acoustic_stress_score": 0,
            "acoustic_confidence": 0.0,
            "acoustic_features": {"error": "segment too short"},
        }

    import librosa
    if sr != 16000:
        seg = librosa.resample(seg, orig_sr=sr, target_sr=16000)
        sr = 16000

    inputs = _processor(seg, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        out = _model(**inputs)

    hidden_states = out.last_hidden_state  # (1, seq_len, 768)

    mean_activation = hidden_states.mean(dim=1).norm(dim=-1).item()
    temporal_var = hidden_states.var(dim=1).mean().item()
    latent_norm = hidden_states.norm(dim=-1).mean().item()

    raw_stress = (
        min(mean_activation / 8.0, 1.0) * 40
        + min(temporal_var / 200.0, 1.0) * 35
        + min(latent_norm / 15.0, 1.0) * 25
    )
    acoustic_stress = int(max(0, min(100, raw_stress * 100)))

    duration = (end_sec - start_sec)
    confidence = min(0.90, 0.30 + duration * 0.02)

    return {
        "acoustic_stress_score": acoustic_stress,
        "acoustic_confidence": round(confidence, 2),
        "acoustic_features": {
            "mean_activation": round(mean_activation, 4),
            "temporal_variance": round(temporal_var, 4),
            "latent_norm": round(latent_norm, 4),
            "duration_seconds": round(duration, 2),
        },
    }


class Wav2Vec2Service:
    """Acoustic feature extraction from audio segments via wav2vec2-base."""

    def analyze_segment(self, audio_path: str, start_sec: float, end_sec: float) -> dict[str, Any]:
        """Analyze an audio segment for acoustic stress indicators."""
        return _extract_audio_features(audio_path, start_sec, end_sec)

    def provider_name(self) -> str:
        return "wav2vec2"


_wav_svc: Wav2Vec2Service | None = None


def get_wav2vec2_service() -> Wav2Vec2Service:
    global _wav_svc
    if _wav_svc is None:
        _wav_svc = Wav2Vec2Service()
    return _wav_svc
