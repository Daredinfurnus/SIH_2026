"""
Spoken language identification service for TraumaSense.

Runs BEFORE ASR to determine the spoken language of an audio recording,
then routes to the correct ASR:
    en  -> Faster-Whisper (language="en")
    hi  -> IndicConformer (language="hi")
    ur  -> REJECTED (unsupported, never sent to ASR)
    other -> REJECTED

Primary model: SpeechBrain VoxLingua107 ECAPA-TDNN.
The model's exact training pipeline (Fbank + InputNormalization) is used
for feature extraction to ensure compatibility.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import torch
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from speechbrain.lobes.models.Xvector import Classifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LID_CONFIDENCE_THRESHOLD = 0.65
SUPPORTED_LANGUAGES = frozenset({"en", "hi"})

MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
)
LID_DIR = os.path.join(MODELS_DIR, "lid", "voxlingua107")
REF_DIR = os.path.join(MODELS_DIR, "lid", "references")

LID_BUNDLE_PATH = os.path.join(LID_DIR, "lid_bundle.pt")


def _parse_label_encoder(path: str) -> dict[str, int]:
    """Parse VoxLingua107 label_encoder.txt into {label: index}."""
    result: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "================" or line.startswith("starting_index"):
                continue
            if " => " not in line:
                continue
            label_part, idx_part = line.split(" => ", 1)
            label = label_part.strip("'")
            try:
                idx = int(idx_part.strip())
            except ValueError:
                continue
            result[label] = idx
    return result


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _load_audio_mono_16k(path: str) -> tuple[np.ndarray, int]:
    """Load audio file as mono float32 numpy array at 16kHz."""
    import soundfile as sf
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        return audio.astype(np.float32), sr
    except Exception as exc:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        audio = waveform.cpu().numpy().astype(np.float32)
        if sr != 16000:
            import torchaudio.functional as F
            t = torch.from_numpy(audio)
            t = F.resample(t, sr, 16000)
            audio = t.numpy()
            sr = 16000
        return audio, sr


# ---------------------------------------------------------------------------
# VoxLingua107 model with matching feature pipeline
# ---------------------------------------------------------------------------

class _VoxLinguaModel:
    """VoxLingua107 prediction with matching Fbank + normalization pipeline."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.embedding_model: ECAPA_TDNN | None = None
        self.classifier: Classifier | None = None
        self.label_map: dict[str, int] | None = None
        self.compute_features: Fbank | None = None
        self.mean_var_norm: InputNormalization | None = None
        self._loaded = False

    def load(self) -> None:
        """Load model and feature pipeline from local checkpoints."""
        if not os.path.isdir(LID_DIR):
            raise FileNotFoundError(f"VoxLingua107 dir not found: {LID_DIR}")

        if os.path.isfile(LID_BUNDLE_PATH):
            self._load_bundle()
            return

        self._load_from_checkpoints()
        self._save_bundle()

    def _build_feature_pipeline(self) -> None:
        """Build Fbank + InputNormalization matching the hyperparams.yaml."""
        self.compute_features = Fbank(
            sample_rate=16000,
            n_mels=60,
            left_frames=0,
            right_frames=0,
            deltas=False,
        )
        self.mean_var_norm = InputNormalization(
            norm_type="sentence",
            std_norm=False,
        )

    def _load_bundle(self) -> None:
        bundle = torch.load(LID_BUNDLE_PATH, map_location=self.device, weights_only=False)
        self._build_models()
        self._build_feature_pipeline()
        self.embedding_model.load_state_dict(bundle["embedding_model"])
        self.classifier.load_state_dict(bundle["classifier"])
        self.label_map = bundle["label_map"]
        self._move_to_device()
        # Try to load normalizer state from bundle if present
        if "normalizer_state" in bundle:
            self._load_normalizer_state(bundle["normalizer_state"])
        self._loaded = True

    def _load_from_checkpoints(self) -> None:
        self._build_models()
        self._build_feature_pipeline()

        # Embedding model
        emb_path = os.path.join(LID_DIR, "embedding_model.ckpt")
        state = torch.load(emb_path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            key = k.replace("embedding_model.", "", 1) if k.startswith("embedding_model.") else k
            cleaned[key] = v
        self.embedding_model.load_state_dict(cleaned, strict=False)

        # Classifier
        clf_path = os.path.join(LID_DIR, "classifier.ckpt")
        state = torch.load(clf_path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {}
        for k, v in state.items():
            key = k.replace("classifier.", "", 1) if k.startswith("classifier.") else k
            cleaned[key] = v
        self.classifier.load_state_dict(cleaned, strict=False)

        # Label map
        label_file = os.path.join(LID_DIR, "label_encoder.txt")
        self.label_map = _parse_label_encoder(label_file)

        # Normalizer: load from checkpoint if available
        norm_path = os.path.join(LID_DIR, "normalizer.ckpt")
        if os.path.isfile(norm_path):
            try:
                self._load_normalizer_from_checkpoint(norm_path)
            except Exception as exc:
                logger.warning("LID: could not load normalizer state: %s", exc)

        self._move_to_device()
        self._loaded = True

    def _load_normalizer_state(self, state: dict[str, Any]) -> None:
        """Load normalizer running statistics from a dict."""
        if self.mean_var_norm is None:
            return
        if hasattr(self.mean_var_norm, "count"):
            self.mean_var_norm.count = state.get("count", 0)
        if hasattr(self.mean_var_norm, "glob_mean") and "glob_mean" in state:
            self.mean_var_norm.glob_mean = torch.tensor(state["glob_mean"], device=self.device)
        if hasattr(self.mean_var_norm, "glob_std") and "glob_std" in state:
            self.mean_var_norm.glob_std = torch.tensor(state["glob_std"], device=self.device)

    def _load_normalizer_from_checkpoint(self, path: str) -> None:
        """Load normalizer state from normalizer.ckpt."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self._load_normalizer_state(state)

    def _build_models(self) -> None:
        self.embedding_model = ECAPA_TDNN(
            input_size=60,
            channels=[1024, 1024, 1024, 1024, 3072],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            lin_neurons=256,
        )
        self.classifier = Classifier(
            input_shape=[None, None, 256],
            activation=torch.nn.LeakyReLU,
            lin_blocks=1,
            lin_neurons=512,
            out_neurons=107,
        )

    def _move_to_device(self) -> None:
        self.embedding_model.to(self.device).eval()
        self.classifier.to(self.device).eval()
        if self.compute_features is not None:
            self.compute_features.to(self.device)
        if self.mean_var_norm is not None:
            self.mean_var_norm.to(self.device)

    def _save_bundle(self) -> None:
        if self.embedding_model is None or self.classifier is None or self.label_map is None:
            return
        bundle = {
            "embedding_model": self.embedding_model.state_dict(),
            "classifier": self.classifier.state_dict(),
            "label_map": self.label_map,
        }
        # Save normalizer state if available
        if self.mean_var_norm is not None:
            norm_state = {}
            if hasattr(self.mean_var_norm, "count"):
                norm_state["count"] = self.mean_var_norm.count
            if hasattr(self.mean_var_norm, "glob_mean"):
                norm_state["glob_mean"] = self.mean_var_norm.glob_mean.tolist() if isinstance(self.mean_var_norm.glob_mean, torch.Tensor) else self.mean_var_norm.glob_mean
            if hasattr(self.mean_var_norm, "glob_std"):
                norm_state["glob_std"] = self.mean_var_norm.glob_std.tolist() if isinstance(self.mean_var_norm.glob_std, torch.Tensor) else self.mean_var_norm.glob_std
            bundle["normalizer_state"] = norm_state
        torch.save(bundle, LID_BUNDLE_PATH)
        logger.info("LID bundle saved to %s", LID_BUNDLE_PATH)

    def predict(self, audio_path: str) -> dict[str, Any]:
        """Run LID inference using the VoxLingua107 model with matching features."""
        if not self._loaded:
            self.load()

        # Load audio as torch tensor (batch, time)
        audio, sr = _load_audio_mono_16k(audio_path)
        waveform = torch.from_numpy(audio).float().unsqueeze(0)  # (1, time)

        # Compute Fbank features — output is (batch, time, n_mels=60)
        with torch.no_grad():
            features = self.compute_features(waveform)  # (1, T, 60)

            # Apply sentence-level mean normalization
            features = self.mean_var_norm(features, torch.tensor([features.shape[1]], device=self.device))

            # ECAPA-TDNN.forward expects (batch, time, channel) = (1, T, 60) — already correct
            embedding = self.embedding_model(features)
            logits = self.classifier(embedding)
            probs = torch.softmax(logits, dim=-1).squeeze()

        top_idx = torch.argmax(probs).item()
        top_conf = probs[top_idx].item()

        # Label indices in label_encoder.txt start at 0 (no offset)
        # The classifier outputs 107 logits, and argmax directly maps to label index
        language_code = "unknown"
        for label, idx in self.label_map.items():
            if idx == top_idx:
                language_code = label.split(":")[0].strip().lower()
                break

        return {
            "language": language_code,
            "confidence": round(float(top_conf), 3),
        }


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class LanguageIdentifier:
    """Spoken-language identification for TraumaSense audio routing.

    Primary: VoxLingua107 ECAPA-TDNN (model_available=True when loaded).
    The LID service runs BEFORE ASR to determine the spoken language.
    """

    def __init__(self) -> None:
        self._vox_model: _VoxLinguaModel | None = None
        self._vox_available = False
        self._vox_error: str | None = None

    def identify(self, audio_path: str) -> dict[str, Any]:
        """Identify the spoken language of an audio file."""
        if not self._vox_available:
            self._ensure_vox()

        if self._vox_available and self._vox_model is not None:
            try:
                result = self._vox_model.predict(audio_path)
                result["model_available"] = True
                result["supported"] = result["language"] in SUPPORTED_LANGUAGES
                if result["confidence"] < LID_CONFIDENCE_THRESHOLD:
                    result["language"] = "language_uncertain"
                    result["supported"] = False
                result["note"] = f"VoxLingua107: {result['language']}/{result['confidence']}"
                return result
            except Exception as exc:
                logger.error("VoxLingua107 inference failed: %s", exc, exc_info=True)
                self._vox_available = False
                self._vox_error = str(exc)

        # If we get here, VoxLingua107 is not available
        return {
            "language": "language_uncertain",
            "confidence": 0.0,
            "supported": False,
            "model_available": False,
            "note": f"LID unavailable: {self._vox_error or 'unknown'}",
        }

    def _ensure_vox(self) -> None:
        """Try to load the VoxLingua107 model."""
        if self._vox_available:
            return
        if self._vox_error:
            return

        try:
            logger.info("LID: loading VoxLingua107 on %s", self.device if hasattr(self, 'device') else "cpu")
            self._vox_model = _VoxLinguaModel(device="cpu")
            self._vox_model.load()
            self._vox_available = True
            logger.info("LID: VoxLingua107 loaded successfully")
        except Exception as exc:
            self._vox_error = f"VoxLingua107 load failed: {exc}"
            logger.warning("LID: %s", self._vox_error)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_lid_service: LanguageIdentifier | None = None


def get_language_identifier() -> LanguageIdentifier:
    """Get or create the singleton LanguageIdentifier instance."""
    global _lid_service
    if _lid_service is None:
        _lid_service = LanguageIdentifier()
    return _lid_service


# ---------------------------------------------------------------------------
# Router helper (used by routes.py)
# ---------------------------------------------------------------------------

def route_language(lid_result: dict[str, Any]) -> dict[str, Any]:
    """Convert an LID result into an ASR routing decision.

    Returns:
        dict with: route, language, reason, halt

    en  -> Whisper
    hi  -> IndicConformer
    ur  -> REJECTED
    other/unsupported -> REJECTED

    When model_available=False, falls back to Whisper auto-detect (route="auto").
    """
    model_available = lid_result.get("model_available", True)
    language = lid_result.get("language", "language_uncertain")
    confidence = lid_result.get("confidence", 0.0)
    supported = lid_result.get("supported", False)

    if not model_available:
        return {
            "route": "auto",
            "language": "auto-detected",
            "reason": "LID model unavailable -- using Whisper auto-detection for language routing.",
            "halt": False,
        }

    if language == "language_uncertain" or not supported:
        return {
            "route": "unsupported",
            "language": language,
            "reason": (
                "Language not supported or too uncertain to route. "
                "Currently supported: English (en), Hindi (hi)."
            ),
            "halt": True,
        }

    if language == "en":
        return {
            "route": "en",
            "language": "en",
            "reason": "English speech -- routed to Faster-Whisper (language=en).",
            "halt": False,
        }

    if language == "hi":
        return {
            "route": "hi",
            "language": "hi",
            "reason": "Hindi speech -- routed to IndicConformer (language=hi).",
            "halt": False,
        }

    # Any other language (including ur) -- reject
    return {
        "route": "unsupported",
        "language": language,
        "reason": (
            f"Detected language '{language}' is not supported. "
            "Currently supported: English (en), Hindi (hi)."
        ),
        "halt": True,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys as _sys
    import json as _json

    def _run_tests() -> None:
        lid = get_language_identifier()
        print(f"VoxLingua107 available: {lid._vox_available}")
        if lid._vox_error:
            print(f"VoxLingua107 error: {lid._vox_error[:200]}")
        print()

        test_cases = [
            ("test_en_tts.wav", "en"),
            ("test_hi_tts.wav", "hi"),
            ("test_ur_tts.wav", "ur"),
        ]

        for path, expected_lang in test_cases:
            result = lid.identify(path)
            routing = route_language(result)
            if expected_lang == "ur":
                status = "PASS" if routing["halt"] else "FAIL"
            else:
                status = "PASS" if (routing["halt"] is False and routing["route"] == expected_lang) else "FAIL"
            print(f"[{status}] {path} (expected {expected_lang})")
            print(f"  LID: language={result['language']}/{result['confidence']} supported={result['supported']} model_avail={result.get('model_available')}")
            print(f"  Route: {routing['route']} halt={routing['halt']}")
            print(f"  Reason: {routing['reason']}")
            print()

    _run_tests()
