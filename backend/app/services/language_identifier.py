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
import math
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

LID_CONFIDENCE_THRESHOLD = 0.35

# Chunked LID is used when whole-recording confidence is weak.
LID_CHUNK_SECONDS = 8.0
LID_CHUNK_STEP_SECONDS = 4.0
LID_MIN_CHUNK_SECONDS = 2.0

# Ignore genuinely silent/near-silent chunks.
LID_MIN_RMS = 0.003

# Prevent very long recordings from causing excessive LID inference.
LID_MAX_CHUNKS = 24

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
        """Run VoxLingua107 prediction on an audio file."""
        audio, sr = _load_audio_mono_16k(audio_path)
        return self.predict_array(audio, sr)

    def predict_array(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> dict[str, Any]:
        """
        Run VoxLingua107 prediction directly on an audio array.

        Uses the same Fbank, sentence normalization, ECAPA-TDNN,
        classifier, and label mapping as file-based prediction.
        """
        if not self._loaded:
            self.load()

        if self.compute_features is None:
            raise RuntimeError(
                "VoxLingua107 feature extractor is not initialized."
            )

        if self.mean_var_norm is None:
            raise RuntimeError(
                "VoxLingua107 normalizer is not initialized."
            )

        if self.embedding_model is None:
            raise RuntimeError(
                "VoxLingua107 embedding model is not initialized."
            )

        if self.classifier is None:
            raise RuntimeError(
                "VoxLingua107 classifier is not initialized."
            )

        if self.label_map is None:
            raise RuntimeError(
                "VoxLingua107 label map is not initialized."
            )

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if audio.size == 0:
            raise ValueError(
                "Cannot run VoxLingua107 on empty audio."
            )

        # All model inputs must be 16 kHz.
        if sr != 16000:
            import librosa

            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=16000,
            )
            sr = 16000

        waveform = (
            torch.from_numpy(audio)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        # ----------------------------------------------------------
        # Exact VoxLingua feature/model pipeline.
        # ----------------------------------------------------------
        with torch.no_grad():
            features = self.compute_features(waveform)

            features = self.mean_var_norm(
                features,
                torch.tensor(
                    [features.shape[1]],
                    device=self.device,
                ),
            )

            embedding = self.embedding_model(features)
            logits = self.classifier(embedding)

            probs = torch.softmax(
                logits,
                dim=-1,
            ).squeeze()

        top_idx = int(torch.argmax(probs).item())
        top_conf = float(probs[top_idx].item())

        language_code = "unknown"

        for label, idx in self.label_map.items():
            if idx == top_idx:
                language_code = (
                    label.split(":")[0]
                    .strip()
                    .lower()
                )
                break

        return {
            "language": language_code,
            "confidence": round(top_conf, 3),
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
        """
        Identify spoken language using whole-audio LID first, then
        speech-focused chunks when the whole recording is uncertain.

        English and Hindi are the only routable languages.
        Whisper is never used for language identification.
        """
        if not self._vox_available:
            self._ensure_vox()

        if not self._vox_available or self._vox_model is None:
            return {
                "language": "language_uncertain",
                "confidence": 0.0,
                "supported": False,
                "model_available": False,
                "note": f"LID unavailable: {self._vox_error or 'unknown'}",
            }

        try:
            # ----------------------------------------------------------
            # 1. Whole-recording prediction
            # ----------------------------------------------------------
            whole_result = self._vox_model.predict(audio_path)

            whole_language = whole_result.get("language", "unknown")

            try:
                whole_confidence = float(
                    whole_result.get("confidence", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                whole_confidence = 0.0

            if not math.isfinite(whole_confidence):
                whole_confidence = 0.0

            whole_confidence = max(
                0.0,
                min(1.0, whole_confidence),
            )

            # A strong supported prediction does not need chunking.
            if (
                whole_language in SUPPORTED_LANGUAGES
                and whole_confidence >= LID_CONFIDENCE_THRESHOLD
            ):
                return {
                    "language": whole_language,
                    "confidence": round(whole_confidence, 3),
                    "supported": True,
                    "model_available": True,
                    "note": (
                        "VoxLingua107 whole-audio decision: "
                        f"{whole_language}/{whole_confidence:.3f}"
                    ),
                }

            # ----------------------------------------------------------
            # 2. Load audio for chunk-level analysis
            # ----------------------------------------------------------
            import soundfile as sf

            audio, sr = sf.read(
                audio_path,
                dtype="float32",
                always_2d=False,
            )

            if audio.ndim > 1:
                audio = audio.mean(axis=-1)

            audio = np.asarray(audio, dtype=np.float32)

            if audio.size == 0:
                return {
                    "language": "language_uncertain",
                    "confidence": 0.0,
                    "supported": False,
                    "model_available": True,
                    "note": "Audio contains no samples.",
                }

            if sr != 16000:
                import librosa

                audio = librosa.resample(
                    audio,
                    orig_sr=sr,
                    target_sr=16000,
                )
                sr = 16000

            # Remove NaN/Inf before calculating RMS.
            audio = np.nan_to_num(
                audio,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)

            # ----------------------------------------------------------
            # 3. Generate candidate chunks
            #
            # We use RMS only as a cheap speech/energy gate.
            # This is NOT language detection.
            # ----------------------------------------------------------
            chunk_size = max(
                1,
                int(LID_CHUNK_SECONDS * sr),
            )
            step_size = max(
                1,
                int(LID_CHUNK_STEP_SECONDS * sr),
            )
            min_chunk_size = max(
                1,
                int(LID_MIN_CHUNK_SECONDS * sr),
            )

            candidates: list[np.ndarray] = []

            for start in range(
                0,
                len(audio),
                step_size,
            ):
                chunk = audio[start:start + chunk_size]

                if len(chunk) < min_chunk_size:
                    continue

                rms = float(
                    np.sqrt(np.mean(np.square(chunk))) + 1e-12
                )

                if not math.isfinite(rms):
                    continue

                if rms < LID_MIN_RMS:
                    continue

                candidates.append(chunk)

            if not candidates:
                return {
                    "language": "language_uncertain",
                    "confidence": round(whole_confidence, 3),
                    "supported": False,
                    "model_available": True,
                    "note": (
                        "No sufficiently audible speech chunks were "
                        "available for chunk-level language identification."
                    ),
                }

            # ----------------------------------------------------------
            # 4. Cap the number of expensive model inferences.
            #
            # Select chunks distributed across the recording rather
            # than processing only the beginning.
            # ----------------------------------------------------------
            if len(candidates) > LID_MAX_CHUNKS:
                indices = np.linspace(
                    0,
                    len(candidates) - 1,
                    num=LID_MAX_CHUNKS,
                    dtype=int,
                )
                candidates = [
                    candidates[int(i)]
                    for i in indices
                ]

            # ----------------------------------------------------------
            # 5. Run VoxLingua on each usable chunk
            # ----------------------------------------------------------
            language_scores: dict[str, list[float]] = {
                "en": [],
                "hi": [],
            }

            supported_chunk_count = 0

            for chunk in candidates:
                chunk_result = self._vox_model.predict_array(
                    chunk,
                    sr,
                )

                chunk_language = str(
                    chunk_result.get("language", "unknown")
                ).strip().lower()

                try:
                    chunk_confidence = float(
                        chunk_result.get("confidence", 0.0) or 0.0
                    )
                except (TypeError, ValueError):
                    chunk_confidence = 0.0

                if not math.isfinite(chunk_confidence):
                    chunk_confidence = 0.0

                chunk_confidence = max(
                    0.0,
                    min(1.0, chunk_confidence),
                )

                if chunk_language in language_scores:
                    language_scores[chunk_language].append(
                        chunk_confidence
                    )
                    supported_chunk_count += 1

            # ----------------------------------------------------------
            # 6. No EN/HI evidence
            # ----------------------------------------------------------
            if supported_chunk_count == 0:
                return {
                    "language": "language_uncertain",
                    "confidence": 0.0,
                    "supported": False,
                    "model_available": True,
                    "note": (
                        "VoxLingua107 found audible speech, but did not "
                        "produce usable English/Hindi evidence."
                    ),
                }

            en_scores = language_scores["en"]
            hi_scores = language_scores["hi"]

            en_mean = (
                float(np.mean(en_scores))
                if en_scores
                else 0.0
            )

            hi_mean = (
                float(np.mean(hi_scores))
                if hi_scores
                else 0.0
            )

            # Sum of confidence is useful because it considers both
            # confidence and the amount of supporting evidence.
            en_weight = float(np.sum(en_scores))
            hi_weight = float(np.sum(hi_scores))

            if en_weight >= hi_weight:
                final_language = "en"
                winning_scores = en_scores
                losing_scores = hi_scores
                winning_weight = en_weight
                losing_weight = hi_weight
            else:
                final_language = "hi"
                winning_scores = hi_scores
                losing_scores = en_scores
                winning_weight = hi_weight
                losing_weight = en_weight

            if not winning_scores:
                return {
                    "language": "language_uncertain",
                    "confidence": 0.0,
                    "supported": False,
                    "model_available": True,
                    "note": "No winning supported-language evidence.",
                }

            # ----------------------------------------------------------
            # 7. Confidence calculation
            # ----------------------------------------------------------
            mean_confidence = float(
                np.mean(winning_scores)
            )

            total_weight = (
                winning_weight + losing_weight
            )

            agreement = (
                winning_weight / total_weight
                if total_weight > 0.0
                else 0.0
            )

            # Strong confidence + agreement across chunks.
            combined_confidence = (
                0.75 * mean_confidence
                + 0.25 * agreement
            )

            combined_confidence = max(
                0.0,
                min(1.0, combined_confidence),
            )

            # ----------------------------------------------------------
            # 8. Require the final aggregated decision to clear
            # the tolerant threshold.
            # ----------------------------------------------------------
            if combined_confidence < LID_CONFIDENCE_THRESHOLD:
                return {
                    "language": "language_uncertain",
                    "confidence": round(combined_confidence, 3),
                    "supported": False,
                    "model_available": True,
                    "note": (
                        "Chunked VoxLingua107 analysis remained "
                        f"uncertain: confidence={combined_confidence:.3f}; "
                        f"chunks={len(candidates)}; "
                        f"en_mean={en_mean:.3f}; "
                        f"hi_mean={hi_mean:.3f}; "
                        f"agreement={agreement:.3f}"
                    ),
                }

            return {
                "language": final_language,
                "confidence": round(combined_confidence, 3),
                "supported": True,
                "model_available": True,
                "note": (
                    "VoxLingua107 chunked decision: "
                    f"{final_language}/{combined_confidence:.3f}; "
                    f"chunks={len(candidates)}; "
                    f"supported_chunks={supported_chunk_count}; "
                    f"en_mean={en_mean:.3f}; "
                    f"hi_mean={hi_mean:.3f}; "
                    f"agreement={agreement:.3f}"
                ),
            }

        except Exception as exc:
            logger.error(
                "VoxLingua107 inference failed: %s",
                exc,
                exc_info=True,
            )

            self._vox_available = False
            self._vox_error = str(exc)

            return {
                "language": "language_uncertain",
                "confidence": 0.0,
                "supported": False,
                "model_available": False,
                "note": f"LID inference failed: {exc}",
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
    """
    Convert authoritative VoxLingua107 LID output into an ASR routing decision.

    Supported:
        en -> Faster-Whisper (forced language="en")
        hi -> IndicConformer (forced language="hi")

    Rejected:
        ur -> unsupported
        any language other than en/hi -> unsupported
        low-confidence detection -> rejected
        unavailable/failed LID -> rejected

    IMPORTANT:
        Whisper is NEVER used for language detection or language routing.
        There is NO automatic-language-detection fallback.
        The pipeline fails closed whenever authoritative LID is unavailable
        or cannot produce a confident supported language.
    """

    # --------------------------------------------------------------
    # Validate LID result itself.
    # --------------------------------------------------------------
    if not isinstance(lid_result, dict):
        return {
            "route": "unsupported",
            "language": "language_uncertain",
            "confidence": 0.0,
            "reason": "Invalid language identification result.",
            "halt": True,
        }

    # --------------------------------------------------------------
    # Extract and sanitize authoritative LID values.
    # --------------------------------------------------------------
    model_available = bool(lid_result.get("model_available", False))

    language = lid_result.get("language")
    if not isinstance(language, str):
        language = "language_uncertain"
    else:
        language = language.strip().lower()

    try:
        confidence = float(lid_result.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    # Reject NaN / infinity rather than allowing them through the gate.
    if not math.isfinite(confidence):
        confidence = 0.0

    # Keep confidence in a valid probability range.
    confidence = max(0.0, min(1.0, confidence))

    supported = bool(lid_result.get("supported", False))

    # --------------------------------------------------------------
    # LID model MUST be available.
    # Never run ASR without authoritative language identification.
    # --------------------------------------------------------------
    if not model_available:
        return {
            "route": "unsupported",
            "language": "language_uncertain",
            "confidence": confidence,
            "reason": (
                "Language identification model is unavailable. "
                "ASR routing is halted because the language cannot "
                "be determined safely."
            ),
            "halt": True,
        }

    # --------------------------------------------------------------
    # Confidence gate.
    # --------------------------------------------------------------
    if confidence < LID_CONFIDENCE_THRESHOLD:
        return {
            "route": "unsupported",
            "language": language,
            "confidence": confidence,
            "reason": (
                f"Language identification confidence "
                f"{confidence:.3f} is below the required "
                f"threshold of {LID_CONFIDENCE_THRESHOLD:.2f}."
            ),
            "halt": True,
        }

    # --------------------------------------------------------------
    # Only English and Hindi are supported.
    #
    # This explicitly rejects Urdu and every other language.
    # --------------------------------------------------------------
    if language not in SUPPORTED_LANGUAGES:
        return {
            "route": "unsupported",
            "language": language,
            "confidence": confidence,
            "reason": (
                f"Detected language '{language}' is not supported. "
                "Currently supported languages are English (en) "
                "and Hindi (hi)."
            ),
            "halt": True,
        }

    # --------------------------------------------------------------
    # The LID service must agree that the detected language is
    # supported. This prevents contradictory LID output from being
    # routed accidentally.
    # --------------------------------------------------------------
    if not supported:
        return {
            "route": "unsupported",
            "language": language,
            "confidence": confidence,
            "reason": (
                f"Language identification reported '{language}', "
                "but marked it as unsupported. ASR routing is halted."
            ),
            "halt": True,
        }

    # --------------------------------------------------------------
    # English -> Faster-Whisper.
    # The ASR layer MUST receive language="en".
    # --------------------------------------------------------------
    if language == "en":
        return {
            "route": "en",
            "language": "en",
            "confidence": confidence,
            "reason": (
                "English speech detected by VoxLingua107. "
                "Route to Faster-Whisper with language='en'."
            ),
            "halt": False,
        }

    # --------------------------------------------------------------
    # Hindi -> IndicConformer.
    # The ASR layer MUST receive language="hi".
    # --------------------------------------------------------------
    if language == "hi":
        return {
            "route": "hi",
            "language": "hi",
            "confidence": confidence,
            "reason": (
                "Hindi speech detected by VoxLingua107. "
                "Route to IndicConformer with language='hi'."
            ),
            "halt": False,
        }

    # --------------------------------------------------------------
    # Defensive fail-closed path.
    #
    # This should be unreachable because SUPPORTED_LANGUAGES is
    # restricted to {"en", "hi"}, but keeping this makes the function
    # safe if the supported-language configuration changes later.
    # --------------------------------------------------------------
    return {
        "route": "unsupported",
        "language": language,
        "confidence": confidence,
        "reason": (
            f"Language '{language}' cannot be routed safely."
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
