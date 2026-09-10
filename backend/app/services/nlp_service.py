"""
IndicBERT v2 NLP service for multilingual conversational analysis.

Uses ai4bharat/IndicBERTv2-MLM-only to extract contextual embeddings from
transcript text in any of 24 supported languages (including Hindi, Bengali,
Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, and more),
then scores against prototype indicator embeddings via cosine similarity for
multilingual indicator detection without forcing translation to English.

Design:
- Lazy model loading: downloaded and loaded on first use, not at import time.
- Thread-safe: a lock guards model loading and inference.
- Graceful fallback: returns empty results when the model is unavailable,
  allowing the caller to use lexical analysis as a fallback.
- Script-based language detection: Devanagari → Hindi, Latin → English,
  other Indic scripts → "other". A heuristic, not a proper LID, but
  sufficient for routing text to the right analysis path.
- Prototype-based indicator scoring: cosine similarity between the segment
  embedding and pre-computed prototype embeddings for each indicator category.
  Scores above 0.5 indicate meaningful presence of that indicator.

When IndicBERT is configured (AI_PROVIDER=indicbert) and the text is
non-English, the analysis service uses these NLP scores to derive indicators,
stress, distress, emotion, and confidence — replacing the near-zero lexical
scores that English-only keyword matching produces for Indic-language text.
"""

from __future__ import annotations

import threading
from typing import Any

try:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False
    np = None          # type: ignore[assignment]
    torch = None       # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Prototype sentences for each indicator category.
# Each category has prototype sentences in English and Hindi (Devanagari).
# Add more (lang_code, sentence) pairs to extend language coverage.
# These are used to compute prototype embeddings at load time.
# ---------------------------------------------------------------------------

_PROTOTYPE_SENTENCES: dict[str, list[tuple[str, str]]] = {
    "fear": [
        ("en", "I am scared and afraid for my safety"),
        ("en", "I feel terrified and don't feel safe anywhere"),
        ("hi", "मुझे बहुत डर लगता है और मुझे अपनी सुरक्षा की चिंता है"),
        ("hi", "मैं डरा हुआ हूँ और मुझे नहीं पता कि क्या होगा"),
    ],
    "anxiety": [
        ("en", "I am very anxious and worried all the time"),
        ("en", "I can't stop worrying about what might happen next"),
        ("hi", "मुझे बहुत चिंता होती है और मैं हमेशा चिंतित रहता हूँ"),
        ("hi", "मुझे लगातार चिंता रहती है कि आगे क्या हो सकता है"),
    ],
    "helplessness": [
        ("en", "I don't know what to do, I feel completely helpless"),
        ("en", "No one can help me, I am alone in this situation"),
        ("hi", "मुझे नहीं पता क्या करना चाहिए, मैं बेकाबू हूँ"),
        ("hi", "कोई मेरी मदद नहीं कर सकता, मैं इस स्थिति में अकेला हूँ"),
    ],
    "hopelessness": [
        ("en", "I feel hopeless, nothing will ever change"),
        ("en", "There is no point, things won't get better"),
        ("hi", "मुझे एक उम्मीद नहीं है, कुछ कभी नहीं बदलेगा"),
        ("hi", "कोई फायदा नहीं, स्थिति नहीं सुधरेगी"),
    ],
    "distress": [
        ("en", "I am in great distress and suffering deeply"),
        ("en", "I feel miserable and tormented by what happened"),
        ("hi", "मैं बहुत तकलीफ में हूँ और गहरे तौर पर पीड़ित हूँ"),
        ("hi", "मुझे बेचैनी और पीड़ा हो रही है क्योंकि वह घटना हुई"),
    ],
    "safety concern": [
        ("en", "I don't feel safe in my own home anymore"),
        ("en", "There are threats against me and my family"),
        ("hi", "मुझे अपने घर में अब सुरक्षित महसूस नहीं होता"),
        ("hi", "मेरे खिलाफ और मेरे परिवार के खिलाफ धमकियाँ हैं"),
    ],
    "isolation": [
        ("en", "I feel alone and isolated, no one believes me"),
        ("en", "I have nowhere to go and no one to turn to for help"),
        ("hi", "मैं एकला और अलग महसूस करता हूँ, कोई मुझ पर विश्वास नहीं करता"),
        ("hi", "मेरे पास जाने की जगह नहीं है और कोई नहीं है जिस पर मैं भरोसा कर सकूँ"),
    ],
    "sleep disturbance": [
        ("en", "I can't sleep at all since this happened to me"),
        ("en", "I keep waking up and cannot get proper rest"),
        ("hi", "चूँकि यह घटना हुई है, मैं बिल्कुल सो नहीं पा रहा हूँ"),
        ("hi", "मैं बार-बार जाग जाता हूँ और ठीक से आराम नहीं कर पा रहा हूँ"),
    ],
}


# Indicator → stress contribution weight (for deriving stress_score from NLP)
_NLP_STRESS_WEIGHTS: dict[str, float] = {
    "fear": 0.35,
    "anxiety": 0.25,
    "safety concern": 0.25,
    "sleep disturbance": 0.15,
}

# Indicator → distress contribution weight
_NLP_DISTRESS_WEIGHTS: dict[str, float] = {
    "helplessness": 0.30,
    "hopelessness": 0.25,
    "distress": 0.25,
    "isolation": 0.20,
}

# Indicator → emotion mapping (priority order)
_NLP_EMOTION_PRIORITIES: list[tuple[str, list[str]]] = [
    ("Fear", ["fear", "safety concern"]),
    ("Helplessness", ["helplessness", "isolation"]),
    ("Hopelessness", ["hopelessness"]),
    ("Distress", ["distress"]),
    ("Anxiety", ["anxiety"]),
    ("Sadness", ["isolation"]),
]

# Emotion prototypes for direct emotion scoring (multilingual)
# These are used to compute prototype embeddings at load time.
_EMOTION_PROTOTYPES: dict[str, list[tuple[str, str]]] = {
    "fear": [
        ("en", "I am scared and afraid for my safety"),
        ("en", "I feel terrified and don't feel safe anywhere"),
        ("hi", "मुझे बहुत डर लगता है और मुझे अपनी सुरक्षा की चिंता है"),
        ("hi", "मैं डरा हुआ हूँ और मुझे नहीं पता कि क्या होगा"),
    ],
    "anxiety": [
        ("en", "I am very anxious and worried all the time"),
        ("en", "I can't stop worrying about what might happen next"),
        ("hi", "मुझे बहुत चिंता होती है और मैं हमेशा चिंतित रहता हूँ"),
        ("hi", "मुझे लगातार चिंता रहती है कि आगे क्या हो सकता है"),
    ],
    "helplessness": [
        ("en", "I don't know what to do, I feel completely helpless"),
        ("en", "No one can help me, I am alone in this situation"),
        ("hi", "मुझे नहीं पता क्या करना चाहिए, मैं बेकाबू हूँ"),
        ("hi", "कोई मेरी मदद नहीं कर सकता, मैं इस स्थिति में अकेला हूँ"),
    ],
    "hopelessness": [
        ("en", "I feel hopeless, nothing will ever change"),
        ("en", "There is no point, things won't get better"),
        ("hi", "मुझे एक उम्मीद नहीं है, कुछ कभी नहीं बदलेगा"),
        ("hi", "कोई फायदा नहीं, स्थिति नहीं सुधरेगी"),
    ],
    "distress": [
        ("en", "I am in great distress and suffering deeply"),
        ("en", "I feel miserable and tormented by what happened"),
        ("hi", "मैं बहुत तकलीफ में हूँ और गहरे तौर पर पीड़ित हूँ"),
        ("hi", "मुझे बेचैनी और पीड़ा हो रही है क्योंकि वह घटना हुई"),
    ],
    "sadness": [
        ("en", "I feel sad and depressed, crying all the time"),
        ("en", "I am grieving and feel heartbroken"),
        ("hi", "मुझे बहुत उदास और दुखी है, मैं बार-बार रो रहा हूँ"),
        ("hi", "मैं शोक और दिल टूटा हुआ महसूस कर रहा हूँ"),
    ],
    "anger": [
        ("en", "I am angry and furious about what they did"),
        ("en", "I feel rage and frustration"),
        ("hi", "मुझे बहुत गुस्सा और क्रोध आ रहा है"),
        ("hi", "मैं क्रोधित और हताश हूँ"),
    ],
    "safety concern": [
        ("en", "I don't feel safe in my own home anymore"),
        ("en", "There are threats against me and my family"),
        ("hi", "मुझे अपने घर में अब सुरक्षित महसूस नहीं होता"),
        ("hi", "मेरे खिलाफ और मेरे परिवार के खिलाफ धमकियाँ हैं"),
    ],
    "isolation": [
        ("en", "I feel alone and isolated, no one believes me"),
        ("en", "I have nowhere to go and no one to turn to for help"),
        ("hi", "मैं एकला और अलग महसूस करता हूँ, कोई मुझ पर विश्वास नहीं करता"),
        ("hi", "मेरे पास जाने की जगह नहीं है और कोई नहीं है जिस पर मैं भरोसा कर सकूँ"),
    ],
    "sleep disturbance": [
        ("en", "I can't sleep at all since this happened to me"),
        ("en", "I keep waking up and cannot get proper rest"),
        ("hi", "चूँकि यह घटना हुई है, मैं बिल्कुल सो नहीं पा रहा हूँ"),
        ("hi", "मैं बार-बार जाग जाता हूँ और ठीक से आराम नहीं कर पा रहा हूँ"),
    ],
}


class IndicBERTNLP:
    """
    Multilingual NLP service using IndicBERT v2 (ai4bharat/IndicBERTv2-MLM-only).

    Responsibilities:
    - Load the IndicBERT v2 model and tokenizer (lazy, thread-safe).
    - Compute contextual [CLS] embeddings for transcript text.
    - Score text against prototype indicator embeddings via cosine similarity.
    - Detect script/language of input text (Devanagari → Hindi, Latin → English).
    - Derive stress, distress, emotion, and confidence from indicator scores.

    Usage as a singleton: call get_nlp_service() once, use for all segments.
    Falls back gracefully when the model is unavailable.
    """

    MODEL_ID = "ai4bharat/IndicBERTv2-MLM-only"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._prototype_embeddings: dict[str, Any] = {}
        self._embedding_dim = 0
        self._loaded = False
        self._load_error: str | None = None

    def ensure_loaded(self) -> bool:
        """
        Load the model and precompute prototype embeddings.
        Thread-safe: only one thread loads, others wait.
        Returns True if loading succeeded, False otherwise.
        """
        if self._loaded:
            return self._model is not None and self._tokenizer is not None

        with self._lock:
            if self._loaded:
                return self._model is not None and self._tokenizer is not None

            if not _ML_AVAILABLE:
                self._load_error = (
                    "Machine learning dependencies not available "
                    "(transformers, torch, numpy)"
                )
                self._loaded = True
                return False

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
                self._model = AutoModel.from_pretrained(self.MODEL_ID)
                self._model.eval()
                self._embedding_dim = self._model.config.hidden_size

                # Precompute prototype embeddings for each indicator
                self._prototype_embeddings = {}
                for indicator, sentences in _PROTOTYPE_SENTENCES.items():
                    texts = [s for _, s in sentences]
                    emb = self._compute_mean_embedding(texts)
                    self._prototype_embeddings[indicator] = emb

                # Precompute emotion prototype embeddings
                self._emotion_embeddings: dict[str, Any] = {}
                for emotion, sentences in _EMOTION_PROTOTYPES.items():
                    texts = [s for _, s in sentences]
                    emb = self._compute_mean_embedding(texts)
                    self._emotion_embeddings[emotion] = emb

                self._loaded = True
                self._load_error = None
                return True
            except Exception as e:
                self._load_error = f"Failed to load IndicBERT: {e}"
                self._loaded = True  # prevent repeated retries
                return False

    def _compute_embedding(self, text: str) -> Any | None:
        """Compute L2-normalized [CLS] token embedding for a single text."""
        if self._model is None or self._tokenizer is None:
            return None

        try:
            inputs = self._tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
            # [CLS] token = first token of last hidden state
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            # L2 normalize
            norm = np.linalg.norm(cls_embedding)
            if norm > 0:
                cls_embedding = cls_embedding / norm
            return cls_embedding
        except Exception:
            return None

    def _compute_mean_embedding(self, texts: list[str]) -> Any:
        """Compute mean-normalized embedding across multiple texts."""
        embeddings = []
        for text in texts:
            emb = self._compute_embedding(text)
            if emb is not None:
                embeddings.append(emb)
        if not embeddings:
            return np.zeros(self._embedding_dim) if self._embedding_dim else np.array([])
        mean_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        return mean_emb

    def score_indicators(self, text: str) -> dict[str, float]:
        """
        Score a text segment against all indicator prototypes.

        Returns a dict mapping indicator name → cosine similarity score (0–1).
        Higher scores indicate stronger presence of that indicator in the text.

        The score is computed as (cosine_similarity + 1) / 2, mapping [-1, 1]
        to [0, 1]. Scores above 0.5 indicate meaningful similarity.
        """
        if not self.ensure_loaded():
            return {}

        if not self._prototype_embeddings:
            return {}

        embedding = self._compute_embedding(text)
        if embedding is None:
            return {}

        scores: dict[str, float] = {}
        for indicator, prototype in self._prototype_embeddings.items():
            sim = float(np.dot(embedding, prototype))
            # Map cosine similarity from [-1, 1] to [0, 1]
            scores[indicator] = max(0.0, min(1.0, (sim + 1.0) / 2.0))

        return scores

    def detect_language(self, text: str) -> str:
        """
        Approximate language detection based on script analysis.

        Returns:
        - "hi" for Devanagari script (Hindi, Marathi, Nepali, Sanskrit…)
        - "en" for Latin script (English and other Latin-based languages)
        - "other" for other Indic scripts (Bengali, Tamil, Telugu, Kannada,
          Malayalam, Gujarati, Punjabi/Gurmukhi…)

        This is a script-based heuristic, not a proper language identifier.
        It is sufficient for routing text to the appropriate analysis path.
        """
        if not text:
            return "en"

        for ch in text:
            cp = ord(ch)
            if 0x0900 <= cp <= 0x097F:  # Devanagari
                return "hi"

        for ch in text:
            cp = ord(ch)
            if (
                0x0980 <= cp <= 0x09FF    # Bengali
                or 0x0B80 <= cp <= 0x0BFF  # Tamil
                or 0x0C00 <= cp <= 0x0C7F  # Telugu
                or 0x0C80 <= cp <= 0x0CFF  # Kannada
                or 0x0D00 <= cp <= 0x0D7F  # Malayalam
                or 0x0A80 <= cp <= 0x0AFF  # Gujarati
                or 0x0A00 <= cp <= 0x0A7F  # Gurmukhi (Punjabi)
            ):
                return "other"

        return "en"

    def derive_stress(self, nlp_indicators: dict[str, float]) -> int:
        """Derive a stress score (0–100) from NLP indicator scores."""
        weighted = sum(
            nlp_indicators.get(ind, 0.0) * weight
            for ind, weight in _NLP_STRESS_WEIGHTS.items()
        )
        return max(0, min(100, int(weighted * 100)))

    def derive_distress(self, nlp_indicators: dict[str, float]) -> int:
        """Derive a distress score (0–100) from NLP indicator scores."""
        weighted = sum(
            nlp_indicators.get(ind, 0.0) * weight
            for ind, weight in _NLP_DISTRESS_WEIGHTS.items()
        )
        return max(0, min(100, int(weighted * 100)))

    def derive_emotion(self, nlp_indicators: dict[str, float]) -> str:
        """Derive an emotion label from NLP indicator scores."""
        from app.schemas import Emotion

        for emotion_name, indicators in _NLP_EMOTION_PRIORITIES:
            if any(nlp_indicators.get(ind, 0.0) > 0.45 for ind in indicators):
                try:
                    return Emotion(emotion_name).value
                except ValueError:
                    continue
        return Emotion.UNCERTAINTY.value

    def derive_confidence(self, nlp_indicators: dict[str, float]) -> float:
        """
        Derive confidence (0.0–1.0) from NLP indicator score distribution.

        Higher confidence when:
        - Top indicator score is high (strong signal).
        - Multiple indicators are above threshold (consistent signal).
        - Clear margin between top and second indicator.
        """
        if not nlp_indicators:
            return 0.5

        values = sorted(nlp_indicators.values(), reverse=True)
        top = values[0] if values else 0.0

        # Base confidence from top score
        conf = 0.50 + top * 0.30  # 0.50 to 0.80

        # Boost for multiple strong signals
        strong_count = sum(1 for v in values if v > 0.40)
        conf += min(strong_count, 3) * 0.05  # up to +0.15

        # Boost for clear margin
        if len(values) >= 2 and (top - values[1]) > 0.20:
            conf += 0.05

        return max(0.30, min(0.95, conf))

    def score_emotions(self, text: str) -> dict[str, float]:
        """
        Score a text segment against all emotion prototypes.

        Returns a dict mapping emotion name → cosine similarity score (0-1).
        Higher scores indicate stronger presence of that emotion in the text.

        Uses the same [CLS] embedding approach as score_indicators but against
        emotion-specific prototypes in both English and Hindi (Devanagari).
        """
        if not self.ensure_loaded():
            return {}

        if not self._emotion_embeddings:
            return {}

        embedding = self._compute_embedding(text)
        if embedding is None:
            return {}

        scores: dict[str, float] = {}
        for emotion, prototype in self._emotion_embeddings.items():
            sim = float(np.dot(embedding, prototype))
            scores[emotion] = max(0.0, min(1.0, (sim + 1.0) / 2.0))

        return scores

    def detect_accent_dialect(self, text: str) -> dict[str, float]:
        """
        Detect possible accent/dialect markers in text.

        Returns dict with accent signals and confidence:
            - 'hindi_dravidian_influence': float — markers suggesting
              Hindi spoken with Dravidian (South Indian) influence
            - 'hindi_urdū_overlap': float — markers suggesting Hindustani/Urdū
              overlap (Persian/Arabic loanwords in Devanagari)
            - 'confidence': float — overall detection confidence

        This is a script + vocabulary heuristic, not a trained classifier.
        It flags patterns that may indicate the speaker's regional background,
        which can be useful for operator context.
        """
        if not text:
            return {"hindi_dravidian_influence": 0.0, "hindi_urdū_overlap": 0.0, "confidence": 0.0}

        has_devanagari = any(0x0900 <= ord(c) <= 0x097F for c in text)
        if not has_devanagari:
            return {"hindi_dravidian_influence": 0.0, "hindi_urdū_overlap": 0.0, "confidence": 0.0}

        # Dravidian-influenced Hindi markers: use of certain South Indian
        # loan patterns, sentence-final particles, specific constructions.
        # These are approximate heuristics based on common features.
        dravidian_markers = [
            "ಯಾರ",  # Kannada influence markers (rare in pure Hindi)
            "ஆமாம்",  # Tamil influence
        ]
        urdu_overlap_markers = [
            "क़",    # Arabic-derived aspirated ق used in Urdū
            "ख़",    # Arabic-derived ख़ (kha) used in Urdū/Hindustani
            "ग़",    # Arabic-derived ग़ (gha)
            "ज़",    # Arabic-derived ज़ (za)
            "फ़",    # Arabic-derived फ़ (fa)
            "क़रिश",  # Urdū loan
            "महसूस",  # Urdū-influenced "feel" (vs Hindi "महसूस" vs "अनुभव")
            "मुबारिक",  # Urdū blessing
        ]

        lower = text
        dravidian_score = sum(1 for m in dravidian_markers if m in lower) / len(dravidian_markers)
        urdu_score = sum(1 for m in urdu_overlap_markers if m in lower) / len(urdu_overlap_markers)

        # Normalize to 0-1
        dravidian_score = min(1.0, dravidian_score)
        urdu_score = min(1.0, urdu_score)

        confidence = 0.30 + (dravidian_score + urdu_score) * 0.35

        return {
            "hindi_dravidian_influence": round(dravidian_score, 3),
            "hindi_urdū_overlap": round(urdu_score, 3),
            "confidence": round(min(0.95, confidence), 3),
        }

    def derive_indicators(
        self, nlp_indicators: dict[str, float], threshold: float = 0.40
    ) -> list[str]:
        """Derive a list of indicator labels from NLP scores above threshold."""
        result = [
            indicator
            for indicator, score in nlp_indicators.items()
            if score >= threshold
        ]
        return result if result else ["general concern"]


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_nlp_service: IndicBERTNLP | None = None


def get_nlp_service() -> IndicBERTNLP:
    """Get or create the singleton IndicBERTNLP instance."""
    global _nlp_service
    if _nlp_service is None:
        _nlp_service = IndicBERTNLP()
    return _nlp_service
