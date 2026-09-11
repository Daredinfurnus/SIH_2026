"""\nConversational analysis engine for helpline call transcripts.
Upload-based workflow with real ASR + NLP + acoustic + recommendation pipeline.
"""
from __future__ import annotations

import re
from typing import Any

from app.schemas import Emotion, RiskLevel

try:
    from app.services.nlp_service import get_nlp_service
except ImportError:
    get_nlp_service = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Prototype lexical signals — grouped by the dimension they inform
# ---------------------------------------------------------------------------

_STRESS_SIGNS = [
    "stressed", "stress", "anxious", "anxiety", "nervous", "scared",
    "terrified", "panic", "shaking", "can't breathe", "breathing",
    "heart racing", "can't sleep", "insomnia", "exhausted", "tired",
    "worn out", "overwhelmed", "tense", "restless", "jittery",
    "on edge", "constant worry", "worrying", "fearful",
]

_DISTRESS_SIGNS = [
    "sad", "sadness", "crying", "cry", "cry", "tears", "grief",
    "hopeless", "hopelessness", "helpless", "helplessness", "alone",
    "isolated", "isolated", "lonely", "suffering", "miserable",
    "depressed", "depression", "distress", "devastated", "broken",
    "pain", "hurt", "it hurts", "I can't", "I can't go on",
    "no point", "nothing helps", "I don't know what to do",
    "I don't know where to go", "please help", "help me",
    "I feel so", "I feel so alone", "I am tired of",
]

_SAFETY_SIGNS = [
    "not safe", "unsafe", "threat", "threatening", "threatened",
    "threats", "danger", "in danger", "harm", "hurt me",
    "they will", "they might", "what if they", "scared to",
    "scared of", "afraid of", "afraid for", "afraid to",
    "don't feel safe", "nowhere safe", "nowhere to go",
    "I am in danger", "please protect", "save me",
]

_IMMEDIATE_SAFETY_SIGNS = [
    "they are here", "they are coming", "right now", "urgent",
    "emergency", "call the police", "call someone now",
    "please hurry", "immediately", "now or never",
    "I am in immediate danger", "they are outside",
]

# Lexical emotion cues — used as a tie-breaker when multiple emotions compete
_EMOTION_CUE_MAP = {
    Emotion.FEAR: [
        "scared", "terrified", "fear", "fearful", "afraid",
        "frightened", "panic", "shake", "shaking", "don't feel safe",
    ],
    Emotion.ANXIETY: [
        "anxious", "anxiety", "worried", "worry", "nervous",
        "restless", "tense", "on edge", "constant worry",
    ],
    Emotion.SADNESS: [
        "sad", "sadness", "grief", "crying", "cry", "tears",
        "sorrow", "heartbroken", "devastated",
    ],
    Emotion.ANGER: [
        "angry", "anger", "furious", "rage", "frustrated",
        "irritated", "fed up", "had enough",
    ],
    Emotion.HOPELESSNESS: [
        "hopeless", "hopelessness", "no hope", "nothing will change",
        "no point", "it won't get better", "giving up",
    ],
    Emotion.HELPLESSNESS: [
        "helpless", "helplessness", "don't know what to do",
        "don't know where to go", "please help", "help me",
        "I can't", "I can't handle this", "no one can help",
    ],
    Emotion.DISTRESS: [
        "distress", "suffering", "miserable", "agony", "tormented",
        "torment", "in pain", "painful",
    ],
    Emotion.UNCERTAINTY: [
        "don't know", "not sure", "uncertain", "confused",
        "what if", "what happens", "what next", "I don't know",
        "I am not sure",
    ],
    Emotion.THREAT: [
        "threat", "threatening", "threatened", "threats", "danger",
        "harm", "they will", "they might", "what if they",
    ],
    Emotion.MIXED: [],  # fallback for overlapping cues
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _count_signals(text: str, signals: list[str]) -> int:
    """Count how many distinct signals appear in the text (case-insensitive)."""
    lower = text.lower()
    found = set()
    for sig in signals:
        # Simple substring match; a real model would use token/semantic matching.
        if sig in lower:
            found.add(sig)
    return len(found)


def _lexical_density(text: str) -> float:
    """Ratio of meaningful tokens to total words — a tiny quality proxy."""
    words = re.findall(r"[a-zA-Z][a-zA-Z']*", text)
    if not words:
        return 0.0
    meaningful = sum(1 for w in words if len(w) > 2)
    return meaningful / len(words)


def _base_confidence(text: str, signal_count: int) -> float:
    """Prototype confidence heuristic.

    More signals + more substance → higher confidence.
    Very short or signal-free text → lower confidence.
    """
    density = _lexical_density(text)
    base = 0.45 + (min(signal_count, 6) * 0.06) + (density * 0.25)
    return max(0.30, min(0.95, base))


# ---------------------------------------------------------------------------
# Public analysis interface
# ---------------------------------------------------------------------------

class AnalysisService:
    """Conversational analysis engine for helpline call transcripts."""

    def __init__(self, ai_provider: str | None = None) -> None:
        self.ai_provider = (ai_provider or "indicbert").lower()

    def analyze_segment(
        self,
        text: str,
        base_emotion: str | None = None,
        ai_provider: str | None = None,
    ) -> dict[str, Any]:
        """Analyze a single transcript segment and return prototype scores.

        Parameters:
            text:               The transcript segment text.
            base_emotion:       Optional emotion hint from the STT layer.
            ai_provider:        NLP provider override (e.g. "indicbert").

        When ai_provider is "indicbert" (or configured via constructor) and
        the text is in a non-English script, the NLP service computes indicator
        scores via cosine similarity against prototype embeddings.  These
        NLP-derived scores replace the English-only lexical scores for that
        segment — so Hindi (and other Indic languages) receive meaningful
        analysis without being forced into English translation.

        Returns a dict with:
            stress_score, distress_score, emotion, confidence,
            indicators, safety_flag, immediate_safety_flag
        """
        provider = (ai_provider or self.ai_provider).lower()
        use_nlp = provider == "indicbert"
        nlp_svc = get_nlp_service() if use_nlp and get_nlp_service else None

        if not text or not text.strip():
            return _empty_analysis()

        # ---- Script detection: Is this likely a non-English script? ----
        # When IndicBERT is configured, route any Indic-script text through NLP.
        # This covers Hindi (Devanagari), Tamil, Bengali, Telugu, Kannada,
        # Malayalam, Gujarati, Punjabi (Gurmukhi), Oriya, Urdu (Arabic script),
        # and any other Indic/regional language — all of which IndicBERT v2
        # supports. English (Latin script) stays on the lexical path so the
        # baseline lexical scores apply to pure-Latin text.
        nlp_enabled = (
            use_nlp
            and nlp_svc is not None
            and _looks_like_non_latin(text)
        )

        # ---- NLP path (multilingual) ----
        if nlp_enabled:
            nlp_indicators = nlp_svc.score_indicators(text)
            if nlp_indicators:
                return {
                    "stress_score": nlp_svc.derive_stress(nlp_indicators),
                    "distress_score": nlp_svc.derive_distress(nlp_indicators),
                    "emotion": nlp_svc.derive_emotion(nlp_indicators),
                    "confidence": nlp_svc.derive_confidence(nlp_indicators),
                    "indicators": nlp_svc.derive_indicators(nlp_indicators),
                    "safety_flag": bool(
                        any(
                            nlp_indicators.get(ind, 0.0) > 0.40
                            for ind in ("safety concern", "fear")
                        )
                    ),
                    "immediate_safety_flag": False,
                }

        # ---- Lexical path (baseline, English-focused) ----
        stress = _score_stress(text)
        distress = _score_distress(text)
        emotion = _classify_emotion(text, base_emotion)
        confidence = _base_confidence(
            text, _count_signals(text, _STRESS_SIGNS + _DISTRESS_SIGNS)
        )
        indicators = _extract_indicators(text)
        safety_flag = bool(_count_signals(text, _SAFETY_SIGNS) > 0)
        immediate_safety_flag = bool(
            _count_signals(text, _IMMEDIATE_SAFETY_SIGNS) > 0
        )

        return {
            "stress_score": stress,
            "distress_score": distress,
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "indicators": indicators,
            "safety_flag": safety_flag,
            "immediate_safety_flag": immediate_safety_flag,
        }


# ---------------------------------------------------------------------------
# Internal scoring functions
# ---------------------------------------------------------------------------

def _score_stress(text: str) -> int:
    """Prototype stress score from lexical stress signals and intensity cues."""
    stress_signals = _count_signals(text, _STRESS_SIGNS)
    intensity = _count_signals(text, ["very", "extremely", "so much", "constantly", "always", "never sleep", "really", "scared"])
    length_factor = min(len(text.split()) / 30.0, 1.0) * 12
    score = (stress_signals * 14) + (intensity * 10) + length_factor
    return max(0, min(100, score))


def _score_distress(text: str) -> int:
    """Prototype distress score from lexical distress signals."""
    distress_signals = _count_signals(text, _DISTRESS_SIGNS)
    intensity = _count_signals(text, ["so alone", "so scared", "nothing helps", "I can't go on", "no point"])
    length_factor = min(len(text.split()) / 30.0, 1.0) * 6
    score = (distress_signals * 10) + (intensity * 8) + length_factor
    return max(0, min(100, score))


def _classify_emotion(text: str, base_emotion: str | None = None) -> Emotion:
    """Pick the dominant emotion using lexical cues, with a base preference."""
    lower = text.lower()
    scores: dict[Emotion, int] = {}
    for emotion, cues in _EMOTION_CUE_MAP.items():
        if not cues:
            continue
        scores[emotion] = _count_signals(text, cues)

    # If the STT/transcript layer already suggests an emotion, nudge it upward.
    if base_emotion:
        try:
            candidate = Emotion(base_emotion)
            scores[candidate] = scores.get(candidate, 0) + 1
        except ValueError:
            pass

    if not scores or max(scores.values()) == 0:
        # No clear lexical signal → gentle default based on raw intensity.
        if _count_signals(text, _DISTRESS_SIGNS) > _count_signals(text, _STRESS_SIGNS):
            return Emotion.DISTRESS
        return Emotion.UNCERTAINTY

    best = max(scores, key=lambda e: scores[e])
    return best


def _extract_indicators(text: str) -> list[str]:
    """Return an explainable list of conversational indicators for the segment."""
    indicators: list[str] = []
    lower = text.lower()

    checks = [
        ("fear", _SAFETY_SIGNS + _EMOTION_CUE_MAP[Emotion.FEAR]),
        ("anxiety", _EMOTION_CUE_MAP[Emotion.ANXIETY]),
        ("helplessness", _EMOTION_CUE_MAP[Emotion.HELPLESSNESS]),
        ("hopelessness", _EMOTION_CUE_MAP[Emotion.HOPELESSNESS]),
        ("distress", _EMOTION_CUE_MAP[Emotion.DISTRESS]),
        ("sleep disturbance", ["can't sleep", "insomnia", "unable to sleep", "not sleeping"]),
        ("threat-related context", _EMOTION_CUE_MAP[Emotion.THREAT] + _SAFETY_SIGNS),
        ("isolation", ["alone", "isolated", "lonely", "no one", "no one believes me", "no one will"]),
        ("uncertainty", _EMOTION_CUE_MAP[Emotion.UNCERTAINTY]),
        ("safety concern", _SAFETY_SIGNS),
        ("relief", ["thank", "better", "relief", "glad", "feel better"]),
        ("calm", ["calm", "thank you", "feel a little better"]),
    ]

    for label, cues in checks:
        if _count_signals(text, cues) > 0:
            indicators.append(label)

    # Deduplicate while preserving order
    seen = set()
    unique: list[str] = []
    for ind in indicators:
        if ind not in seen:
            seen.add(ind)
            unique.append(ind)

    return unique if unique else ["general concern"]


def _looks_like_non_latin(text: str) -> bool:
    """
    Detect whether text contains any Indic-script characters.

    When IndicBERT is configured (AI_PROVIDER=indicbert), any text containing
    Indic-script characters is routed through the NLP path. IndicBERT v2 supports
    24 Indic languages (Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam,
    Gujarati, Punjabi, Oriya, Urdu, Marathi, Nepali, Sinhala, Assamese, etc.)
    plus English — so every regional language gets meaningful analysis without
    being forced into English translation.

    Only pure-Latin text (English and other Latin-based languages) stays on the
    English-only lexical path, which is the prototype baseline.

    Returns True if text contains at least one Indic-script character
    (Devanagari, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati,
    Gurmukhi, Oriya, Sinhala), False otherwise.
    """
    if not text:
        return False

    for ch in text:
        cp = ord(ch)
        # Only route through NLP if text contains non-Latin Indic script chars.
        # Common Latin-extended punctuation (em-dash, smart quotes, etc.) should
        # stay on the English lexical path.
        if (
            0x0900 <= cp <= 0x097F    # Devanagari (Hindi, Marathi, Nepali…)
            or 0x0980 <= cp <= 0x09FF  # Bengali
            or 0x0B80 <= cp <= 0x0BFF  # Tamil
            or 0x0C00 <= cp <= 0x0C7F  # Telugu
            or 0x0C80 <= cp <= 0x0CFF  # Kannada
            or 0x0D00 <= cp <= 0x0D7F  # Malayalam
            or 0x0A80 <= cp <= 0x0AFF  # Gujarati
            or 0x0A00 <= cp <= 0x0A7F  # Gurmukhi (Punjabi)
            or 0x0B00 <= cp <= 0x0B7F  # Oriya
            or 0x0900 <= cp <= 0x097F  # Sinhala
            or 0x0600 <= cp <= 0x06FF  # Arabic (Urdu/Hindī in Nastaʿlīq)
            or 0x0750 <= cp <= 0x077F  # Arabic Supplement
            or 0x08A0 <= cp <= 0x08FF  # Arabic Extended-A
        ):
            return True

    return False


def _empty_analysis() -> dict[str, Any]:
    return {
        "stress_score": 0,
        "distress_score": 0,
        "emotion": Emotion.UNCERTAINTY,
        "confidence": 0.20,
        "indicators": ["insufficient content for analysis"],
        "safety_flag": False,
        "immediate_safety_flag": False,
    }
