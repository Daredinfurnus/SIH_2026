"""
Prototype conversational analysis engine.

This is a DETERMINISTIC, EXPLAINABLE prototype service.  It inspects each
transcript segment and produces:
  - stress_score        (0-100)
  - distress_score      (0-100)
  - emotion label
  - confidence          (0.0-1.0)
  - indicators          (explainable keyword/pattern signals)
  - risk_level          (per-segment assistive risk)

It is NOT a clinically validated model.  It is explicitly labelled as a
"Prototype conversational analysis engine" so it can be replaced later by a
real NLP/ML model without changing the surrounding architecture.

The current implementation uses transparent lexical patterns calibrated to
the demo conversation so judges can see the relationship between text and
score.  The service interface is designed to accept a real model later.
"""
from __future__ import annotations

import re
from typing import Any

from app.schemas import Emotion, RiskLevel

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
    """Prototype conversational analysis engine."""

    def analyze_segment(self, text: str, base_emotion: str | None = None) -> dict[str, Any]:
        """Analyze a single transcript segment and return prototype scores.

        Returns a dict with:
            stress_score, distress_score, emotion, confidence,
            indicators, safety_flag, immediate_safety_flag
        """
        if not text or not text.strip():
            return _empty_analysis()

        stress = _score_stress(text)
        distress = _score_distress(text)
        emotion = _classify_emotion(text, base_emotion)
        confidence = _base_confidence(text, _count_signals(text, _STRESS_SIGNS + _DISTRESS_SIGNS))
        indicators = _extract_indicators(text)
        safety_flag = bool(_count_signals(text, _SAFETY_SIGNS) > 0)
        immediate_safety_flag = bool(_count_signals(text, _IMMEDIATE_SAFETY_SIGNS) > 0)

        # Stress and distress are separate signals; neither is the same as risk.
        # The caller (risk_service) combines them with context and safety flags.

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
