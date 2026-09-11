
"""
TraumaSense Unified Scoring Engine
====================================
Deterministic, explainable, synchronized, testable scoring pipeline.

Architecture:
  RAW INPUT → FEATURE EXTRACTION → FEATURE VALIDATION → NORMALIZATION →
  CORRELATED FEATURE FUSION → STRESS ENGINE → DISTRESS ENGINE → SAFETY ENGINE →
  CONFIDENCE ENGINE → TEMPORAL ENGINE → SVI ENGINE → RISK ENGINE →
  EXPLANATION ENGINE → TIMELINE → API → FRONTEND

Single authoritative engine. Frontend NEVER independently calculates scores.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoringConfig:
    stress_weights: Dict[str, float] = field(default_factory=lambda: {
        "linguistic_arousal":    0.25,
        "emotional_arousal":     0.20,
        "acoustic_energy":       0.20,
        "speech_disruption":     0.15,
        "physiological_arousal": 0.20,
    })
    distress_weights: Dict[str, float] = field(default_factory=lambda: {
        "emotional_suffering":   0.20,
        "hopelessness":          0.20,
        "helplessness":          0.15,
        "loss_of_control":       0.10,
        "isolation":             0.10,
        "coping_difficulty":     0.10,
        "threat_distress":       0.10,
        "severity":              0.05,
    })
    svi_weights: Dict[str, float] = field(default_factory=lambda: {
        "stress_component":      0.25,
        "distress_component":    0.30,
        "safety_component":      0.25,
        "context_component":     0.10,
        "persistence_component": 0.10,
    })
    safety_severity: Dict[str, float] = field(default_factory=lambda: {
        "none_threshold":        10.0,
        "general_concern":       25.0,
        "elevated_concern":      55.0,
        "immediate_concern":     80.0,
    })
    risk_bands: List[Tuple[str, float, float]] = field(default_factory=lambda: [
        ("LOW",       0.0,  24.0),
        ("MODERATE", 25.0,  49.0),
        ("HIGH",     50.0,  74.0),
        ("CRITICAL", 75.0, 100.0),
    ])
    safety_risk_override_severity: str = "ELEVATED_CONCERN"
    safety_risk_override_level: str = "HIGH"
    temporal_alpha: float = 0.35
    temporal_min_samples: int = 3
    trend_min_points: int = 3
    trend_window: int = 3
    trend_threshold: float = 5.0
    spike_deviation_threshold: float = 20.0
    conf_min_acceptable: float = 30.0
    svi_interactions: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "distress_x_safety", "formula": "distress_norm * safety_norm", "max_contribution": 15.0},
    ])


DEFAULT_CONFIG = ScoringConfig()


# ────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE REPRESENTATION
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Feature:
    feature_id: str
    raw_value: Optional[float] = None
    normalized_value: Optional[float] = None
    available: bool = False
    source: str = ""
    timestamp: Optional[float] = None
    window: Optional[float] = None
    model_confidence: float = 0.0
    quality: str = "UNKNOWN"
    rejection_reason: Optional[str] = None


# ────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE VALIDATION
# ────────────────────────────────────────────────────────────────────────────

class FeatureValidationError(Exception):
    def __init__(self, feature_id: str, reason: str):
        self.feature_id = feature_id
        self.reason = reason
        super().__init__(f"Feature {feature_id} rejected: {reason}")


def validate_feature(f: Feature) -> Tuple[bool, Optional[str]]:
    if not f.available:
        return True, None
    if f.raw_value is not None and not isinstance(f.raw_value, (int, float)):
        return False, f"non_numeric_type: {type(f.raw_value).__name__}"
    if f.raw_value is not None:
        if math.isnan(f.raw_value) or math.isinf(f.raw_value):
            return False, "nan_or_infinity"
    return True, None


def reject_feature(f: Feature, reason: str) -> Feature:
    f.available = False
    f.quality = "UNAVAILABLE"
    f.rejection_reason = reason
    f.normalized_value = None
    return f


# ────────────────────────────────────────────────────────────────────────────
# 4.  NORMALIZATION
# ────────────────────────────────────────────────────────────────────────────

def normalize_to_100(value: float, source: str, min_val: float = 0.0, max_val: float = 1.0) -> float:
    if source in ("probability", "classifier_confidence", "nlp_probability"):
        normalized = value * 100.0
    else:
        if max_val == min_val:
            normalized = 0.0
        else:
            normalized = (value - min_val) / (max_val - min_val) * 100.0
    return float(max(min(normalized, 100.0), 0.0))


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def normalize_from_config(raw: float, source: str, expected_range: Optional[Tuple[float, float]] = None) -> float:
    if expected_range:
        lo, hi = expected_range
        return normalize_to_100(raw, source, lo, hi)
    return normalize_to_100(raw, source, 0.0, 1.0)


# ────────────────────────────────────────────────────────────────────────────
# 5.  AVAILABILITY-AWARE WEIGHTED AGGREGATION
# ────────────────────────────────────────────────────────────────────────────

def weighted_aggregate(
    contributions: List[Tuple[float, float, str]],
) -> Tuple[float, float, List[Dict[str, Any]]]:
    total_weighted = 0.0
    total_weight = 0.0
    contributors = []

    for norm_val, base_weight, source in contributions:
        if norm_val is None or not math.isfinite(norm_val):
            continue
        effective = base_weight
        total_weighted += norm_val * effective
        total_weight += effective
        contributors.append({
            "source": source,
            "normalized_value": round(norm_val, 4),
            "base_weight": base_weight,
            "effective_weight": round(effective, 4),
            "contribution": round(norm_val * effective, 4),
        })

    if total_weight <= 0.0:
        return 0.0, 0.0, contributors

    score = total_weighted / total_weight
    return float(max(min(score, 100.0), 0.0)), total_weight, contributors


def coverage_ratio(available_weights: List[float], expected_weights: List[float]) -> float:
    exp_sum = sum(expected_weights)
    if exp_sum <= 0.0:
        return 1.0
    return sum(available_weights) / exp_sum


# ────────────────────────────────────────────────────────────────────────────
# 6.  CROSS-SIGNAL AGREEMENT
# ────────────────────────────────────────────────────────────────────────────

def cross_signal_agreement(values: List[float]) -> float:
    """Deterministic: agreement = clamp(100 - mad * scale, 0, 100). Scale=3."""
    if len(values) < 2:
        return 100.0
    mean = sum(values) / len(values)
    mad = sum(abs(v - mean) for v in values) / len(values)
    scale = 3.0
    return float(max(min(100.0 - mad * scale, 100.0), 0.0))


# ────────────────────────────────────────────────────────────────────────────
# 7.  ENGINES
# ────────────────────────────────────────────────────────────────────────────

class StressEngine:
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.cfg = config or DEFAULT_CONFIG

    def compute(self, features: Dict[str, Feature]) -> Dict[str, Any]:
        cfg = self.cfg
        contributions: List[Tuple[float, float, str]] = []
        available_weights: List[float] = []
        expected_weights: List[float] = []

        for fid, weight in cfg.stress_weights.items():
            f = features.get(fid)
            expected_weights.append(weight)
            if f is None or not f.available or f.normalized_value is None:
                available_weights.append(0.0)
                continue
            ok, reason = validate_feature(f)
            if not ok:
                f.available = False
                f.quality = "UNAVAILABLE"
                f.rejection_reason = reason
                available_weights.append(0.0)
                continue
            available_weights.append(weight)
            contributions.append((f.normalized_value, weight, fid))

        score, avail_sum, contributors = weighted_aggregate(contributions)
        coverage = coverage_ratio(available_weights, expected_weights)
        confidence = coverage * 100.0

        return {
            "score": round(score, 2),
            "confidence": round(confidence, 2),
            "coverage": round(coverage, 4),
            "contributors": contributors,
            "available_signals": [c["source"] for c in contributors],
            "missing_signals": [
                fid for fid, w in cfg.stress_weights.items()
                if fid not in [c["source"] for c in contributors]
            ],
        }


class DistressEngine:
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.cfg = config or DEFAULT_CONFIG

    def compute(self, features: Dict[str, Feature]) -> Dict[str, Any]:
        cfg = self.cfg
        contributions: List[Tuple[float, float, str]] = []
        available_weights: List[float] = []
        expected_weights: List[float] = []

        for fid, weight in cfg.distress_weights.items():
            f = features.get(fid)
            expected_weights.append(weight)
            if f is None or not f.available or f.normalized_value is None:
                available_weights.append(0.0)
                continue
            ok, reason = validate_feature(f)
            if not ok:
                f.available = False
                f.quality = "UNAVAILABLE"
                f.rejection_reason = reason
                available_weights.append(0.0)
                continue
            available_weights.append(weight)
            contributions.append((f.normalized_value, weight, fid))

        score, avail_sum, contributors = weighted_aggregate(contributions)
        coverage = coverage_ratio(available_weights, expected_weights)

        return {
            "score": round(score, 2),
            "confidence": round(coverage * 100.0, 2),
            "coverage": round(coverage, 4),
            "contributors": contributors,
            "available_signals": [c["source"] for c in contributors],
            "missing_signals": [
                fid for fid, w in cfg.distress_weights.items()
                if fid not in [c["source"] for c in contributors]
            ],
        }


class SafetyEngine:
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.cfg = config or DEFAULT_CONFIG

    def compute(self, features: Dict[str, Feature]) -> Dict[str, Any]:
        cfg = self.cfg
        safety_features = {
            "safety_threat_keyword":        0.30,
            "safety_self_harm_keyword":     0.25,
            "safety_violation_mentioned":   0.20,
            "safety_urgent_language":       0.15,
            "safety_protector_present":      0.10,
        }

        contributions: List[Tuple[float, float, str]] = []
        available_weights: List[float] = []
        expected_weights: List[float] = []

        for fid, weight in safety_features.items():
            f = features.get(fid)
            expected_weights.append(weight)
            if f is None or not f.available or f.normalized_value is None:
                available_weights.append(0.0)
                continue
            ok, reason = validate_feature(f)
            if not ok:
                f.available = False
                f.quality = "UNAVAILABLE"
                available_weights.append(0.0)
                continue
            available_weights.append(weight)
            contributions.append((f.normalized_value, weight, fid))

        score, avail_sum, contributors = weighted_aggregate(contributions)
        coverage = coverage_ratio(available_weights, expected_weights)
        severity = self._classify_severity(score)

        immediate_flag = (
            score >= cfg.safety_severity["immediate_concern"]
            and coverage >= 0.5
        )

        return {
            "score": round(score, 2),
            "severity": severity,
            "immediate_safety_flag": immediate_flag,
            "confidence": round(coverage * 100.0, 2),
            "coverage": round(coverage, 4),
            "contributors": contributors,
        }

    def _classify_severity(self, score: float) -> str:
        cfg = self.cfg
        if score <= cfg.safety_severity["none_threshold"]:
            return "NONE"
        if score <= cfg.safety_severity["general_concern"]:
            return "GENERAL_CONCERN"
        if score <= cfg.safety_severity["elevated_concern"]:
            return "ELEVATED_CONCERN"
        return "IMMEDIATE_CONCERN"


class ConfidenceEngine:
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.cfg = config or DEFAULT_CONFIG

    def compute(
        self,
        stress: Dict[str, Any],
        distress: Dict[str, Any],
        safety: Dict[str, Any],
        features: Dict[str, Feature],
        timeline_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        cfg = self.cfg

        all_expected_weights = (
            list(cfg.stress_weights.values()) + list(cfg.distress_weights.values())
        )
        all_available_weights = [
            w for _, w, _ in self._collect_all_contributions(stress, distress)
        ]
        coverage_score = coverage_ratio(
            all_available_weights if all_available_weights else [0.0],
            all_expected_weights,
        ) * 100.0

        model_confs = [
            f.model_confidence * 100.0
            for f in features.values()
            if f.available and f.normalized_value is not None
        ]
        model_conf_score = sum(model_confs) / len(model_confs) if model_confs else 0.0

        quality_scores = [
            self._quality_to_score(f.quality)
            for f in features.values()
            if f.available
        ]
        quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 50.0

        signal_values = [stress["score"], distress["score"]]
        if safety:
            signal_values.append(safety["score"])
        agreement = cross_signal_agreement(signal_values)

        baseline_score = 50.0
        if timeline_history and len(timeline_history) >= cfg.temporal_min_samples:
            baseline_score = min(100.0, len(timeline_history) * 10.0)

        temporal_score = min(100.0, (len(timeline_history) if timeline_history else 0) * 20.0)

        components = {
            "signal_coverage":     (coverage_score,      0.20),
            "model_confidence":    (model_conf_score,    0.20),
            "quality":             (quality_score,       0.15),
            "cross_signal_agreement": (agreement,        0.20),
            "baseline_quality":    (baseline_score,      0.15),
            "temporal_coverage":   (temporal_score,      0.10),
        }

        total = sum(v * w for v, w in components.values())
        total_weight = sum(w for _, w in components.values())
        final_confidence = total / total_weight if total_weight > 0 else 0.0

        return {
            "score": round(float(max(min(final_confidence, 100.0), 0.0)), 2),
            "components": {k: round(v, 2) for k, (v, _) in components.items()},
            "component_weights": {k: w for k, (_, w) in components.items()},
        }

    def _collect_all_contributions(self, stress, distress):
        result = []
        for entry in [stress, distress]:
            for c in entry.get("contributors", []):
                result.append((c["source"], c["base_weight"], c.get("source", "")))
        return result

    def _quality_to_score(self, q: str) -> float:
        return {
            "SUFFICIENT": 100.0, "LIMITED": 60.0, "INSUFFICIENT": 30.0,
            "UNAVAILABLE": 10.0, "UNKNOWN": 50.0,
        }.get(q, 50.0)


class TemporalEngine:
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.cfg = config or DEFAULT_CONFIG
        self._history: List[float] = []

    def update(self, current_score: float) -> Dict[str, Any]:
        self._history.append(current_score)
        rolling = self._rolling_average()
        deviation = current_score - rolling if rolling else 0.0
        spike = abs(deviation) >= self.cfg.spike_deviation_threshold
        trend = self._classify_trend()

        return {
            "current_score": round(current_score, 2),
            "rolling_score": round(rolling, 2) if rolling else None,
            "acute_deviation": round(deviation, 2),
            "spike_detected": spike,
            "trend": trend,
        }

    def _rolling_average(self) -> Optional[float]:
        if not self._history:
            return None
        if len(self._history) == 1:
            return self._history[0]
        alpha = self.cfg.temporal_alpha
        return alpha * self._history[-1] + (1 - alpha) * self._history[-2]

    def _classify_trend(self) -> str:
        cfg = self.cfg
        if len(self._history) < cfg.trend_min_points:
            return "INSUFFICIENT_DATA"
        window = self._history[-cfg.trend_window:]
        recent_mean = sum(window) / len(window)
        if len(self._history) > cfg.trend_window:
            prev_window = self._history[-2 * cfg.trend_window:-cfg.trend_window]
            prev_mean = sum(prev_window) / len(prev_window)
            diff = recent_mean - prev_mean
            if diff > cfg.trend_threshold:
                return "RISING"
            if diff < -cfg.trend_threshold:
                return "FALLING"
        return "STABLE"

    def reset(self):
        self._history.clear()


class SVIELSengine:
    """SVI != Stress + Distress. SVI != Distress. Separate composite."""

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.cfg = config or DEFAULT_CONFIG

    def compute(
        self,
        stress_score: float,
        distress_score: float,
        safety_score: float,
        stress_confidence: float,
        distress_confidence: float,
        safety_confidence: float,
        context_score: float = 0.0,
        context_confidence: float = 50.0,
        persistence_score: float = 0.0,
        persistence_confidence: float = 50.0,
    ) -> Dict[str, Any]:
        cfg = self.cfg

        components = {
            "stress_component":      (stress_score,      stress_confidence),
            "distress_component":    (distress_score,    distress_confidence),
            "safety_component":      (safety_score,      safety_confidence),
            "context_component":     (context_score,     context_confidence),
            "persistence_component": (persistence_score, persistence_confidence),
        }

        total_weighted = 0.0
        total_weight = 0.0
        component_results: List[Dict[str, Any]] = []

        for name, weight_fn in cfg.svi_weights.items():
            val, conf = components[name]
            eff_weight = weight_fn * (conf / 100.0)
            total_weighted += val * eff_weight
            total_weight += eff_weight
            component_results.append({
                "component": name,
                "value": round(val, 2),
                "confidence": round(conf, 2),
                "base_weight": weight_fn,
                "effective_weight": round(eff_weight, 4),
                "contribution": round(val * eff_weight, 4),
            })

        svi_base = total_weighted / total_weight if total_weight > 0 else 0.0
        svi_base = float(max(min(svi_base, 100.0), 0.0))

        comp_confs = [c["confidence"] for c in component_results]
        svi_confidence = sum(comp_confs) / len(comp_confs) if comp_confs else 0.0

        interaction_total = 0.0
        interaction_results: List[Dict[str, Any]] = []
        for term in cfg.svi_interactions:
            if term["name"] == "distress_x_safety":
                d_norm = distress_score / 100.0
                s_norm = safety_score / 100.0
                interaction = d_norm * s_norm * term["max_contribution"]
                interaction = float(max(min(interaction, term["max_contribution"]), 0.0))
                interaction_total += interaction
                interaction_results.append({
                    "name": term["name"],
                    "formula": term["formula"],
                    "value": round(interaction, 4),
                    "max_contribution": term["max_contribution"],
                })

        svi_final = float(max(min(svi_base + interaction_total, 100.0), 0.0))

        return {
            "svi_base": round(svi_base, 2),
            "interaction_total": round(interaction_total, 2),
            "svi_final": round(svi_final, 2),
            "confidence": round(svi_confidence, 2),
            "components": component_results,
            "interactions": interaction_results,
        }


class RiskEngine:
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.cfg = config or DEFAULT_CONFIG

    def classify(
        self, svi: float, safety_severity: str = "NONE",
        safety_immediate_flag: bool = False
    ) -> Dict[str, Any]:
        cfg = self.cfg
        risk_level = "LOW"
        for label, lo, hi in cfg.risk_bands:
            if lo <= svi <= hi:
                risk_level = label
                break

        override = False
        if safety_severity == cfg.safety_risk_override_severity:
            risk_level = cfg.safety_risk_override_level
            override = True
        elif safety_severity == "IMMEDIATE_CONCERN":
            risk_level = "CRITICAL"
            override = True

        return {
            "risk_level": risk_level,
            "svi_used": round(svi, 2),
            "safety_override": override,
            "safety_severity": safety_severity,
            "safety_immediate_flag": safety_immediate_flag,
        }


# ────────────────────────────────────────────────────────────────────────────
# 8.  EXPLANATION ENGINE
# ────────────────────────────────────────────────────────────────────────────

class ExplanationEngine:
    def build_stress_explanation(self, stress_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dimension": "stress",
            "score": stress_result["score"],
            "confidence": stress_result["confidence"],
            "evidence_quality": self._evidence_quality(stress_result),
            "top_contributors": sorted(
                stress_result.get("contributors", []),
                key=lambda c: abs(c["contribution"]), reverse=True,
            )[:5],
            "missing_signals": stress_result.get("missing_signals", []),
        }

    def build_distress_explanation(self, distress_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dimension": "distress",
            "score": distress_result["score"],
            "confidence": distress_result["confidence"],
            "evidence_quality": self._evidence_quality(distress_result),
            "top_contributors": sorted(
                distress_result.get("contributors", []),
                key=lambda c: abs(c["contribution"]), reverse=True,
            )[:5],
            "missing_signals": distress_result.get("missing_signals", []),
        }

    def build_svi_explanation(self, svi_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dimension": "svi",
            "svi_base": svi_result["svi_base"],
            "interaction_total": svi_result["interaction_total"],
            "svi_final": svi_result["svi_final"],
            "confidence": svi_result["confidence"],
            "components": svi_result.get("components", []),
            "interactions": svi_result.get("interactions", []),
        }

    def build_safety_explanation(self, safety_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dimension": "safety",
            "score": safety_result["score"],
            "severity": safety_result["severity"],
            "immediate_safety_flag": safety_result["immediate_safety_flag"],
            "confidence": safety_result["confidence"],
        }

    def build_confidence_explanation(self, confidence_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dimension": "confidence",
            "score": confidence_result["score"],
            "components": confidence_result["components"],
        }

    def _evidence_quality(self, engine_result: Dict[str, Any]) -> str:
        confidence = engine_result.get("confidence", 0.0)
        coverage = engine_result.get("coverage", 1.0)
        if confidence < 20.0 or coverage < 0.2:
            return "INSUFFICIENT"
        if confidence < 50.0 or coverage < 0.5:
            return "LIMITED"
        return "SUFFICIENT"


# ────────────────────────────────────────────────────────────────────────────
# 9.  TIMELINE
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class TimelineRecord:
    timestamp: float
    duration: float
    stress: float
    distress: float
    safety: float
    svi: float
    confidence: float
    risk_level: str
    contributors: List[Dict[str, Any]] = field(default_factory=list)
    events: List[str] = field(default_factory=list)


class TimelineBuilder:
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.cfg = config or DEFAULT_CONFIG

    def build_record(
        self,
        timestamp: float,
        duration: float,
        stress_score: float,
        distress_score: float,
        safety_score: float,
        svi_score: float,
        confidence: float,
        risk_level: str,
        contributors: Optional[List[Dict[str, Any]]] = None,
        events: Optional[List[str]] = None,
    ) -> TimelineRecord:
        return TimelineRecord(
            timestamp=round(timestamp, 2),
            duration=round(duration, 2),
            stress=round(stress_score, 2),
            distress=round(distress_score, 2),
            safety=round(safety_score, 2),
            svi=round(svi_score, 2),
            confidence=round(confidence, 2),
            risk_level=risk_level,
            contributors=contributors or [],
            events=events or [],
        )


# ────────────────────────────────────────────────────────────────────────────
# 10.  UNIFIED SCORING PIPELINE
# ────────────────────────────────────────────────────────────────────────────

class ScoringPipeline:
    def __init__(self, config: Optional[ScoringConfig] = None):
        self.cfg = config or DEFAULT_CONFIG
        self.stress_engine = StressEngine(self.cfg)
        self.distress_engine = DistressEngine(self.cfg)
        self.safety_engine = SafetyEngine(self.cfg)
        self.confidence_engine = ConfidenceEngine(self.cfg)
        self.temporal_engine = TemporalEngine(self.cfg)
        self.svi_engine = SVIELSengine(self.cfg)
        self.risk_engine = RiskEngine(self.cfg)
        self.explanation_engine = ExplanationEngine()
        self.timeline_builder = TimelineBuilder(self.cfg)
        self._stress_temporal = TemporalEngine(self.cfg)
        self._distress_temporal = TemporalEngine(self.cfg)

    def analyze_segment(
        self,
        timestamp: float,
        duration: float,
        features: Dict[str, Feature],
        timeline_history: Optional[List[TimelineRecord]] = None,
    ) -> Dict[str, Any]:
        for fid, f in list(features.items()):
            ok, reason = validate_feature(f)
            if not ok:
                f.available = False
                f.quality = "UNAVAILABLE"
                f.rejection_reason = reason

        stress_result = self.stress_engine.compute(features)
        distress_result = self.distress_engine.compute(features)
        safety_result = self.safety_engine.compute(features)

        stress_temporal = self._stress_temporal.update(stress_result["score"])
        distress_temporal = self._distress_temporal.update(distress_result["score"])

        confidence_result = self.confidence_engine.compute(
            stress_result, distress_result, safety_result,
            features, timeline_history,
        )

        _cv = features.get("context_vulnerability")
        svi_result = self.svi_engine.compute(
            stress_score=stress_result["score"],
            distress_score=distress_result["score"],
            safety_score=safety_result["score"],
            stress_confidence=stress_result["confidence"],
            distress_confidence=distress_result["confidence"],
            safety_confidence=safety_result["confidence"],
            context_score=_cv.normalized_value if (_cv and _cv.available) else 0.0,
            context_confidence=(_cv.model_confidence * 100.0) if (_cv and _cv.available) else 50.0,
            persistence_score=0.0,
            persistence_confidence=0.0,
        )

        risk_result = self.risk_engine.classify(
            svi=svi_result["svi_final"],
            safety_severity=safety_result["severity"],
            safety_immediate_flag=safety_result["immediate_safety_flag"],
        )

        explanations = {
            "stress": self.explanation_engine.build_stress_explanation(stress_result),
            "distress": self.explanation_engine.build_distress_explanation(distress_result),
            "svi": self.explanation_engine.build_svi_explanation(svi_result),
            "safety": self.explanation_engine.build_safety_explanation(safety_result),
            "confidence": self.explanation_engine.build_confidence_explanation(confidence_result),
        }

        events: List[str] = []
        if stress_temporal["spike_detected"]:
            events.append("acute_stress_spike")
        if distress_temporal["spike_detected"]:
            events.append("acute_distress_spike")
        if risk_result["safety_override"]:
            events.append("safety_risk_override")

        timeline_record = self.timeline_builder.build_record(
            timestamp=timestamp,
            duration=duration,
            stress_score=stress_result["score"],
            distress_score=distress_result["score"],
            safety_score=safety_result["score"],
            svi_score=svi_result["svi_final"],
            confidence=confidence_result["score"],
            risk_level=risk_result["risk_level"],
            contributors=(
                stress_result.get("contributors", [])
                + distress_result.get("contributors", [])
            ),
            events=events,
        )

        return {
            "timestamp": timestamp,
            "duration": duration,
            "stress": stress_result,
            "distress": distress_result,
            "safety": safety_result,
            "confidence": confidence_result,
            "svi": svi_result,
            "risk": risk_result,
            "explanations": explanations,
            "timeline_record": timeline_record,
            "temporal": {
                "stress": stress_temporal,
                "distress": distress_temporal,
            },
        }

    def analyze_full_conversation(
        self,
        segments: List[Dict[str, Any]],
        features_by_segment: List[Dict[str, Feature]],
    ) -> Dict[str, Any]:
        if len(segments) != len(features_by_segment):
            raise ValueError(
                f"Segment count mismatch: {len(segments)} segments vs "
                f"{len(features_by_segment)} feature-sets"
            )

        self._stress_temporal.reset()
        self._distress_temporal.reset()

        timeline: List[TimelineRecord] = []
        segment_results: List[Dict[str, Any]] = []

        for i, (seg, feats) in enumerate(zip(segments, features_by_segment)):
            result = self.analyze_segment(
                timestamp=seg.get("start_time", seg.get("timestamp", 0.0)),
                duration=seg.get("duration", seg.get("window", 0.0)),
                features=feats,
                timeline_history=timeline if i > 0 else None,
            )
            segment_results.append(result)
            timeline.append(result["timeline_record"])

        overall = self._compute_overall(timeline)
        debug = self._build_debug_info(segment_results, timeline)

        return {
            "overall": overall,
            "timeline": [self._record_to_dict(r) for r in timeline],
            "segments": segment_results,
            "debug": debug,
        }

    def _compute_overall(self, timeline: List[TimelineRecord]) -> Dict[str, Any]:
        n = len(timeline)
        if n == 0:
            return self._empty_overall()

        total_weight = 0.0
        weighted_stress = 0.0
        weighted_distress = 0.0
        weighted_safety = 0.0
        weighted_svi = 0.0
        weighted_confidence = 0.0

        for rec in timeline:
            w = max(rec.confidence, 1.0)
            total_weight += w
            weighted_stress += rec.stress * w
            weighted_distress += rec.distress * w
            weighted_safety += rec.safety * w
            weighted_svi += rec.svi * w
            weighted_confidence += rec.confidence * w

        if total_weight <= 0:
            return self._empty_overall()

        overall_safety_val = weighted_safety / total_weight
        risk = self.risk_engine.classify(
            weighted_svi / total_weight, "NONE", False
        )

        return {
            "stress_score": round(weighted_stress / total_weight, 2),
            "distress_score": round(weighted_distress / total_weight, 2),
            "safety_score": round(overall_safety_val, 2),
            "svi_score": round(weighted_svi / total_weight, 2),
            "confidence": round(weighted_confidence / total_weight, 2),
            "risk_level": risk["risk_level"],
            "safety_severity": risk.get("safety_severity", "NONE"),
            "safety_immediate_flag": risk.get("safety_immediate_flag", False),
            "timeline_length": n,
        }

    def _empty_overall(self) -> Dict[str, Any]:
        return {
            "stress_score": 0.0, "distress_score": 0.0,
            "safety_score": 0.0, "svi_score": 0.0,
            "confidence": 0.0, "risk_level": "LOW",
            "safety_severity": "NONE", "safety_immediate_flag": False,
            "timeline_length": 0,
        }

    def _record_to_dict(self, rec: TimelineRecord) -> Dict[str, Any]:
        return {
            "timestamp": rec.timestamp,
            "duration": rec.duration,
            "stress": rec.stress,
            "distress": rec.distress,
            "safety": rec.safety,
            "svi": rec.svi,
            "confidence": rec.confidence,
            "risk_level": rec.risk_level,
            "contributors": rec.contributors,
            "events": rec.events,
        }

    def _build_debug_info(
        self, segment_results: List[Dict[str, Any]], timeline: List[TimelineRecord]
    ) -> Dict[str, Any]:
        return {
            "per_segment": [
                {
                    "segment_index": i,
                    "stress": r["stress"]["score"],
                    "distress": r["distress"]["score"],
                    "safety": r["safety"]["score"],
                    "svi": r["svi"]["svi_final"],
                    "confidence": r["confidence"]["score"],
                    "risk_level": r["risk"]["risk_level"],
                    "stress_contributors": r["stress"].get("contributors", []),
                    "distress_contributors": r["distress"].get("contributors", []),
                    "svi_components": r["svi"].get("components", []),
                    "events": r["timeline_record"].events if "timeline_record" in r else [],
                }
                for i, r in enumerate(segment_results)
            ],
            "overall_timeline": [
                {
                    "timestamp": r.timestamp,
                    "stress": r.stress,
                    "distress": r.distress,
                    "safety": r.safety,
                    "svi": r.svi,
                    "confidence": r.confidence,
                    "risk_level": r.risk_level,
                    "events": r.events,
                }
                for r in timeline
            ],
        }


# ────────────────────────────────────────────────────────────────────────────
# 11.  LEGACY COMPATIBILITY — bridge from analysis segment → features
# ────────────────────────────────────────────────────────────────────────────

def build_features_from_analysis_segment(
    seg: Dict[str, Any],
    config: Optional[ScoringConfig] = None,
) -> Dict[str, Feature]:
    cfg = config or DEFAULT_CONFIG
    features: Dict[str, Feature] = {}

    def _maybe_feature(
        fid: str, raw: Optional[float], source: str,
        expected_range: Optional[Tuple[float, float]] = None,
        timestamp: Optional[float] = None,
        window: Optional[float] = None,
        model_confidence: float = 0.5,
    ) -> Optional[Feature]:
        if raw is None:
            f = Feature(
                feature_id=fid, source=source,
                available=False, quality="UNAVAILABLE",
                timestamp=timestamp, window=window,
                model_confidence=model_confidence,
            )
            features[fid] = f
            return f
        f = Feature(
            feature_id=fid,
            raw_value=float(raw),
            source=source,
            available=True,
            quality="SUFFICIENT",
            timestamp=timestamp,
            window=window,
            model_confidence=model_confidence,
        )
        try:
            f.normalized_value = normalize_from_config(raw, source, expected_range)
        except Exception:
            f.normalized_value = None
            f.available = False
            f.quality = "INSUFFICIENT"
        features[fid] = f
        return f

    ts = seg.get("start_time", seg.get("timestamp", 0.0))
    dur = seg.get("duration", seg.get("window", 0.0))

    # Linguistic / emotional features (from NLP/IndicBERT/emotion_service)
    _maybe_feature("linguistic_arousal", seg.get("stress_indicator"),
                   source="linguistic", expected_range=(0.0, 1.0),
                   timestamp=ts, window=dur, model_confidence=seg.get("confidence", 0.5))
    _maybe_feature("emotional_suffering", seg.get("distress_indicator"),
                   source="linguistic", expected_range=(0.0, 1.0),
                   timestamp=ts, window=dur, model_confidence=seg.get("confidence", 0.5))
    _maybe_feature("hopelessness", seg.get("hopelessness_indicator"),
                   source="linguistic", expected_range=(0.0, 1.0),
                   timestamp=ts, window=dur, model_confidence=seg.get("confidence", 0.5))
    _maybe_feature("helplessness", seg.get("helplessness_indicator"),
                   source="linguistic", expected_range=(0.0, 1.0),
                   timestamp=ts, window=dur, model_confidence=seg.get("confidence", 0.5))
    _maybe_feature("loss_of_control", seg.get("loss_of_control_indicator"),
                   source="linguistic", expected_range=(0.0, 1.0),
                   timestamp=ts, window=dur, model_confidence=seg.get("confidence", 0.5))
    _maybe_feature("isolation", seg.get("isolation_indicator"),
                   source="linguistic", expected_range=(0.0, 1.0),
                   timestamp=ts, window=dur, model_confidence=seg.get("confidence", 0.5))
    _maybe_feature("coping_difficulty", seg.get("coping_difficulty_indicator"),
                   source="linguistic", expected_range=(0.0, 1.0),
                   timestamp=ts, window=dur, model_confidence=seg.get("confidence", 0.5))
    _maybe_feature("threat_distress", seg.get("threat_indicator"),
                   source="linguistic", expected_range=(0.0, 1.0),
                   timestamp=ts, window=dur, model_confidence=seg.get("confidence", 0.5))
    _maybe_feature("severity", seg.get("severity_indicator"),
                   source="linguistic", expected_range=(0.0, 1.0),
                   timestamp=ts, window=dur, model_confidence=seg.get("confidence", 0.5))

    # Emotional classifier arousal
    emotion = seg.get("emotion", {})
    if isinstance(emotion, dict):
        arousal = emotion.get("arousal")
        _maybe_feature("emotional_arousal", arousal,
                       source="emotional", expected_range=(0.0, 1.0),
                       timestamp=ts, window=dur, model_confidence=emotion.get("confidence", 0.5))
        neg_emotions = ["sadness", "fear", "anger", "disgust", "shame", "guilt"]
        detected = emotion.get("primary", "").lower()
        if detected in neg_emotions:
            _maybe_feature("emotional_suffering", 1.0,
                           source="emotional", expected_range=(0.0, 1.0),
                           timestamp=ts, window=dur, model_confidence=emotion.get("confidence", 0.5))
        else:
            _maybe_feature("emotional_suffering", 0.1,
                           source="emotional", expected_range=(0.0, 1.0),
                           timestamp=ts, window=dur, model_confidence=emotion.get("confidence", 0.5))

    # Acoustic features
    acoustic = seg.get("acoustic", {})
    if isinstance(acoustic, dict):
        energy = acoustic.get("energy")
        _maybe_feature("acoustic_energy", energy,
                       source="acoustic", expected_range=(0.0, 1.0),
                       timestamp=ts, window=dur, model_confidence=acoustic.get("confidence", 0.0))
        disruption = acoustic.get("disruption")
        _maybe_feature("speech_disruption", disruption,
                       source="speech", expected_range=(0.0, 1.0),
                       timestamp=ts, window=dur, model_confidence=acoustic.get("confidence", 0.0))

    # Safety features (from existing indicator extraction)
    indicators = seg.get("indicators", [])
    if isinstance(indicators, list):
        has_threat = any(
            k in str(ind).lower()
            for ind in indicators
            for k in ["threat", "violence", "weapon", "harm", "danger"]
        )
        has_self_harm = any(
            k in str(ind).lower()
            for ind in indicators
            for k in ["suicide", "self-harm", "hurt myself", "end it"]
        )
        has_violation = any(
            k in str(ind).lower()
            for ind in indicators
            for k in ["violation", "assault", "abuse", "attack", "forced"]
        )
        has_urgent = any(
            k in str(seg.get("text", "")).lower()
            for k in ["now", "immediately", "right now", "help", "emergency", "urgent"]
        )
        has_protector = any(
            k in str(seg.get("text", "")).lower()
            for k in ["family", "friend", "support", "help available", "protect"]
        )

        _maybe_feature("safety_threat_keyword", 1.0 if has_threat else 0.0,
                       source="safety", expected_range=(0.0, 1.0),
                       timestamp=ts, window=dur, model_confidence=0.7)
        _maybe_feature("safety_self_harm_keyword", 1.0 if has_self_harm else 0.0,
                       source="safety", expected_range=(0.0, 1.0),
                       timestamp=ts, window=dur, model_confidence=0.7)
        _maybe_feature("safety_violation_mentioned", 1.0 if has_violation else 0.0,
                       source="safety", expected_range=(0.0, 1.0),
                       timestamp=ts, window=dur, model_confidence=0.7)
        _maybe_feature("safety_urgent_language", 1.0 if has_urgent else 0.0,
                       source="safety", expected_range=(0.0, 1.0),
                       timestamp=ts, window=dur, model_confidence=0.6)
        _maybe_feature("safety_protector_present", 1.0 if has_protector else 0.0,
                       source="safety", expected_range=(0.0, 1.0),
                       timestamp=ts, window=dur, model_confidence=0.5)

    # Physiological — NOT fabricated
    features["physiological_arousal"] = Feature(
        feature_id="physiological_arousal",
        source="physiological",
        available=False,
        quality="UNAVAILABLE",
        rejection_reason="no_physiological_sensors_available",
        timestamp=ts, window=dur,
    )

    # Context vulnerability
    context_val = seg.get("context_score", 0.0)
    _maybe_feature("context_vulnerability", context_val if context_val else None,
                   source="context", expected_range=(0.0, 1.0),
                   timestamp=ts, window=dur, model_confidence=0.4)

    return features
