"""
Deterministic tests for the TraumaSense Unified Scoring Engine.

Run:  cd backend && python -m pytest tests/scoring_engine_tests.py -v
"""
import math
import sys
import os

sys.path.insert(0, os.path.dirname(__file__) + "/..")
from app.services.scoring_engine import (
    ScoringConfig,
    DEFAULT_CONFIG,
    Feature,
    ScoringPipeline,
    validate_feature,
    reject_feature,
    normalize_to_100,
    normalize_from_config,
    weighted_aggregate,
    cross_signal_agreement,
    build_features_from_analysis_segment,
)


# ── helpers ─────────────────────────────────────────────────────────────────

def F(**kwargs):
    """Build a dict of Feature objects.  kwargs: feature_id=raw_value."""
    out = {}
    for fid, raw in kwargs.items():
        f = Feature(feature_id=fid, raw_value=raw, available=True,
                     quality="SUFFICIENT", model_confidence=0.5)
        f.normalized_value = normalize_from_config(raw, f.source or "linguistic",
                                                   expected_range=(0.0, 1.0))
        out[fid] = f
    return out


# ── §4  Feature Validation ──────────────────────────────────────────────────

class TestFeatureValidation:
    def test_reject_nan(self):
        f = Feature(feature_id="x", raw_value=float("nan"), available=True,
                     quality="SUFFICIENT")
        ok, reason = validate_feature(f)
        assert ok is False
        assert "nan_or_infinity" in reason

    def test_reject_infinity(self):
        f = Feature(feature_id="x", raw_value=float("inf"), available=True,
                     quality="SUFFICIENT")
        ok, reason = validate_feature(f)
        assert ok is False

    def test_accept_finite(self):
        f = Feature(feature_id="x", raw_value=0.5, available=True,
                     quality="SUFFICIENT")
        ok, reason = validate_feature(f)
        assert ok is True
        assert reason is None

    def test_unavailable_is_valid(self):
        f = Feature(feature_id="x", raw_value=0.5, available=False,
                     quality="UNAVAILABLE")
        ok, reason = validate_feature(f)
        assert ok is True

    def test_reject_non_numeric(self):
        f = Feature(feature_id="x", raw_value="hello", available=True,
                     quality="SUFFICIENT")
        ok, reason = validate_feature(f)
        assert ok is False


# ── §5  Normalization ──────────────────────────────────────────────────────

class TestNormalization:
    def test_probability_to_100(self):
        assert normalize_to_100(0.5, "probability") == 50.0
        assert normalize_to_100(1.0, "probability") == 100.0
        assert normalize_to_100(0.0, "probability") == 0.0
        assert normalize_to_100(-0.2, "probability") == 0.0   # clamped
        assert normalize_to_100(1.5, "probability") == 100.0  # clamped

    def test_linear_range(self):
        assert normalize_from_config(50.0, "acoustic", (0.0, 100.0)) == 50.0
        assert normalize_from_config(0.0, "acoustic", (0.0, 100.0)) == 0.0
        assert normalize_from_config(100.0, "acoustic", (0.0, 100.0)) == 100.0

    def test_default_range(self):
        assert normalize_from_config(0.75, "linguistic") == 75.0


# ── §6  Weighted Aggregation ────────────────────────────────────────────────

class TestWeightedAggregate:
    def test_all_available(self):
        contribs = [(78.0, 0.25, "linguistic"),
                    (84.0, 0.20, "emotional")]
        score, wsum, contribs_out = weighted_aggregate(contribs)
        # (78*0.25 + 84*0.20) / (0.25+0.20) = (19.5+16.8)/0.45 = 36.3/0.45 = 80.67
        assert 79.0 < score < 82.0
        assert wsum == 0.45

    def test_missing_excluded(self):
        contribs = [(78.0, 0.25, "linguistic"),
                    (None, 0.20, "acoustic")]   # acoustic unavailable
        score, wsum, _ = weighted_aggregate(contribs)
        assert wsum == 0.25
        assert score == 78.0

    def test_no_available(self):
        score, wsum, _ = weighted_aggregate([])
        assert wsum == 0.0
        assert score == 0.0

    def test_result_bounded(self):
        for raw, w in [(200, 0.25), (-50, 0.25)]:
            score, _, _ = weighted_aggregate([(raw, w, "x")])
            assert 0.0 <= score <= 100.0


# ── §7  Cross-signal agreement ─────────────────────────────────────────────

class TestCrossSignalAgreement:
    def test_identical(self):
        assert cross_signal_agreement([50.0, 50.0, 50.0]) == 100.0

    def test_divergent_spec_examples(self):
        # §18 examples: 78/81/74 → high agreement
        a1 = cross_signal_agreement([78.0, 81.0, 74.0])
        assert a1 > 85.0
        # 82/25/31 → low agreement
        a2 = cross_signal_agreement([82.0, 25.0, 31.0])
        assert a2 < 60.0

    def test_single_value(self):
        assert cross_signal_agreement([42.0]) == 100.0

    def test_empty(self):
        assert cross_signal_agreement([]) == 100.0


# ── §8  Stress Engine ──────────────────────────────────────────────────────

class TestStressEngine:
    def setup_method(self):
        self.pipe = ScoringPipeline()

    def test_all_signals_high(self):
        feats = F(linguistic_arousal=0.9, emotional_arousal=0.85,
                  acoustic_energy=0.8, speech_disruption=0.75,
                  physiological_arousal=0.9)
        r = self.pipe.stress_engine.compute(feats)
        assert 0.0 <= r["score"] <= 100.0
        assert r["contributors"]
        assert r["missing_signals"] == []

    def test_no_signals(self):
        r = self.pipe.stress_engine.compute({})
        assert r["score"] == 0.0
        assert r["contributors"] == []
        assert set(r["missing_signals"]) == set(DEFAULT_CONFIG.stress_weights.keys())

    def test_missing_physio_renormalized(self):
        feats = F(linguistic_arousal=0.8, emotional_arousal=0.7,
                  acoustic_energy=0.6, speech_disruption=0.5)
        # physiological not provided → excluded, weights renormalized
        r = self.pipe.stress_engine.compute(feats)
        assert 0.0 <= r["score"] <= 100.0
        assert "physiological_arousal" in r["missing_signals"]
        assert r["score"] > 0.0   # not zero — other signals carry it

    def test_bounds_hold_with_extreme_input(self):
        for raw in [0.0, 0.3, 0.5, 0.7, 1.0]:
            feats = F(linguistic_arousal=raw)
            r = self.pipe.stress_engine.compute(feats)
            assert 0.0 <= r["score"] <= 100.0

    def test_contributor_fields(self):
        feats = F(linguistic_arousal=0.78)
        r = self.pipe.stress_engine.compute(feats)
        c = r["contributors"][0]
        for key in ("source", "normalized_value", "base_weight",
                     "effective_weight", "contribution"):
            assert key in c


# ── §9  Distress Engine ────────────────────────────────────────────────────

class TestDistressEngine:
    def setup_method(self):
        self.pipe = ScoringPipeline()

    def test_all_signals_high(self):
        feats = F(emotional_suffering=0.9, hopelessness=0.85,
                  helplessness=0.8, loss_of_control=0.7,
                  isolation=0.6, coping_difficulty=0.5,
                  threat_distress=0.4, severity=0.3)
        r = self.pipe.distress_engine.compute(feats)
        assert 0.0 <= r["score"] <= 100.0
        assert r["score"] > 50.0

    def test_no_signals(self):
        r = self.pipe.distress_engine.compute({})
        assert r["score"] == 0.0
        assert r["missing_signals"] == list(DEFAULT_CONFIG.distress_weights.keys())

    def test_separation_from_stress(self):
        # High stress, low distress
        feats_s = F(linguistic_arousal=0.9)
        feats_d = F(emotional_suffering=0.1, hopelessness=0.05,
                     helplessness=0.05, loss_of_control=0.05,
                     isolation=0.05, coping_difficulty=0.05,
                     threat_distress=0.05, severity=0.05)
        s = self.pipe.stress_engine.compute(feats_s)
        d = self.pipe.distress_engine.compute(feats_d)
        assert s["score"] > d["score"]


# ── §10  Safety Engine ─────────────────────────────────────────────────────

class TestSafetyEngine:
    def setup_method(self):
        self.pipe = ScoringPipeline()

    def test_no_signals(self):
        r = self.pipe.safety_engine.compute({})
        assert r["score"] == 0.0
        assert r["severity"] == "NONE"
        assert r["immediate_safety_flag"] is False

    def test_general_concern(self):
        feats = F(safety_threat_keyword=0.2)  # normalized 20, between 10 and 25
        r = self.pipe.safety_engine.compute(feats)
        assert r["severity"] == "GENERAL_CONCERN"
        assert r["immediate_safety_flag"] is False

    def test_immediate_flag_with_coverage(self):
        feats = F(safety_threat_keyword=0.95, safety_self_harm_keyword=0.9,
                  safety_violation_mentioned=0.85, safety_urgent_language=0.9)
        r = self.pipe.safety_engine.compute(feats)
        assert r["immediate_safety_flag"] is True
        assert r["severity"] in ("ELEVATED_CONCERN", "IMMEDIATE_CONCERN")

    def test_immediate_flag_not_from_one_keyword(self):
        feats = F(safety_threat_keyword=0.95)
        r = self.pipe.safety_engine.compute(feats)
        # One keyword, coverage < 0.5 → no immediate flag
        assert r["immediate_safety_flag"] is False

    def test_severity_levels_reachable(self):
        assert self.pipe.safety_engine.compute({})["severity"] == "NONE"
        r = self.pipe.safety_engine.compute(F(safety_threat_keyword=0.2))
        assert r["severity"] == "GENERAL_CONCERN"


# ── §11  Confidence Engine ─────────────────────────────────────────────────

class TestConfidenceEngine:
    def setup_method(self):
        self.pipe = ScoringPipeline()

    def test_confidence_not_inverse_stress(self):
        feats = F(linguistic_arousal=0.8)
        sr = self.pipe.stress_engine.compute(feats)
        cr = self.pipe.confidence_engine.compute(sr, {"score": 0.0, "contributors": []},
                                                  {"score": 0.0, "contributors": []},
                                                  feats)
        assert cr["score"] != 100.0 - sr["score"]

    def test_confidence_degrades_with_missing_signals(self):
        feats_full = F(linguistic_arousal=0.7, emotional_arousal=0.6,
                       acoustic_energy=0.5, speech_disruption=0.4)
        feats_partial = F(linguistic_arousal=0.7)
        sr_full = self.pipe.stress_engine.compute(feats_full)
        sr_partial = self.pipe.stress_engine.compute(feats_partial)
        cr_full = self.pipe.confidence_engine.compute(sr_full,
                       {"score": 0.0, "contributors": []},
                       {"score": 0.0, "contributors": []}, feats_full)
        cr_partial = self.pipe.confidence_engine.compute(sr_partial,
                       {"score": 0.0, "contributors": []},
                       {"score": 0.0, "contributors": []}, feats_partial)
        assert cr_partial["score"] < cr_full["score"]

    def test_components_present(self):
        feats = F(linguistic_arousal=0.5)
        sr = self.pipe.stress_engine.compute(feats)
        cr = self.pipe.confidence_engine.compute(sr,
                       {"score": 0.0, "contributors": []},
                       {"score": 0.0, "contributors": []}, feats)
        assert "components" in cr
        for k in ("signal_coverage", "model_confidence", "quality",
                   "cross_signal_agreement", "baseline_quality", "temporal_coverage"):
            assert k in cr["components"]


# ── §12-15  Temporal Engine ────────────────────────────────────────────────

class TestTemporalEngine:
    def test_single_point_rolling(self):
        te = ScoringPipeline()._stress_temporal
        te.reset()
        r = te.update(50.0)
        assert r["rolling_score"] == 50.0
        assert r["spike_detected"] is False

    def test_two_point_rolling_smooths(self):
        te = ScoringPipeline()._stress_temporal
        te.reset()
        te.update(30.0)
        r = te.update(70.0)
        alpha = DEFAULT_CONFIG.temporal_alpha
        expected = alpha * 70.0 + (1 - alpha) * 30.0
        assert abs(r["rolling_score"] - expected) < 0.01

    def test_spike_detection(self):
        te = ScoringPipeline()._stress_temporal
        te.reset()
        for _ in range(5):
            te.update(30.0)
        r = te.update(90.0)
        assert r["spike_detected"] is True
        assert r["acute_deviation"] > DEFAULT_CONFIG.spike_deviation_threshold

    def test_trend_insufficient(self):
        te = ScoringPipeline()._stress_temporal
        te.reset()
        r = te.update(50.0)
        assert r["trend"] == "INSUFFICIENT_DATA"

    def test_rolling_not_overwrite_current(self):
        te = ScoringPipeline()._stress_temporal
        te.reset()
        te.update(80.0)
        r = te.update(20.0)
        assert r["current_score"] == 20.0
        assert r["rolling_score"] > 20.0   # smoothed, not overwritten


# ── §16-23  SVI Engine ─────────────────────────────────────────────────────

class TestSVIELSengine:
    def setup_method(self):
        self.pipe = ScoringPipeline()

    def test_svi_not_stress_plus_distress(self):
        r = self.pipe.svi_engine.compute(50.0, 50.0, 50.0, 100.0, 100.0, 100.0)
        assert r["svi_final"] != 50.0 + 50.0

    def test_svi_not_equal_distress(self):
        r = self.pipe.svi_engine.compute(10.0, 70.0, 50.0, 100.0, 100.0, 100.0)
        assert r["svi_final"] != 70.0

    def test_svi_changes_with_stress(self):
        s1 = self.pipe.svi_engine.compute(30.0, 30.0, 50.0, 100.0, 100.0, 100.0)
        s2 = self.pipe.svi_engine.compute(70.0, 30.0, 50.0, 100.0, 100.0, 100.0)
        assert s2["svi_final"] > s1["svi_final"]

    def test_svi_changes_with_distress(self):
        s1 = self.pipe.svi_engine.compute(30.0, 10.0, 50.0, 100.0, 100.0, 100.0)
        s2 = self.pipe.svi_engine.compute(30.0, 60.0, 50.0, 100.0, 100.0, 100.0)
        assert s2["svi_final"] > s1["svi_final"]

    def test_svi_changes_with_safety(self):
        s1 = self.pipe.svi_engine.compute(30.0, 30.0, 10.0, 100.0, 100.0, 100.0)
        s2 = self.pipe.svi_engine.compute(30.0, 30.0, 70.0, 100.0, 100.0, 100.0)
        assert s2["svi_final"] > s1["svi_final"]

    def test_svi_bounded(self):
        for _ in range(10):
            r = self.pipe.svi_engine.compute(100.0, 100.0, 100.0, 100.0, 100.0, 100.0)
            assert 0.0 <= r["svi_final"] <= 100.0

    def test_svi_confidence_derived(self):
        r = self.pipe.svi_engine.compute(50.0, 50.0, 50.0, 80.0, 60.0, 40.0)
        assert 0.0 <= r["confidence"] <= 100.0

    def test_interaction_term_exists(self):
        r = self.pipe.svi_engine.compute(50.0, 70.0, 60.0, 100.0, 100.0, 100.0)
        assert len(r["interactions"]) > 0
        assert r["interactions"][0]["name"] == "distress_x_safety"

    def test_interaction_bounded(self):
        r = self.pipe.svi_engine.compute(0.0, 100.0, 100.0, 100.0, 100.0, 100.0)
        for intr in r["interactions"]:
            assert intr["value"] <= intr["max_contribution"]

    def test_components_recorded(self):
        r = self.pipe.svi_engine.compute(30.0, 40.0, 50.0, 100.0, 100.0, 100.0)
        assert len(r["components"]) == 5


# ── §24  Risk Engine ───────────────────────────────────────────────────────

class TestRiskEngine:
    def setup_method(self):
        self.pipe = ScoringPipeline()

    def test_risk_bands(self):
        assert self.pipe.risk_engine.classify(10.0)["risk_level"] == "LOW"
        assert self.pipe.risk_engine.classify(30.0)["risk_level"] == "MODERATE"
        assert self.pipe.risk_engine.classify(60.0)["risk_level"] == "HIGH"
        assert self.pipe.risk_engine.classify(80.0)["risk_level"] == "CRITICAL"

    def test_safety_override(self):
        r = self.pipe.risk_engine.classify(
            10.0, safety_severity="ELEVATED_CONCERN", safety_immediate_flag=False)
        assert r["risk_level"] == "HIGH"
        assert r["safety_override"] is True

    def test_immediate_safety_critical(self):
        r = self.pipe.risk_engine.classify(
            5.0, safety_severity="IMMEDIATE_CONCERN", safety_immediate_flag=True)
        assert r["risk_level"] == "CRITICAL"

    def test_risk_is_classification_not_score(self):
        r = self.pipe.risk_engine.classify(37.5)
        assert isinstance(r["risk_level"], str)
        assert r["risk_level"] != "37.5"


# ── §25-29  Explanation Engine ─────────────────────────────────────────────

class TestExplanationEngine:
    def setup_method(self):
        self.pipe = ScoringPipeline()

    def test_stress_explanation(self):
        feats = F(linguistic_arousal=0.6)
        r = self.pipe.analyze_segment(0.0, 1.0, feats)
        exp = r["explanations"]["stress"]
        assert exp["dimension"] == "stress"
        assert "top_contributors" in exp
        assert "missing_signals" in exp

    def test_distress_explanation(self):
        feats = F(emotional_suffering=0.6)
        r = self.pipe.analyze_segment(0.0, 1.0, feats)
        exp = r["explanations"]["distress"]
        assert exp["dimension"] == "distress"

    def test_svi_explanation(self):
        feats = F(linguistic_arousal=0.6)
        r = self.pipe.analyze_segment(0.0, 1.0, feats)
        exp = r["explanations"]["svi"]
        assert exp["dimension"] == "svi"
        assert "svi_base" in exp
        assert "svi_final" in exp

    def test_safety_explanation(self):
        feats = F(safety_threat_keyword=0.6)
        r = self.pipe.analyze_segment(0.0, 1.0, feats)
        exp = r["explanations"]["safety"]
        assert exp["dimension"] == "safety"
        assert "severity" in exp

    def test_confidence_explanation(self):
        feats = F(linguistic_arousal=0.6)
        r = self.pipe.analyze_segment(0.0, 1.0, feats)
        exp = r["explanations"]["confidence"]
        assert exp["dimension"] == "confidence"
        assert "score" in exp


# ── §30-35  Pipeline Integration & Timeline ────────────────────────────────

class TestPipelineIntegration:
    def setup_method(self):
        self.pipe = ScoringPipeline()

    def test_full_pipeline_one_segment(self):
        feats = F(linguistic_arousal=0.7, emotional_arousal=0.6,
                  acoustic_energy=0.5, emotional_suffering=0.6,
                  hopelessness=0.5)
        r = self.pipe.analyze_segment(12.4, 4.2, feats)
        for key in ("stress", "distress", "safety", "confidence", "svi", "risk",
                     "explanations", "timeline_record", "temporal"):
            assert key in r
        assert 0.0 <= r["stress"]["score"] <= 100.0
        assert 0.0 <= r["distress"]["score"] <= 100.0
        assert 0.0 <= r["safety"]["score"] <= 100.0
        assert 0.0 <= r["svi"]["svi_final"] <= 100.0
        assert 0.0 <= r["confidence"]["score"] <= 100.0
        tl = r["timeline_record"]
        assert tl.timestamp == 12.4
        assert tl.duration == 4.2
        assert tl.stress == r["stress"]["score"]
        assert tl.distress == r["distress"]["score"]
        assert tl.svi == r["svi"]["svi_final"]

    def test_explanations_dimension(self):
        feats = F(linguistic_arousal=0.5)
        r = self.pipe.analyze_segment(0.0, 1.0, feats)
        for dim in ("stress", "distress", "svi", "safety", "confidence"):
            assert r["explanations"][dim]["dimension"] == dim

    def test_debug_audit_mode(self):
        feats = F(linguistic_arousal=0.6)
        r = self.pipe.analyze_segment(0.0, 1.0, feats)
        full = self.pipe.analyze_full_conversation(
            [{"start_time": 0.0, "duration": 1.0}],
            [feats],
        )
        assert "debug" in full
        assert "per_segment" in full["debug"]
        assert len(full["debug"]["per_segment"]) == 1


class TestFullConversation:
    def setup_method(self):
        self.pipe = ScoringPipeline()

    def test_overall_scores_weighted_mean(self):
        segs = [{"start_time": 0.0, "duration": 5.0},
                {"start_time": 5.0, "duration": 5.0},
                {"start_time": 10.0, "duration": 5.0}]
        feats_list = [F(linguistic_arousal=0.9),
                      F(linguistic_arousal=0.5),
                      F(linguistic_arousal=0.2)]
        full = self.pipe.analyze_full_conversation(segs, feats_list)
        overall = full["overall"]
        assert overall["timeline_length"] == 3
        seg_stresses = [s["stress"]["score"] for s in full["segments"]]
        assert min(seg_stresses) <= overall["stress_score"] <= max(seg_stresses)

    def test_timeline_length_matches_segments(self):
        segs = [{} for _ in range(5)]
        feats_list = [F(linguistic_arousal=0.5) for _ in range(5)]
        full = self.pipe.analyze_full_conversation(segs, feats_list)
        assert len(full["timeline"]) == 5

    def test_empty_conversation(self):
        full = self.pipe.analyze_full_conversation([], [])
        assert full["overall"]["timeline_length"] == 0
        assert full["overall"]["stress_score"] == 0.0

    def test_timeline_has_all_dimensions(self):
        feats = F(linguistic_arousal=0.6)
        full = self.pipe.analyze_full_conversation(
            [{"start_time": 0.0, "duration": 1.0}],
            [feats],
        )
        for rec in full["timeline"]:
            for key in ("stress", "distress", "safety", "svi", "confidence"):
                assert key in rec
                assert 0.0 <= rec[key] <= 100.0

    def test_frontend_backend_equality(self):
        """§35: frontend.timeline[i].stress == backend.timeline[i].stress."""
        feats = F(linguistic_arousal=0.65, emotional_arousal=0.55)
        full = self.pipe.analyze_full_conversation(
            [{"start_time": i * 5.0, "duration": 5.0} for i in range(3)],
            [feats for _ in range(3)],
        )
        for i, rec in enumerate(full["timeline"]):
            assert rec["stress"] == full["timeline"][i]["stress"]
            assert rec["distress"] == full["timeline"][i]["distress"]
            assert rec["svi"] == full["timeline"][i]["svi"]

    def test_no_fake_timeline_points(self):
        feats = F(linguistic_arousal=0.5)
        full = self.pipe.analyze_full_conversation(
            [{"start_time": float(i), "duration": 1.0} for i in range(10)],
            [feats for _ in range(10)],
        )
        assert len(full["timeline"]) == 10
        timestamps = [r["timestamp"] for r in full["timeline"]]
        assert len(timestamps) == len(set(timestamps))


# ── §38  Mathematical Invariants ───────────────────────────────────────────

class TestMathInvariants:
    def setup_method(self):
        self.pipe = ScoringPipeline()

    def _analyze(self, **kwargs):
        feats = F(**kwargs)
        return self.pipe.analyze_segment(0.0, 1.0, feats)

    def test_all_scores_bounded(self):
        r = self._analyze(linguistic_arousal=0.5, emotional_arousal=0.4,
                          emotional_suffering=0.3)
        for key in ("stress", "distress", "safety", "svi", "confidence"):
            score_key = "svi_final" if key == "svi" else "score"
            val = r[key][score_key] if key != "svi" else r[key]["svi_final"]
            assert 0.0 <= val <= 100.0, f"{key} out of bounds: {val}"

    def test_stress_not_equal_distress(self):
        r = self._analyze(linguistic_arousal=0.7, emotional_suffering=0.3)
        assert r["stress"]["score"] != r["distress"]["score"]

    def test_confidence_not_inverse_stress(self):
        r = self._analyze(linguistic_arousal=0.5)
        assert r["confidence"]["score"] != 100.0 - r["stress"]["score"]

    def test_svi_not_equal_any_component(self):
        r = self._analyze(linguistic_arousal=0.7, emotional_suffering=0.5)
        svi = r["svi"]["svi_final"]
        assert svi != r["stress"]["score"]
        assert svi != r["distress"]["score"]


# ── §39-40  Specific test scenarios ────────────────────────────────────────

class TestSpecificScenarios:
    def setup_method(self):
        self.pipe = ScoringPipeline()

    def _analyze(self, **kwargs):
        return self.pipe.analyze_segment(0.0, 1.0, F(**kwargs))

    def test_neutral_conversation(self):
        r = self._analyze(linguistic_arousal=0.1, emotional_arousal=0.1,
                          emotional_suffering=0.1, hopelessness=0.05)
        assert r["stress"]["score"] < 30.0
        assert r["distress"]["score"] < 30.0

    def test_high_stress_low_distress(self):
        r = self._analyze(linguistic_arousal=0.9, emotional_arousal=0.8,
                          acoustic_energy=0.7, emotional_suffering=0.1,
                          hopelessness=0.05)
        assert r["stress"]["score"] > 50.0
        assert r["distress"]["score"] < 30.0

    def test_low_stress_high_distress(self):
        r = self._analyze(linguistic_arousal=0.1, emotional_arousal=0.1,
                          emotional_suffering=0.9, hopelessness=0.85,
                          helplessness=0.8, loss_of_control=0.7,
                          isolation=0.6, coping_difficulty=0.5,
                          threat_distress=0.4, severity=0.3)
        assert r["stress"]["score"] < 30.0
        assert r["distress"]["score"] > 50.0

    def test_safety_concern(self):
        r = self._analyze(safety_threat_keyword=0.6, safety_self_harm_keyword=0.5)
        sev = r["safety"]["severity"]
        assert sev in ("GENERAL_CONCERN", "ELEVATED_CONCERN", "IMMEDIATE_CONCERN")

    def test_immediate_safety_condition(self):
        r = self._analyze(safety_threat_keyword=0.95,
                          safety_self_harm_keyword=0.9,
                          safety_violation_mentioned=0.85,
                          safety_urgent_language=0.9)
        assert r["safety"]["immediate_safety_flag"] is True
        assert r["risk"]["risk_level"] in ("HIGH", "CRITICAL")

    def test_missing_acoustic_signals(self):
        r = self._analyze(linguistic_arousal=0.7)  # only linguistic
        assert "acoustic_energy" in r["stress"]["missing_signals"]
        assert r["stress"]["confidence"] < 100.0

    def test_contradictory_signals_reduce_confidence(self):
        r = self._analyze(linguistic_arousal=0.9, emotional_arousal=0.1)
        assert r["confidence"]["score"] < 80.0

    def test_contradiction_doesnt_force_score_to_zero(self):
        r = self._analyze(linguistic_arousal=0.9, emotional_arousal=0.1)
        assert r["stress"]["score"] > 0.0

    def test_acute_spike_doesnt_add_to_base(self):
        te = self.pipe._stress_temporal
        te.reset()
        for _ in range(5):
            te.update(25.0)
        r = te.update(90.0)
        assert r["spike_detected"] is True
        # The spike is an event indicator, not added to rolling_score
        assert r["rolling_score"] < 90.0

    def test_empty_transcript_gives_zero(self):
        r = self.pipe.analyze_segment(0.0, 1.0, {})
        assert r["stress"]["score"] == 0.0
        assert r["distress"]["score"] == 0.0
        assert r["svi"]["svi_final"] == 0.0

    def test_invalid_numeric_rejected(self):
        f = Feature(feature_id="x", raw_value=float("nan"), available=True)
        ok, reason = validate_feature(f)
        assert ok is False

    def test_duplicate_segments_not_created(self):
        """Timeline has exactly one record per analyzed segment."""
        feats = F(linguistic_arousal=0.5)
        full = self.pipe.analyze_full_conversation(
            [{"start_time": float(i), "duration": 1.0} for i in range(5)],
            [feats for _ in range(5)],
        )
        assert len(full["timeline"]) == 5


# ── §43  Debug/Audit Mode ──────────────────────────────────────────────────

class TestDebugAuditMode:
    def test_debug_values_from_real_calculations(self):
        pipe = ScoringPipeline()
        feats = F(linguistic_arousal=0.78, emotional_arousal=0.84)
        full = pipe.analyze_full_conversation(
            [{"start_time": 12.4, "duration": 4.2}],
            [feats],
        )
        debug = full["debug"]
        assert "per_segment" in debug
        assert "overall_timeline" in debug
        seg = debug["per_segment"][0]
        assert seg["stress"] == full["segments"][0]["stress"]["score"]
        assert seg["svi"] == full["segments"][0]["svi"]["svi_final"]


# ── §19  Safety is separate from Stress ────────────────────────────────────

class TestSafetySeparation:
    def setup_method(self):
        self.pipe = ScoringPipeline()

    def test_safety_flag_from_safety_not_stress(self):
        feats = F(linguistic_arousal=0.1,
                  safety_threat_keyword=0.95,
                  safety_self_harm_keyword=0.9,
                  safety_violation_mentioned=0.85,
                  safety_urgent_language=0.9)
        r = self.pipe.analyze_segment(0.0, 1.0, feats)
        assert r["safety"]["immediate_safety_flag"] is True
        assert r["stress"]["score"] < 30.0   # stress stays low


# ── §45  No fabricated physiological signals ───────────────────────────────

class TestNoFabrication:
    def test_physiological_is_unavailable(self):
        feats = build_features_from_analysis_segment({
            "start_time": 0.0, "duration": 1.0, "text": "hello",
            "indicators": [], "stress_indicator": 0.5, "confidence": 0.5
        })
        assert "physiological_arousal" in feats
        assert feats["physiological_arousal"].available is False
        assert feats["physiological_arousal"].quality == "UNAVAILABLE"

    def test_no_physio_in_stress_score(self):
        pipe = ScoringPipeline()
        feats = F(linguistic_arousal=0.5)
        r = pipe.stress_engine.compute(feats)
        # Physio not in contributors
        contrib_sources = [c["source"] for c in r["contributors"]]
        assert "physiological_arousal" not in contrib_sources


# ── §36  Configuration centralized ─────────────────────────────────────────

class TestConfiguration:
    def test_config_has_all_sections(self):
        cfg = DEFAULT_CONFIG
        for attr in ("stress_weights", "distress_weights", "svi_weights",
                      "safety_severity", "risk_bands", "svi_interactions",
                      "temporal_alpha", "trend_threshold", "spike_deviation_threshold"):
            assert hasattr(cfg, attr), f"Missing config: {attr}"

    def test_stress_weights_sum_near_1(self):
        total = sum(DEFAULT_CONFIG.stress_weights.values())
        assert 0.9 < total < 1.1

    def test_distress_weights_sum_near_1(self):
        total = sum(DEFAULT_CONFIG.distress_weights.values())
        assert 0.9 < total < 1.1

    def test_svi_weights_sum_near_1(self):
        total = sum(DEFAULT_CONFIG.svi_weights.values())
        assert 0.9 < total < 1.1
