"""Backend tests for TraumaSense MVP."""
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.api.routes import _mean_float, _mean_int
from app.schemas import AnalysisResponse, Emotion, RiskLevel, Speaker, TranscriptSegment
from app.services.analysis_service import AnalysisService
from app.services.svi_service import SVIService
from app.services.risk_service import RiskService
from app.services.recommendation_service import RecommendationService


# ===========================================================================
# Helpers
# ===========================================================================

def _seg(
    stress=50,
    distress=45,
    emotion="Fear",
    confidence=0.85,
    indicators=None,
    safety_flag=False,
    immediate_safety_flag=False,
    risk_level="MODERATE",
    risk_explanation=None,
    **overrides,
):
    d = dict(
        start=0.0,
        end=20.0,
        text="I feel scared and don't know what to do.",
        speaker=Speaker.CALLER.value,
        stress_score=stress,
        distress_score=distress,
        emotion=emotion,
        confidence=confidence,
        indicators=indicators or ["fear", "uncertainty"],
        safety_flag=safety_flag,
        immediate_safety_flag=immediate_safety_flag,
        risk_level=risk_level,
        risk_explanation=risk_explanation or ["Fear-related language"],
        svi_score=50,
        **overrides,
    )
    return d


# ===========================================================================
# Health / demo
# ===========================================================================

class TestHealthEndpoint(unittest.TestCase):
    def test_health_model_skeleton(self):
        hr = AnalysisResponse(
            case_id="CASE-26093-0001",
            file_name="demo.wav",
            duration_seconds=120.0,
            transcript=[_seg()],
            overall_stress_score=50,
            overall_distress_score=45,
            overall_svi_score=55,
            overall_risk_score=55,
            overall_risk_level=RiskLevel.MODERATE,
            overall_confidence=0.85,
            overall_indicators=["fear"],
            risk_explanation=["Fear-related language"],
            recommendation="Provide information.",
            mode="demo",
        )
        self.assertEqual(hr.case_id, "CASE-26093-0001")
        self.assertEqual(hr.overall_risk_level, RiskLevel.MODERATE)


# ===========================================================================
# Analysis service
# ===========================================================================

class TestAnalysisService(unittest.TestCase):
    def setUp(self):
        self.svc = AnalysisService()

    def test_empty_text_returns_low_scores(self):
        r = self.svc.analyze_segment("")
        self.assertEqual(r["stress_score"], 0)
        self.assertEqual(r["distress_score"], 0)
        self.assertLess(r["confidence"], 0.4)

    def test_fear_text_scores_stress(self):
        r = self.svc.analyze_segment("I feel really scared and don't feel safe.")
        self.assertGreater(r["stress_score"], 30)
        self.assertIn("fear", [i.lower() for i in r["indicators"]])

    def test_distress_text_scores_distress(self):
        r = self.svc.analyze_segment("I am hopeless and alone and crying every night.")
        self.assertGreater(r["distress_score"], 40)
        self.assertIn("hopelessness", [i.lower() for i in r["indicators"]])

    def test_confidence_ranges(self):
        r = self.svc.analyze_segment("I am scared.")
        self.assertGreaterEqual(r["confidence"], 0.0)
        self.assertLessEqual(r["confidence"], 1.0)

    def test_safety_flag_detection(self):
        r = self.svc.analyze_segment("I am in danger and they are here right now.")
        self.assertTrue(r["safety_flag"] or r["immediate_safety_flag"])

    def test_emotion_classification(self):
        r = self.svc.analyze_segment("I am so anxious and worried all the time.")
        self.assertIsInstance(r["emotion"], str)

    def test_indicators_list(self):
        r = self.svc.analyze_segment("I don't know what to do, I feel helpless and alone.")
        self.assertIsInstance(r["indicators"], list)
        self.assertTrue(len(r["indicators"]) > 0)


# ===========================================================================
# SVI service
# ===========================================================================

class TestSVIService(unittest.TestCase):
    def setUp(self):
        self.svc = SVIService()

    def test_empty_segments(self):
        r = self.svc.compute([], [], False)
        self.assertEqual(r["svi_score"], 0)

    def test_basic_svi(self):
        segs = [
            _seg(stress=50, distress=45, indicators=["fear", "uncertainty"], safety_flag=False),
            _seg(stress=60, distress=55, indicators=["fear", "helplessness"], safety_flag=True),
        ]
        r = self.svc.compute(segs, ["fear", "uncertainty", "helplessness"], False)
        self.assertGreaterEqual(r["svi_score"], 0)
        self.assertLessEqual(r["svi_score"], 100)

    def test_immediate_safety_escalation(self):
        segs = [_seg(stress=30, distress=28, indicators=["calm"], safety_flag=False)]
        r = self.svc.compute(segs, ["calm"], True)
        self.assertGreaterEqual(r["svi_score"], 65)

    def test_svi_breakdown_keys(self):
        segs = [_seg(stress=50, distress=45, indicators=["fear"])]
        r = self.svc.compute(segs, ["fear"], False)
        self.assertIn("svi_breakdown", r)
        bd = r["svi_breakdown"]
        self.assertIn("stress_component", bd)
        self.assertIn("distress_component", bd)


# ===========================================================================
# Risk service
# ===========================================================================

class TestRiskService(unittest.TestCase):
    def setUp(self):
        self.svc = RiskService()

    def test_low_risk(self):
        r = self.svc.assess(svi_score=15, overall_stress=15, overall_distress=12,
                            indicators=["calm"], confidence=0.85, immediate_safety=False)
        self.assertEqual(r["risk_level"], RiskLevel.LOW)
        self.assertLessEqual(r["risk_score"], 24)

    def test_moderate_risk(self):
        r = self.svc.assess(svi_score=40, overall_stress=40, overall_distress=38,
                            indicators=["anxiety", "uncertainty"], confidence=0.80, immediate_safety=False)
        self.assertEqual(r["risk_level"], RiskLevel.MODERATE)
        self.assertGreaterEqual(r["risk_score"], 25)
        self.assertLessEqual(r["risk_score"], 49)

    def test_high_risk(self):
        r = self.svc.assess(svi_score=62, overall_stress=62, overall_distress=60,
                            indicators=["fear", "helplessness", "threat-related context"],
                            confidence=0.88, immediate_safety=False)
        self.assertEqual(r["risk_level"], RiskLevel.HIGH)
        self.assertGreaterEqual(r["risk_score"], 50)
        self.assertLessEqual(r["risk_score"], 74)

    def test_critical_risk(self):
        r = self.svc.assess(svi_score=88, overall_stress=88, overall_distress=90,
                            indicators=["fear", "hopelessness", "helplessness", "threat-related context"],
                            confidence=0.92, immediate_safety=True)
        self.assertEqual(r["risk_level"], RiskLevel.CRITICAL)
        self.assertGreaterEqual(r["risk_score"], 75)

    def test_immediate_safety_escalation(self):
        # Low SVI but immediate safety flag should push risk up
        r = self.svc.assess(svi_score=20, overall_stress=20, overall_distress=18,
                            indicators=["safety concern"], confidence=0.80, immediate_safety=True)
        self.assertGreaterEqual(r["risk_score"], 60)

    def test_confidence_adjustment(self):
        r = self.svc.assess(svi_score=60, overall_stress=60, overall_distress=58,
                            indicators=["fear", "anxiety"], confidence=0.40, immediate_safety=False)
        self.assertLess(r["risk_score"], 60)

    def test_explanation_present(self):
        r = self.svc.assess(svi_score=60, overall_stress=60, overall_distress=58,
                            indicators=["fear", "helplessness"], confidence=0.85, immediate_safety=False)
        self.assertIsInstance(r["explanation"], list)
        self.assertTrue(len(r["explanation"]) > 0)

    def test_contributing_indicators_present(self):
        r = self.svc.assess(svi_score=60, overall_stress=60, overall_distress=58,
                            indicators=["fear", "helplessness", "threat-related context"],
                            confidence=0.85, immediate_safety=False)
        self.assertIsInstance(r["contributing_indicators"], list)


# ===========================================================================
# Recommendation service
# ===========================================================================

class TestRecommendationService(unittest.TestCase):
    def setUp(self):
        self.svc = RecommendationService()

    def test_low_recommendation(self):
        rec = self.svc.recommend(RiskLevel.LOW)
        self.assertIn("Low", rec)

    def test_moderate_recommendation(self):
        rec = self.svc.recommend(RiskLevel.MODERATE)
        self.assertIn("Moderate", rec)

    def test_high_recommendation(self):
        rec = self.svc.recommend(RiskLevel.HIGH)
        self.assertIn("High", rec)
        self.assertIn("priority human review", rec)

    def test_critical_recommendation(self):
        rec = self.svc.recommend(RiskLevel.CRITICAL)
        self.assertIn("Critical", rec)
        self.assertIn("handoff", rec)

    def test_immediate_safety_taken_seriously(self):
        rec = self.svc.recommend(RiskLevel.MODERATE, immediate_safety=True)
        self.assertIn("Immediate safety indicators detected", rec)


# ===========================================================================
# Utility helpers
# ===========================================================================

class TestHelpers(unittest.TestCase):
    def test_mean_int(self):
        self.assertEqual(_mean_int([10, 20, 30]), 20)
        self.assertEqual(_mean_int([]), 0)

    def test_mean_float(self):
        self.assertAlmostEqual(_mean_float([0.8, 0.9, 0.7]), 0.8)
        self.assertAlmostEqual(_mean_float([]), 0.0)


# ===========================================================================
# Upload validation
# ===========================================================================

class TestValidation(unittest.TestCase):
    def test_valid_extension(self):
        from app.schemas import is_allowed_extension
        self.assertTrue(is_allowed_extension("call.wav"))
        self.assertTrue(is_allowed_extension("call.MP3"))
        self.assertTrue(is_allowed_extension("call.m4a"))
        self.assertTrue(is_allowed_extension("call.ogg"))
        self.assertTrue(is_allowed_extension("call.webm"))

    def test_invalid_extension(self):
        from app.schemas import is_allowed_extension
        self.assertFalse(is_allowed_extension("call.txt"))
        self.assertFalse(is_allowed_extension("call.exe"))
        self.assertFalse(is_allowed_extension("call"))

    def test_validate_upload_rejects_missing(self):
        from app.utils.validation import validate_upload
        self.assertIsNotNone(validate_upload(None, 0, 50 * 1024 * 1024))
        self.assertIsNotNone(validate_upload("", 0, 50 * 1024 * 1024))

    def test_validate_upload_rejects_bad_format(self):
        from app.utils.validation import validate_upload
        err = validate_upload("call.txt", 1000, 50 * 1024 * 1024)
        self.assertIsNotNone(err)
        self.assertIn("Unsupported", err)

    def test_validate_upload_rejects_oversize(self):
        from app.utils.validation import validate_upload
        err = validate_upload("call.wav", 100 * 1024 * 1024, 50 * 1024 * 1024)
        self.assertIsNotNone(err)
        self.assertIn("too large", err)

    def test_validate_upload_accepts_valid(self):
        from app.utils.validation import validate_upload
        err = validate_upload("call.wav", 10 * 1024 * 1024, 50 * 1024 * 1024)
        self.assertIsNone(err)


# ===========================================================================
# Demo case fields match schema
# ===========================================================================

# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
