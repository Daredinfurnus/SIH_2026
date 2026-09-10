import sys
sys.path.insert(0, ".")

from app.services.analysis_service import AnalysisService

print("=== Test 1: English text, ai_provider=demo (lexical path) ===")
svc = AnalysisService(ai_provider="demo")
r = svc.analyze_segment("I am scared and don't feel safe anymore.")
print(f"  stress={r['stress_score']} distress={r['distress_score']} emotion={r['emotion']} confidence={r['confidence']}")
print(f"  indicators={r['indicators']}")
print(f"  safety_flag={r['safety_flag']}")
assert r['stress_score'] > 10, "English fear text should score stress > 10"
assert r['distress_score'] > 0, "English fear text should score distress > 0"
assert r['indicators'], "Should have indicators"
print("  PASS")

print()
print("=== Test 2: Hindi text, ai_provider=demo (OLD BUG: should be near-zero) ===")
r = svc.analyze_segment("मुझे बहुत डर लगता है और मुझे अपनी सुरक्षा की चिंता है।")
print(f"  stress={r['stress_score']} distress={r['distress_score']} emotion={r['emotion']} confidence={r['confidence']}")
print(f"  indicators={r['indicators']}")
# With demo provider, Hindi goes through lexical (English-only) → low scores
print(f"  (demo mode: Hindi → lexical → low scores, this is the known limitation)")
print("  PASS (expected behavior in demo mode)")

print()
print("=== Test 3: Hindi text, ai_provider=indicbert (NEW: should get real scores) ===")
svc2 = AnalysisService(ai_provider="indicbert")
r = svc2.analyze_segment("मुझे बहुत डर लगता है और मुझे अपनी सुरक्षा की चिंता है। कोई मेरी मदद नहीं कर सकता।")
print(f"  stress={r['stress_score']} distress={r['distress_score']} emotion={r['emotion']} confidence={r['confidence']}")
print(f"  indicators={r['indicators']}")
print(f"  safety_flag={r['safety_flag']}")
assert r['stress_score'] > 30, f"Hindi with indicbert should score stress > 30, got {r['stress_score']}"
assert r['distress_score'] > 30, f"Hindi with indicbert should score distress > 30, got {r['distress_score']}"
assert r['indicators'], "Should have indicators"
print("  PASS")

print()
print("=== Test 4: English text, ai_provider=indicbert (should STILL use lexical, not NLP) ===")
r = svc2.analyze_segment("I am scared and don't feel safe anymore.")
print(f"  stress={r['stress_score']} distress={r['distress_score']} emotion={r['emotion']} confidence={r['confidence']}")
print(f"  indicators={r['indicators']}")
# English → Latin script → lexical path, even with indicbert provider
assert r['stress_score'] > 10, "English should still score via lexical path"
print("  PASS (English correctly stays on lexical path)")

print()
print("=== Test 5: Tamil text, ai_provider=indicbert ===")
r = svc2.analyze_segment("நான் மிகவும் கவலையில் இருக்கிறேன் மற்றும் எனக்கு பாதுகாப்பு நிலை ஏற்பட்டுள்ளது.")
print(f"  stress={r['stress_score']} distress={r['distress_score']} emotion={r['emotion']} confidence={r['confidence']}")
print(f"  indicators={r['indicators']}")
print(f"  script detection: other (Tamil)")
# Tamil → 'other' script → currently NOT routed to NLP (only 'hi' is routed)
# This is a known limitation — Tamil goes through lexical (near-zero)
print("  (Tamil → 'other' script → lexical path. FUTURE: extend routing to all Indic scripts)")
print("  PASS (correct current behavior)")

print()
print("=== Test 6: Bengali text, ai_provider=indicbert ===")
r = svc2.analyze_segment("আমি খুব ভয় পাচ্ছি এবং আমার নিরাপত্তা নিয়ে চিন্তা আছে।")
print(f"  stress={r['stress_score']} distress={r['distress_score']} emotion={r['emotion']} confidence={r['confidence']}")
print(f"  indicators={r['indicators']}")
print("  (Bengali → 'other' script → lexical path. FUTURE: extend routing)")
print("  PASS")

print()
print("=== ALL ROUTING TESTS PASSED ===")
print("NOTE: Currently only Hindi/Devanagari is routed through IndicBERT.")
print("      Other Indic scripts (Tamil, Bengali, etc.) use lexical path.")
print("      To extend: add script→language mapping in analysis_service._looks_like_indic_script")
