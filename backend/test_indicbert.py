import sys
sys.path.insert(0, ".")

from app.services.nlp_service import get_nlp_service

print("=== Loading IndicBERT v2 ===")
nlp = get_nlp_service()
loaded = nlp.ensure_loaded()
print(f"Model loaded: {loaded}")
if not loaded:
    print(f"ERROR: {nlp._load_error}")
    sys.exit(1)
print(f"Embedding dim: {nlp._embedding_dim}")

print()
print("=== Script detection ===")
tests = [
    ("Hello I need help", "en"),
    ("मुझे बहुत डर लगता है", "hi"),
    ("நான் மிகவும் கவலையில் இருக்கிறேன்", "other"),
    ("I feel scared", "en"),
    ("আমি খুব ভয় পাচ্ছি", "other"),
]
for text, expected in tests:
    got = nlp.detect_language(text)
    status = "OK" if got == expected else "FAIL"
    print(f"  {status} expected={expected:5s} got={got:5s}  | {text[:40]}")

print()
print("=== Hindi text through NLP ===")
hindi = "मुझे बहुत डर लगता है और मुझे अपनी सुरक्षा की चिंता है। कोई मेरी मदद नहीं कर सकता।"
scores = nlp.score_indicators(hindi)
for k, v in sorted(scores.items(), key=lambda x: -x[1]):
    bar = chr(0x2588) * int(v * 40)
    print(f"  {k:20s} {v:.3f}  {bar}")
print(f"Derived stress:    {nlp.derive_stress(scores)}")
print(f"Derived distress:  {nlp.derive_distress(scores)}")
print(f"Derived emotion:   {nlp.derive_emotion(scores)}")
print(f"Derived confidence:{nlp.derive_confidence(scores)}")
print(f"Derived indicators:{nlp.derive_indicators(scores)}")

print()
print("=== English text (lexical path, no NLP routing) ===")
eng = "I am scared and do not feel safe anymore."
scores_en = nlp.score_indicators(eng)
for k, v in sorted(scores_en.items(), key=lambda x: -x[1]):
    print(f"  {k:20s} {v:.3f}")
print("(Latin script — IndicBERT not used, lexical path takes over)")

print()
print("DONE")
