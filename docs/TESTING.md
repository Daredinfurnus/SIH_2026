# Testing — TraumaSense

## Backend tests

```bash
cd backend
python -m pytest tests/ -v
```

### What is covered

- **Health endpoint** — schema validity
- **Demo endpoint** — deterministic output, segment count, case ID format, risk level plausibility, indicators populated, disclaimer present, schema compliance
- **Analysis service** — empty text returns low scores, fear text increases stress, distress text increases distress, confidence in range, safety flag detection, emotion classification, indicators list
- **SVI service** — empty segments return 0, basic computation within 0–100, immediate safety escalation, breakdown keys present
- **Risk service** — LOW/MODERATE/HIGH/CRITICAL thresholds, immediate safety escalation, confidence adjustment, explanation present, contributing indicators present
- **Recommendation service** — all four levels return appropriate text, immediate safety taken seriously
- **Helpers** — mean of integers, mean of floats
- **Validation** — allowed extensions, rejected extensions, missing file, bad format, oversized file, valid upload accepted

### Expected result

All tests pass. The demo endpoint produces a valid `AnalysisResponse` with populated transcript, scores, risk, explanation, and recommendation.

---

## Frontend verification

### npm install

```bash
cd frontend
npm install
```

Should complete without errors.

### Development server

```bash
cd frontend
npm run dev
```

Open http://localhost:5173

### Production build

```bash
cd frontend
npm run build
```

Should complete without errors. Output in `frontend/dist/`.

---

## Manual acceptance test (judge flow)

1. Open http://localhost:5173
2. Click **Start Demo Mode**
3. Acknowledge the consent notice
4. Verify:
   - Case ID appears (CASE-26093-0001)
   - Stress score visible
   - Distress score visible
   - SVI visible
   - Risk level visible
   - Risk explanation visible
   - Indicators visible
   - Confidence visible
   - Trend chart renders
   - Recommendation visible
   - Final report tab accessible
   - Print button opens print dialog
5. Click **New Case** to reset
6. Try **Upload Call** with a valid audio file
7. Acknowledge consent
8. Click **Start Analysis**
9. Verify the same dashboard appears with uploaded-file values

### Expected result

The entire flow completes without crashing. No console errors in the browser. No Python stack traces in the API responses.

---

## CORS

The backend allows only:

- `http://localhost:5173`
- `http://127.0.0.1:5173`

No wildcard CORS.

---

## Error handling

Frontend handles:
- Invalid file format
- File too large
- Network failure
- Backend unavailable
- Malformed response
- Missing data

Backend handles:
- Missing file
- Invalid file
- Oversized file
- STT failure (falls back to demo where possible)
- Analysis failure
- Unexpected exceptions (returns clear error, never stack trace)

---

## Known limitations for testing

- The demo STT generates realistic transcripts but is not a real ASR engine.
- Audio duration detection depends on `wave` (stdlib) or `ffprobe`. If neither is available, duration falls back to a size-based estimate.
- In-memory case store is lost on restart (MVP).
