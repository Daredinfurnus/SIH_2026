# TraumaSense — AI-assisted stress & trauma-related conversational assessment

**SMART INDIA HACKATHON 2026 · Problem Statement 26093**

Team: ESPADA-X · KCC Institute of Technology & Management

---

## Problem

SC/ST victims and complainants reaching NHAA 14566 carry trauma from caste-based violence, threats, displacement, and prolonged legal proceedings. The first contact is a mental-health moment hiding in plain sight — and the helpline has minutes, not sessions.

There is no standardised way to assess the psychological condition and vulnerability of a victim at the moment of first contact.

## Solution

TraumaSense is an AI-assisted conversational assessment module that:

- receives a **consented prerecorded helpline call**
- converts speech to timestamped text (ASR / speech-to-text)
- analyzes each segment for stress, distress, emotion, and contextual indicators
- computes a **Stress Vulnerability Index (SVI)** — a prototype composite score
- produces an explainable **assistive risk assessment**
- recommends priority routing for trained human review
- keeps a **human-in-the-loop** at every stage

**This is an assistive decision-support system, not a clinical diagnostic tool.**

## Architecture

```
CONSENTED CALL
    ↓
AUDIO INPUT
    ↓
AUDIO PROCESSING
    ↓
SPEECH-TO-TEXT / ASR
    ↓
TIMESTAMPED TRANSCRIPT
    ↓
CONVERSATIONAL ANALYSIS
    ↓
STRESS · DISTRESS · EMOTION · INDICATORS · CONFIDENCE
    ↓
SVI / COMPOSITE ASSESSMENT
    ↓
RISK ENGINE
    ↓
EXPLANATION + RECOMMENDATION
    ↓
DASHBOARD
    ↓
FINAL REPORT
    ↓
TRAINED HUMAN REVIEW
    ↓
APPROPRIATE ACTION
```

### Key principles

- Stress ≠ Risk. Risk is computed from stress + distress + context + safety indicators + confidence.
- Every analysis belongs to a CASE (e.g. CASE-26093-0001).
- All scores are explainable — the system explains *why* a risk level was assigned.
- Demo mode runs fully offline — no API keys, no internet, no external models.

## Tech Stack

| Layer   | Technology                        |
|---------|-----------------------------------|
| Frontend| React 18 + Vite + JavaScript      |
| Charts  | Recharts                          |
| Icons   | Lucide React                      |
| Styling | Plain CSS (no framework)          |
| Backend | Python + FastAPI + Pydantic       |
| Runtime | uvicorn                           |
| Storage | In-memory for MVP (SQLite ready)  |

## Setup

### Prerequisites

- Python 3.11+
- Node 22+ and npm 12+
- Approximately 2 minutes to install both sides

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The app opens at http://localhost:5173

## Environment Variables

### Backend (`backend/.env.example`)

```
STT_PROVIDER=demo
AI_PROVIDER=demo
MAX_UPLOAD_SIZE_MB=50
```

The prototype runs with these defaults — no .env file required.

### Frontend (`frontend/.env.example`)

```
VITE_API_BASE_URL=http://localhost:8000
```

## Demo Mode

Click **"Start Demo Mode"** on the home screen. The app loads a deterministic case with 9 transcript segments showing a realistic progression:

- Segment 1: moderate anxiety
- Segment 2–3: increasing fear and distress
- Segment 4–5: helplessness, threat-related context, high distress
- Segment 6–7: partial stabilisation

The entire analysis is pre-computed and returns instantly. Scores, indicators, risk, explanation, and recommendation are all populated.

The demo is **deterministic** — every run produces the same result.

## Real Mode (future)

The architecture supports swapping the demo provider for a real one by changing environment variables:

```
STT_PROVIDER=whisper
AI_PROVIDER=indicbert
```

The frontend never knows the difference. If a real provider is unavailable, the system falls back to demo mode automatically.

## API

### `GET /api/health`

```json
{ "status": "ok", "service": "traumasense-api", "mode": "demo" }
```

### `GET /api/demo`

Returns a complete deterministic demo case (see API_CONTRACT.md for the full shape).

### `POST /api/analyze`

Multipart upload of an audio file (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.webm`, max 50 MB).

Returns the full analysis response.

### `GET /api/cases/{case_id}`

Retrieve a previously analyzed case.

## Analysis Engine

The prototype conversational analysis engine is:

- **deterministic** — same input produces the same output
- **explainable** — every score is traceable to lexical signals
- **modular** — designed to be replaced by a real NLP/ML model later
- **labelled honestly** — explicitly called a *prototype* engine, not clinically validated AI

It uses transparent lexical patterns that map text to stress, distress, emotion, and indicators. It is NOT keyword detection presented as advanced AI.

## SVI (Stress Vulnerability Index)

A prototype composite score (0–100) fusing:

- stress signals (weight 0.30)
- distress signals (weight 0.35)
- safety indicators (weight 0.20)
- contextual indicators (weight 0.15)

SVI is **not a clinically validated index**. It is a demonstration composite indicator.

## Risk Engine

Prototype thresholds (NOT clinical thresholds):

| Score   | Level      |
|---------|------------|
| 0–24    | LOW        |
| 25–49   | MODERATE   |
| 50–74   | HIGH       |
| 75–100  | CRITICAL   |

Immediate safety indicators trigger conservative escalation.

## Explainability

Every risk level comes with an explanation generated from actual contributing signals:

- elevated distress indicators
- fear-related language
- helplessness indicators
- threat-related context
- confidence level
- immediate safety flag (if present)

The UI never shows only "Risk: HIGH" — it always explains why.

## Privacy

- Audio and transcript data are treated as sensitive.
- Files are stored temporarily and cleaned up after processing.
- No transcript logging, no audio logging.
- No API keys in source code.
- Safe filenames (UUID-based).
- File-size limits enforced.

## Limitations

- Prototype analysis engine, not clinically validated.
- Demo STT generates realistic transcripts; real STT integration is a future step.
- No real ASR, no real NLP model, no real emotion model in the MVP.
- No telecom integration, no NHAA infrastructure, no production database.
- No authentication (MVP).
- No multilingual ASR implemented yet (architecture is ready for it).

## Future Roadmap

- Real ASR provider (Whisper / Indic ASR)
- Real NLP model for text analysis
- Multilingual ASR (Hindi first, then other Indian languages)
- Emotion/context model
- Validated SVI model
- Streaming STT + WebSocket/SSE for real-time analysis
- Real-time telecom integration
- Secure database + authentication
- Counsellor interface
- Case memory / longitudinal monitoring (PS2)
- Production deployment hardening

## Team Roles

| Person | Area                              |
|--------|-----------------------------------|
| 1      | Frontend / UI / UX                |
| 2      | Backend / API / Integration       |
| 3      | Audio / STT                       |
| 4      | Analysis / SVI / Risk / Testing   |

## Demo Script (Judge Flow)

1. Open TraumaSense.
2. Explain: "We receive a consented helpline call."
3. Click **Start Demo Mode** (or upload a file).
4. Acknowledge the consent notice.
5. Watch the analysis complete instantly.
6. Show the transcript with per-segment scores.
7. Show stress rising.
8. Show distress rising.
9. Show SVI changing.
10. Show risk level and explanation.
11. Show indicators and confidence.
12. Show trend chart.
13. Show recommendation.
14. Open and print the final report.
15. Close with: "The system does not replace the counsellor. It helps the counsellor prioritize and understand."
