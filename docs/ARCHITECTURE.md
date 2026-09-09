# Architecture — TraumaSense

## Conceptual pipeline

```
CONSENTED CALL
    ↓
AUDIO INPUT
    ↓
AUDIO PROCESSING (validation, metadata, temporary storage)
    ↓
SPEECH-TO-TEXT / ASR (configurable provider)
    ↓
TIMESTAMPED TRANSCRIPT (segments with start/end/speaker/text)
    ↓
CONVERSATIONAL ANALYSIS (per-segment stress, distress, emotion, indicators, confidence)
    ↓
SVI / COMPOSITE ASSESSMENT (aggregation across segments)
    ↓
RISK ENGINE (level + score + explanation + contributing indicators)
    ↓
RECOMMENDATION (assistive action suggestion)
    ↓
DASHBOARD (progressive display, trend chart, indicators)
    ↓
FINAL REPORT (printable summary)
    ↓
TRAINED HUMAN REVIEW
```

## Component boundaries

### Frontend (React)

- `App.jsx` — main state machine (idle → consent → uploaded → analyzing → done)
- `components/` — presentational components, no business logic
- `services/api.js` — single API client, all calls go through here

State tracked in App:
`selectedFile`, `audioUrl`, `case`, `analysis`, `currentSegment`,
`isUploading`, `isAnalyzing`, `error`, `mode`, `isPlaying`, `progress`, `consentAck`

### Backend (FastAPI)

- `app/main.py` — FastAPI app, CORS, router mount
- `app/config.py` — environment + settings
- `app/schemas.py` — Pydantic models (source of truth for API contract)
- `app/api/routes.py` — endpoint implementations
- `app/services/` — business logic, one service per responsibility

### Services

| Service                | Responsibility                                      |
|------------------------|-----------------------------------------------------|
| `audio_service.py`     | validate, inspect, store temp, cleanup              |
| `speech_to_text.py`    | configurable STT provider, demo + real integration  |
| `analysis_service.py`  | prototype conversational analysis engine             |
| `svi_service.py`       | composite SVI calculation                           |
| `risk_service.py`      | risk level + explanation + contributing indicators   |
| `recommendation_service.py` | assistive recommendation mapping              |

### Data flow for `/api/analyze`

1. Receive multipart file
2. Validate (extension, size, empty)
3. Store temporarily (sanitized UUID filename)
4. Inspect metadata (duration where available)
5. Run STT → list of segments
6. For each segment: run analysis engine → scores + indicators
7. Compute SVI from all segments
8. Compute risk from SVI + safety + confidence
9. Build recommendation
10. Return `AnalysisResponse`
11. Clean up temp file

## Provider architecture

Both STT and AI analysis are behind provider abstractions:

```python
STT_PROVIDER=demo   # or "whisper", "vendor-asr", etc.
AI_PROVIDER=demo    # or "indicbert", "emotion-model", etc.
```

The frontend never knows which provider is active. Missing providers → automatic fallback to demo.

## Data model

### Case
```
CASE-26093-0001
├── file_name
├── duration_seconds
├── mode
├── transcript[]
│   └── segments[] (start, end, text, speaker, scores, emotion, indicators, risk)
├── overall_stress_score
├── overall_distress_score
├── overall_svi_score
├── overall_risk_score
├── overall_risk_level
├── overall_confidence
├── overall_indicators[]
├── risk_explanation[]
├── recommendation
└── disclaimer
```

### Persistence

For MVP: in-memory dictionary.  
For PS2 (longitudinal): SQLite schema in `backend/app/db/schema.sql`.

## Demo data design

The demo conversation is a 9-segment narrative with a realistic emotional arc:

| Segment | Emotion        | Stress | Distress |
|---------|----------------|--------|----------|
| 1       | Anxiety        | 34     | 30       |
| 2       | Distress       | 48     | 45       |
| 3       | Fear           | 62     | 60       |
| 4       | Fear           | 71     | 70       |
| 5       | Fear           | 76     | 77       |
| 6       | Distress       | 80     | 82       |
| 7       | Helplessness   | 73     | 68       |
| 8       | Uncertainty    | 66     | 60       |
| 9       | Calm           | 52     | 46       |

This demonstrates increasing concern followed by partial stabilisation — a realistic helpline interaction.

## Extensibility points

- Replace `SpeechToTextService._real_or_fallback` with a real ASR call.
- Replace `AnalysisService.analyze_segment` with a real NLP model.
- Add streaming via SSE in `routes.py` without changing the frontend contract.
- Swap the in-memory `_case_store` for a real database.
