# API Contract — TraumaSense

**SMART INDIA HACKATHON 2026 · PS 26093**

All endpoints are prefixed with `/api`.

---

## `GET /api/health`

Service health check.

**Response**

```json
{
  "status": "ok",
  "service": "traumasense-api",
  "mode": "demo"
}
```

---

## `GET /api/demo`

Returns a complete deterministic demo analysis case. Useful for judge demonstrations and for frontend development without uploading audio.

**Response shape**

```json
{
  "case_id": "CASE-26093-0001",
  "file_name": "demo_call.wav",
  "duration_seconds": 120.0,
  "transcript": [
    {
      "start": 0.0,
      "end": 18.0,
      "text": "Hello, I need some help...",
      "speaker": "caller",
      "stress_score": 34,
      "distress_score": 30,
      "emotion": "Anxiety",
      "confidence": 0.78,
      "indicators": ["anxiety", "uncertainty"],
      "svi_score": 32,
      "risk_level": "LOW",
      "risk_explanation": ["Low stress and distress indicators..."]
    }
  ],
  "overall_stress_score": 58,
  "overall_distress_score": 55,
  "overall_svi_score": 61,
  "overall_risk_score": 61,
  "overall_risk_level": "HIGH",
  "overall_confidence": 0.84,
  "overall_indicators": ["fear", "anxiety", "helplessness"],
  "risk_explanation": [
    "Elevated distress indicators across the conversation.",
    "Fear-related and threat-related context detected."
  ],
  "recommendation": "Professional routing and priority human review recommended.",
  "mode": "demo",
  "disclaimer": "Assistive risk indicator, not a clinical diagnosis...",
  "immediate_safety_indicators": false,
  "language": "en",
  "analyzed_at": "2026-09-09T12:00:00+00:00"
}
```

---

## `POST /api/analyze`

Upload an audio file and receive a full analysis.

**Request**

Content-Type: `multipart/form-data`

| Field | Type             | Description                    |
|-------|------------------|--------------------------------|
| file  | binary audio     | `.wav`, `.mp3`, `.m4a`, `.ogg`, `.webm` |

**Constraints**

- Maximum file size: 50 MB (configurable via `MAX_UPLOAD_SIZE_MB`)
- Filename is sanitized with a UUID prefix

**Success response**

Same shape as `GET /api/demo`, with real values derived from the uploaded file.

**Error responses**

| Status | error_code       | detail                                              |
|--------|------------------|-----------------------------------------------------|
| 400    | NO_FILE          | No file provided.                                   |
| 400    | VALIDATION_ERROR | Unsupported format / file too large / empty file.   |
| 400    | READ_ERROR       | Failed to read the uploaded file.                   |
| 500    | STT_EMPTY        | Speech-to-text returned no segments.                |

---

## `GET /api/cases/{case_id}`

Retrieve a previously analyzed case by ID (in-memory for MVP).

**Response**

Same shape as `GET /api/demo`, or 404 if not found.

---

## Data types

### `TranscriptSegment`

| Field             | Type      | Range / values                        |
|-------------------|-----------|---------------------------------------|
| start             | float     | ≥ 0 seconds                           |
| end               | float     | ≥ 0 seconds                           |
| text              | string    | non-empty                             |
| speaker           | string    | `caller` or `operator`                |
| stress_score      | integer   | 0–100                                 |
| distress_score    | integer   | 0–100                                 |
| emotion           | string    | Fear, Anxiety, Sadness, Anger, Uncertainty, Calm, Distress, Hopelessness, Helplessness, Threat, Mixed |
| confidence        | float     | 0.0–1.0                               |
| indicators        | string[]  | e.g. ["fear", "helplessness"]         |
| svi_score         | integer   | 0–100                                 |
| risk_level        | string    | LOW, MODERATE, HIGH, CRITICAL         |
| risk_explanation  | string[]  | human-readable reasons                |

### `AnalysisResponse`

| Field                     | Type      | Description                              |
|---------------------------|-----------|------------------------------------------|
| case_id                   | string    | e.g. CASE-26093-0001                    |
| file_name                 | string    | original filename                        |
| duration_seconds          | float     | audio duration                           |
| transcript                | object[]  | array of TranscriptSegment               |
| overall_stress_score      | integer   | 0–100                                    |
| overall_distress_score    | integer   | 0–100                                    |
| overall_svi_score         | integer   | 0–100                                    |
| overall_risk_score        | integer   | 0–100                                    |
| overall_risk_level        | string    | LOW / MODERATE / HIGH / CRITICAL         |
| overall_confidence        | float     | 0.0–1.0                                  |
| overall_indicators        | string[]  | aggregated indicators                    |
| risk_explanation          | string[]  | human-readable reasons                   |
| recommendation            | string    | assistive recommendation text            |
| mode                      | string    | "demo" or provider name                  |
| disclaimer                | string    | standard disclaimer                      |
| immediate_safety_indicators | boolean | whether immediate safety signals were detected |
| language                  | string    | language code (default "en")             |
| analyzed_at               | string    | ISO-8601 timestamp                       |

---

## Risk thresholds (prototype)

| Score   | Level      |
|---------|------------|
| 0–24    | LOW        |
| 25–49   | MODERATE   |
| 50–74   | HIGH       |
| 75–100  | CRITICAL   |

These are **prototype demonstration thresholds**, not clinical thresholds.

---

## Error format

```json
{
  "detail": "Human-readable error message.",
  "error_code": "OPTIONAL_CODE"
}
```
