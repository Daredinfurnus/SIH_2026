# Next Steps — TraumaSense

## Immediate

1. **Run the full stack** — start backend with `uvicorn`, start frontend with `npm run dev`, click Demo Mode, verify the full flow.
2. **Add real STT** — implement `SpeechToTextService._real_or_fallback` with a real ASR provider (e.g. faster-whisper) behind the `STT_PROVIDER` setting.
3. **Add real NLP analysis** — replace `AnalysisService.analyze_segment` with a real model behind `AI_PROVIDER`.
4. **Add persistence** — implement the SQLite schema in `backend/app/db/schema.sql` and wire it into routes for case storage (needed for PS2 longitudinal monitoring).

## For PS2 (Dynamic Monitoring)

- Add a case memory layer that stores SVI trajectory across multiple calls for the same victim/complainant.
- Add follow-up comparison: is distress increasing or decreasing over time?
- Add case notes API for operator annotations.
- Add a counsellor-facing view for case history.

## For production

- Add authentication (JWT or session-based).
- Replace in-memory store with a real database.
- Add file encryption at rest for audio and transcripts.
- Add audit logging (who accessed what, when).
- Add rate limiting and abuse prevention.
- Harden CORS, add Content Security Policy.
- Add structured logging (without PII).
- Add request signing for telecom integration.
- Add WebSocket/SSE for real-time streaming analysis.

## For multilingual

- Add language selection in the upload UI.
- Wire `language` param through STT service.
- Add Indic-language ASR provider.
- Add IndicNLP/IndicBERT-based text analysis.
- Display "Designed for Indian-language and code-mixed conversations" as a capability, not a claim.

## What NOT to add now

- Telecom integration
- Real NHAA infrastructure
- Camera emotion detection
- Wearable integration
- Mobile app
- Complex authentication
- Production database (use SQLite for PS2)
- Kubernetes, Kafka, Redis, microservices
- Custom foundation model
- Model training pipeline
- Autonomous emergency dispatch
- Autonomous medical/legal/police decisions

**STAY WITH THE MVP.** The architecture is designed to make these additions possible later without rewriting the app.
