-- SQLite schema for TraumaSense case persistence
-- Enables longitudinal case tracking across the case journey
-- Required for PS2 (dynamic monitoring + distress prediction)

BEGIN TRANSACTION;

-- Cases table: one row per analyzed helpline recording
CREATE TABLE IF NOT EXISTS cases (
    id              TEXT PRIMARY KEY,   -- CASE-26093-0001 style identifier
    created_at      TEXT NOT NULL,      -- ISO-8601 timestamp
    file_name       TEXT NOT NULL,      -- original uploaded filename (sanitized)
    file_size_bytes INTEGER NOT NULL,
    duration_seconds REAL NOT NULL,
    language        TEXT NOT NULL DEFAULT 'en',
    stt_provider    TEXT NOT NULL DEFAULT 'demo',
    ai_provider     TEXT NOT NULL DEFAULT 'demo',
    mode            TEXT NOT NULL DEFAULT 'demo',
    status          TEXT NOT NULL DEFAULT 'completed',  -- pending | processing | completed | dismissed
    caller_minutes  INTEGER             -- optional operator-entered minutes on call
);

-- Transcript segments: one row per timestamped conversational segment
CREATE TABLE IF NOT EXISTS transcript_segments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id         TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    segment_index   INTEGER NOT NULL,
    start_seconds   REAL NOT NULL,
    end_seconds     REAL NOT NULL,
    speaker         TEXT NOT NULL DEFAULT 'caller',
    text            TEXT NOT NULL,
    UNIQUE(case_id, segment_index)
);

-- Per-segment analysis results
CREATE TABLE IF NOT EXISTS segment_analysis (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id         TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    segment_index   INTEGER NOT NULL,
    stress_score    INTEGER NOT NULL CHECK(stress_score BETWEEN 0 AND 100),
    distress_score  INTEGER NOT NULL CHECK(distress_score BETWEEN 0 AND 100),
    emotion         TEXT NOT NULL,
    confidence      REAL NOT NULL CHECK(confidence BETWEEN 0.0 AND 1.0),
    svi_score       INTEGER NOT NULL CHECK(svi_score BETWEEN 0 AND 100),
    risk_level      TEXT NOT NULL CHECK(risk_level IN ('LOW','MODERATE','HIGH','CRITICAL')),
    risk_score      INTEGER NOT NULL CHECK(risk_score BETWEEN 0 AND 100),
    safety_flag     INTEGER NOT NULL DEFAULT 0 CHECK(safety_flag IN (0,1)),
    UNIQUE(case_id, segment_index)
);

-- Conversational indicators attached to segments (one-to-many)
CREATE TABLE IF NOT EXISTS segment_indicators (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id         TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    segment_index   INTEGER NOT NULL,
    indicator       TEXT NOT NULL,
    UNIQUE(case_id, segment_index, indicator)
);

-- Overall (case-level) aggregate analysis
CREATE TABLE IF NOT EXISTS overall_analysis (
    case_id             TEXT PRIMARY KEY REFERENCES cases(id) ON DELETE CASCADE,
    overall_stress_score   INTEGER NOT NULL CHECK(overall_stress_score BETWEEN 0 AND 100),
    overall_distress_score INTEGER NOT NULL CHECK(overall_distress_score BETWEEN 0 AND 100),
    overall_svi_score      INTEGER NOT NULL CHECK(overall_svi_score BETWEEN 0 AND 100),
    overall_risk_score     INTEGER NOT NULL CHECK(overall_risk_score BETWEEN 0 AND 100),
    overall_risk_level     TEXT NOT NULL CHECK(overall_risk_level IN ('LOW','MODERATE','HIGH','CRITICAL')),
    overall_confidence     REAL NOT NULL CHECK(overall_confidence BETWEEN 0.0 AND 1.0),
    recommendation         TEXT NOT NULL,
    risk_explanation       TEXT NOT NULL,  -- JSON array of strings
    overall_indicators     TEXT NOT NULL,  -- JSON array of strings
    immediate_safety       INTEGER NOT NULL DEFAULT 0 CHECK(immediate_safety IN (0,1)),
    analyzed_at            TEXT NOT NULL  -- ISO-8601
);

-- Case notes / longitudinal log entries (PS2: dynamic monitoring over time)
CREATE TABLE IF NOT EXISTS case_notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id     TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    author      TEXT NOT NULL DEFAULT 'system',
    note_type   TEXT NOT NULL DEFAULT 'analysis',  -- analysis | operator | followup
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_segments_case ON transcript_segments(case_id);
CREATE INDEX IF NOT EXISTS idx_analysis_case  ON segment_analysis(case_id);
CREATE INDEX IF NOT EXISTS idx_indicators_case ON segment_indicators(case_id);
CREATE INDEX IF NOT EXISTS idx_notes_case     ON case_notes(case_id);
CREATE INDEX IF NOT EXISTS idx_cases_created  ON cases(created_at);

COMMIT;
