import React from 'react';

/**
 * Pure-client-side TraumaScene transcript formatter.
 * Converts raw [Voice 1] / [Voice 2] segments into one flowing paragraph
 * that alternates speakers, merging consecutive same-speaker turns.
 *
 * Matches the SIH 26093 / TraumaSense design system.
 */

const LABEL_MAP = {
  'voice 1': 'Voice 1 (NHAA Official)',
  'voice 2': 'Voice 2 (Victim)',
};

function resolveLabel(raw, prevKey) {
  const key = raw.trim().toLowerCase();
  if (key in LABEL_MAP) return LABEL_MAP[key];
  if (!prevKey) return 'Voice 1 (NHAA Official) [assumed]';
  if (prevKey.startsWith('Voice 1')) return 'Voice 2 (Victim) [assumed]';
  return 'Voice 1 (NHAA Official) [assumed]';
}

function parseSegment(text) {
  const t = text.trim();
  if (!t) return null;
  let m = t.match(/^\[?\s*(Voice\s*\d+)\s*\]?\s*:?\s*(.*)$/i);
  if (m) {
    return { label: resolveLabel(m[1], null), body: m[2].trim() };
  }
  m = t.match(/^(Voice\s*\d+)\s*:?\s*(.*)$/i);
  if (m) {
    return { label: resolveLabel(m[1], null), body: m[2].trim() };
  }
  return { label: resolveLabel('', null), body: t };
}

export function formatTranscript(segments) {
  const blocks = []; // { label, text }
  let prevKey = null;

  for (const raw of segments) {
    const parsed = parseSegment(raw);
    if (!parsed || !parsed.body) continue;

    let label = parsed.label;
    const speakerKey = label.split(' (')[0];

    if (blocks.length > 0 && blocks[blocks.length - 1].label.startsWith(speakerKey)) {
      blocks[blocks.length - 1].text += ' ' + parsed.body;
    } else {
      blocks.push({ label, text: parsed.body });
    }
    prevKey = speakerKey;
  }

  return blocks.map(b => `${b.label}: ${b.text}`).join(' ');
}

export function countBlocks(formatted) {
  return {
    voice1: (formatted.match(/Voice 1 \(/g) || []).length,
    voice2: (formatted.match(/Voice 2 \(/g) || []).length,
    total: 0,
  };
}

export default function TranscriptFormatter({ segments, formatted, onFormat, onClear, onError, counts }) {
  return (
    <div className="formatter-panel">
      <div className="section-title">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
          <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
          <circle cx="9" cy="7" r="4"/>
          <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
          <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
        </svg>
        Transcript Formatter
      </div>

      <div className="formatter-layout">
        <div className="formatter-input-zone">
          <label className="formatter-label">
            Raw segments <span className="text-dim text-xs">(one per line)</span>
          </label>
          <textarea
            className="formatter-textarea"
            placeholder="Paste raw segments, one per line:&#10;&#10;[Voice 1] Hello, this is the NHAA official.&#10;[Voice 2] I don't know why this happened.&#10;[Voice 1] I understand. Tell me more."
            value={segments}
            onChange={e => onFormat?.({ segments: e.target.value, formatted: null })}
            spellCheck={false}
          />
          <div className="formatter-hint">
            Accepted: <code>[Voice 1] text</code> · <code>[Voice 2] text</code> · <code>Voice 1: text</code> · <code>Voice 2: text</code><br />
            Unlabeled lines are guessed from context and marked <code>[assumed]</code>. Consecutive same-speaker lines merge into one turn.
          </div>
        </div>

        <div className="formatter-output-zone">
          <label className="formatter-label">
            Formatted paragraph <span className="text-dim text-xs">— one paragraph, alternating speakers</span>
          </label>
          <div className="formatter-output">
            {formatted ? (
              <p className="formatter-paragraph">{formatted}</p>
            ) : (
              <p className="formatter-placeholder">Your single-paragraph transcript will appear here.</p>
            )}
          </div>
          {counts && (
            <div className="formatter-stats">
              <span className="stat-chip stat-v1">
                <span className="stat-dot" />
                NHAA Official: <strong>{counts.voice1}</strong> block{counts.voice1 !== 1 ? 's' : ''}
              </span>
              <span className="stat-chip stat-v2">
                <span className="stat-dot" />
                Victim: <strong>{counts.voice2}</strong> block{counts.voice2 !== 1 ? 's' : ''}
              </span>
            </div>
          )}
          <div className="formatter-actions">
            <button className="btn btn-ghost btn-sm" onClick={onClear}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>
              Clear
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => onFormat?.({ segments: EXAMPLE_SEGMENTS, formatted: null })}>
              Load example
            </button>
            <button className="btn btn-secondary btn-sm" onClick={() => navigator.clipboard.writeText(formatted || '')} disabled={!formatted}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
              Copy
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export const EXAMPLE_SEGMENTS = [
  '[Voice 1] Hello, my name is the NHAA official assigned to your case.',
  '[Voice 2] I don\'t know why this is happening to me.',
  '[Voice 1] I understand this is difficult. Let me ask you a few questions.',
  '[Voice 2] Okay, I\'ll try to answer.',
  '[Voice 1] When did you first notice the harassment?',
  '[Voice 2] It started last monsoon season, around July.',
  '[Voice 2] The incidents kept happening almost every week after that.',
  '[Voice 1] Thank you for sharing that. Let me note the timeline.',
].join('\n');
