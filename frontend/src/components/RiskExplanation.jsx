import React from 'react';
import { AlertTriangle } from 'lucide-react';

export default function RiskExplanation({ level, score, confidence, explanation, safety }) {
  const levelClass = `risk-${level.toLowerCase()}`;

  return (
    <div className="card">
      <div className="section-title"><AlertTriangle size={13} /> Why this assessment?</div>

      <div className="flex items-center gap-3 mb-3">
        <div className={`risk-badge ${levelClass}`} style={{ fontSize: 14 }}>{level}</div>
        <div style={{ fontSize: 22, fontWeight: 700 }}>{score}<span style={{ fontSize: 12, fontWeight: 400, color: 'var(--muted-foreground)' }}>/100</span></div>
        <div className="text-xs text-dim" style={{ marginLeft: 'auto', textAlign: 'right' }}>
          Confidence<br />{Math.round(confidence * 100)}%
        </div>
      </div>

      <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
        {explanation.map((exp, i) => (
          <li key={i} className="mb-2" style={{ display: 'flex', gap: 6, alignItems: 'flex-start', fontSize: 12, color: 'var(--subtle)', lineHeight: 1.5 }}>
            <span style={{ color: 'var(--accent)', marginTop: 2, flexShrink: 0 }}>—</span>
            <span>{exp}</span>
          </li>
        ))}
      </ul>

      {safety && (
        <div className="mt-3" style={{ padding: '8px 10px', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.15)', borderRadius: 5, fontSize: 11 }}>
          <div className="flex items-center gap-2" style={{ color: '#FCA5A5', fontWeight: 600, marginBottom: 2 }}>
            <AlertTriangle size={12} /> Immediate Safety Indicators
          </div>
          <div style={{ color: 'var(--subtle)' }}>
            Language suggesting immediate threat or danger was detected. Trained human review is required —
            the system does not contact emergency services autonomously.
          </div>
        </div>
      )}

      <div className="disclaimer mt-2" style={{ fontSize: 10 }}>
        Prototype risk thresholds — not clinical thresholds.
      </div>
    </div>
  );
}
