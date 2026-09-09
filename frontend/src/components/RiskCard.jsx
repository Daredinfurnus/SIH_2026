import React from 'react';
import { AlertTriangle } from 'lucide-react';

export default function RiskCard({ level, score, confidence, safety }) {
  const levelClass = `risk-${level.toLowerCase()}`;

  return (
    <div className="card" style={{ borderColor: level === 'CRITICAL' ? 'rgba(239,68,68,0.25)' : level === 'HIGH' ? 'rgba(248,113,113,0.2)' : undefined }}>
      <div className="flex items-center justify-between gap-2 mb-2">
        <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--muted-foreground)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Assistive Risk</span>
        {safety && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: 10, color: '#FCA5A5', fontWeight: 600 }}>
            <AlertTriangle size={10} /> Safety flag
          </span>
        )}
      </div>
      <div className="flex items-center gap-3">
        <div className={`risk-badge ${levelClass}`} style={{ fontSize: 15 }}>{level}</div>
        <div style={{ fontSize: 24, fontWeight: 700, lineHeight: 1 }}>
          {score}<span style={{ fontSize: 12, fontWeight: 400, color: 'var(--muted-foreground)' }}>/100</span>
        </div>
      </div>
      <div className="flex items-center gap-2 mt-2 text-xs text-dim">
        <span>Confidence:</span>
        <span style={{ fontWeight: 600, color: 'var(--subtle)' }}>{Math.round(confidence * 100)}%</span>
        <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--dim)' }}>Human-in-the-loop</span>
      </div>
      <div className="text-xs text-dim mt-1" style={{ lineHeight: 1.4 }}>
        Prototype thresholds: LOW 0-24 · MODERATE 25-49 · HIGH 50-74 · CRITICAL 75-100
      </div>
    </div>
  );
}
