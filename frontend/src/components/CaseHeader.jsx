import React from 'react';
import { RefreshCw, Activity } from 'lucide-react';

export default function CaseHeader({
  caseId, fileName, duration, mode,
  riskLevel, riskScore, confidence, onReset, onPrint
}) {
  const riskClass = `risk-${riskLevel.toLowerCase()}`;
  const stressColor = riskLevel === 'CRITICAL' ? 'risk-critical' : riskLevel === 'HIGH' ? 'risk-high' : riskLevel === 'MODERATE' ? 'risk-moderate' : 'risk-low';

  return (
    <div className="card" style={{ marginBottom: 12 }}>
      <div className="flex items-center justify-between mb-3">
        <div>
          <div style={{ fontSize: 13, color: 'var(--muted-foreground)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Case ID</div>
          <div style={{ fontSize: 18, fontWeight: 700 }}>{caseId}</div>
        </div>
        <div className="flex gap-2">
          <button className="btn btn-secondary btn-sm" onClick={onReset}>
            <RefreshCw size={13} /> New Case
          </button>
          <button className="btn btn-secondary btn-sm" onClick={onPrint}>
            <PrinterIcon size={13} /> Print Report
          </button>
        </div>
      </div>

      <div className="report-grid">
        <div className="report-item">
          <div className="label">File</div>
          <div className="value-sm text-subtle">{fileName}</div>
        </div>
        <div className="report-item">
          <div className="label">Duration</div>
          <div className="value-sm">{duration}s</div>
        </div>
        <div className="report-item">
          <div className="label">Processing Mode</div>
          <div className="value-sm" style={{ color: mode === 'demo' ? '#8B9CF5' : '#34D399' }}>{mode === 'demo' ? 'Demo (offline)' : 'Live'}</div>
        </div>
        <div className="report-item">
          <div className="label">Risk Level</div>
          <div className={`value risk-badge ${riskClass}`} style={{ fontSize: 16, padding: '4px 16px' }}>{riskLevel}</div>
        </div>
        <div className="report-item">
          <div className="label">Risk Score</div>
          <div className="value">{riskScore}/100</div>
        </div>
        <div className="report-item">
          <div className="label">Confidence</div>
          <div className="value">{Math.round(confidence * 100)}%</div>
        </div>
      </div>
    </div>
  );
}

const PrinterIcon = ({ size }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="6 9 6 2 18 2 18 9"/>
    <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/>
    <rect x="6" y="14" width="12" height="8"/>
  </svg>
);
