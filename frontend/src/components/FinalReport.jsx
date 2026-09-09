import React from 'react';
import { FileText } from 'lucide-react';

export default function FinalReport({ caseData, onPrint }) {
  const fmt = (s) => {
    if (!s) return '—';
    const d = new Date(s);
    if (isNaN(d)) return s;
    return d.toLocaleString();
  };

  const PrinterIcon = ({ size }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="6 9 6 2 18 2 18 9"/>
      <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/>
      <rect x="6" y="14" width="12" height="8"/>
    </svg>
  );

  return (
    <div className="report">
      <div className="report-header">
        <h2>TraumaSense — Final Assessment Report</h2>
        <div className="report-sub">AI-assisted stress & trauma-related conversational assessment · Assistive decision-support only</div>
      </div>

      <div className="report-grid">
        <div className="report-item">
          <div className="label">Case ID</div>
          <div className="value">{caseData.case_id}</div>
        </div>
        <div className="report-item">
          <div className="label">Source File</div>
          <div className="value-sm text-subtle">{caseData.file_name}</div>
        </div>
        <div className="report-item">
          <div className="label">Duration</div>
          <div className="value">{caseData.duration_seconds}s</div>
        </div>
        <div className="report-item">
          <div className="label">Processing Mode</div>
          <div className="value-sm" style={{ color: caseData.mode === 'demo' ? '#8B9CF5' : '#34D399' }}>
            {caseData.mode === 'demo' ? 'Demo (offline, deterministic)' : 'Production mode'}
          </div>
        </div>
        <div className="report-item">
          <div className="label">Analysis Time</div>
          <div className="value-sm text-subtle">{fmt(caseData.analyzed_at)}</div>
        </div>
        <div className="report-item">
          <div className="label">Segments Analyzed</div>
          <div className="value">{caseData.transcript.length}</div>
        </div>
      </div>

      <div className="report-grid">
        <div className="report-item">
          <div className="label">Overall Stress Score</div>
          <div className="value">{caseData.overall_stress_score}/100</div>
        </div>
        <div className="report-item">
          <div className="label">Overall Distress Score</div>
          <div className="value">{caseData.overall_distress_score}/100</div>
        </div>
        <div className="report-item">
          <div className="label">SVI (Composite)</div>
          <div className="value">{caseData.overall_svi_score}/100</div>
        </div>
        <div className="report-item">
          <div className="label">Risk Score</div>
          <div className="value">{caseData.overall_risk_score}/100</div>
        </div>
        <div className="report-item">
          <div className="label">Risk Level</div>
          <div className={`risk-badge risk-${caseData.overall_risk_level.toLowerCase()}`} style={{ fontSize: 16, padding: '4px 16px', display: 'inline-block' }}>
            {caseData.overall_risk_level}
          </div>
        </div>
        <div className="report-item">
          <div className="label">Confidence</div>
          <div className="value">{Math.round(caseData.overall_confidence * 100)}%</div>
        </div>
      </div>

      <div className="report-section">
        <h4>Risk Explanation</h4>
        <ul>
          {caseData.risk_explanation.map((exp, i) => (
            <li key={i}>{exp}</li>
          ))}
        </ul>
      </div>

      <div className="report-section">
        <h4>Conversational Indicators</h4>
        <div className="indicators" style={{ marginTop: 4 }}>
          {caseData.overall_indicators.map((ind, i) => (
            <span key={i} className="ind-pill">{ind}</span>
          ))}
        </div>
      </div>

      <div className="report-section">
        <h4>Recommendation</h4>
        <p style={{ fontSize: 13, color: 'var(--subtle)', lineHeight: 1.6 }}>{caseData.recommendation}</p>
      </div>

      <div className="report-section">
        <h4>Transcript Summary</h4>
        <p style={{ fontSize: 12, color: 'var(--subtle)', lineHeight: 1.5 }}>
          The conversation was analyzed across {caseData.transcript.length} timestamped segments.
          Each segment received individual stress, distress, emotion, and risk analysis.
          See the Dashboard for per-segment detail.
        </p>
      </div>

      <div className="report-section">
        <h4>Disclaimer</h4>
        <div className="disclaimer" style={{ fontSize: 12 }}>
          <strong>Assistive, not diagnostic.</strong> {caseData.disclaimer}
          <br /><br />
          AI output should be interpreted together with human judgement and available case context.
          This report is a prototype analysis output and does not constitute a clinical diagnosis or legal determination.
        </div>
      </div>

      <div className="flex items-center justify-end gap-2 mt-4 no-print">
        <button className="btn btn-secondary" onClick={onPrint}>
          <PrinterIcon size={13} /> Print / Save as PDF
        </button>
      </div>
    </div>
  );
}
