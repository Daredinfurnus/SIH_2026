import React from 'react';
import { Shield, AlertTriangle } from 'lucide-react';

export default function ConsentNotice({ acknowledged, onAcknowledge, onBack }) {
  return (
    <div>
      <div className="consent-banner">
        <div className="consent-title">
          <Shield size={12} style={{ marginRight: 4 }} /> Consent & Privacy Notice
        </div>
        <div className="consent-text">
          <strong>Please analyze only a recording for which appropriate consent has been obtained.</strong>
        </div>
        <div className="consent-text" style={{ marginBottom: 0 }}>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: 12, color: 'var(--subtle)', lineHeight: 1.6 }}>
            <li>• Audio and transcript data may contain sensitive information.</li>
            <li>• TraumaSense provides assistive indicators and does not provide a clinical diagnosis.</li>
            <li>• High-risk indicators require trained human review.</li>
            <li>• AI output should be interpreted together with human judgement and available case context.</li>
          </ul>
        </div>
        <div className="consent-actions">
          {!acknowledged ? (
            <button className="btn btn-primary btn-sm" onClick={onAcknowledge}>
              I acknowledge — proceed
            </button>
          ) : (
            <span className="text-xs text-green-400" style={{ color: '#34D399' }}>
              <CheckCircle2 size={12} style={{ marginRight: 4 }} /> Consent acknowledged
            </span>
          )}
          <button className="btn btn-secondary btn-sm" onClick={onBack}>
            <AlertTriangle size={12} style={{ marginRight: 4 }} /> Upload different file
          </button>
        </div>
      </div>
    </div>
  );
}
