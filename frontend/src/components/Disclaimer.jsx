import React from 'react';
import { Shield, AlertTriangle } from 'lucide-react';

export default function Disclaimer() {
  return (
    <div style={{ marginTop: 16 }}>
      <div className="card" style={{ background: 'rgba(107,29,42,0.06)', border: '1px solid rgba(107,29,42,0.15)' }}>
        <div className="flex items-center gap-2 mb-2">
          <Shield size={14} color="#FCA5A5" />
          <span style={{ fontSize: 12, fontWeight: 700, color: '#FCA5A5', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Responsible AI Notice</span>
        </div>
        <div style={{ fontSize: 12, color: 'var(--subtle)', lineHeight: 1.6 }}>
          <p style={{ marginBottom: 6 }}>
            TraumaSense provides <strong>assistive conversational risk indicators</strong>. It is <strong>not a clinical diagnostic system</strong>.
            High-risk indicators require trained human review.
          </p>
          <p style={{ marginBottom: 6 }}>
            AI output should be interpreted together with human judgement and available case context.
            Prototype thresholds are demonstration values, not clinical thresholds.
          </p>
          <p>
            <AlertTriangle size={12} style={{ marginRight: 4, verticalAlign: 'middle', color: '#F87171' }} />
            The system does not replace counsellors, psychologists, or trained professionals.
            Human-in-the-loop: trained humans retain decision authority at all times.
          </p>
        </div>
      </div>

      <div className="flex items-center gap-4 mt-3 text-xs text-dim flex-wrap">
        <span className="flex items-center gap-1"><Shield size={11} /> Privacy-first · local/temp processing</span>
        <span className="flex items-center gap-1"><UserCheck size={11} /> Human-in-the-loop</span>
        <span className="flex items-center gap-1"><Brain size={11} /> Prototype analysis engine</span>
        <span className="flex items-center gap-1"><WifiOff size={11} /> Demo mode works offline</span>
      </div>
    </div>
  );
}

function Brain({ size }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-5 2.5 2.5 2.5 0 0 1 0-5 2.5 2.5 0 0 1 2.5-2.5z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 5 2.5 2.5 2.5 0 0 0 0-5 2.5 2.5 0 0 0-2.5-2.5z"/></svg>;
}
function WifiOff({ size }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="1" y1="1" x2="23" y2="23"/><path d="M16.72 11.06A10.92 10.92 0 0 1 19 12.55"/><path d="M5 12.55a10.92 10.92 0 0 1 5.17-2.39"/><path d="M10.71 5.05A16 16 0 0 1 22.58 9"/><path d="M1.42 9a15.91 15.91 0 0 1 4.68-1.68"/><path d="M8.59 15.05a15.56 15.56 0 0 1 0 0"/></svg>;
}
function UserCheck({ size }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>;
}
