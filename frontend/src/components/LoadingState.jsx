import React, { useEffect, useState } from 'react';
import { Sparkles, Activity, FileText, ScanLine, Brain } from 'lucide-react';

const STEPS = [
  { icon: Activity, label: 'Processing audio' },
  { icon: FileText, label: 'Generating transcript' },
  { icon: ScanLine, label: 'Analyzing segments' },
  { icon: Brain, label: 'Computing risk indicators' },
];

export default function LoadingState({ progress, message }) {
  const [step, setStep] = useState(0);
  const [pulse, setPulse] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => {
      setStep((s) => Math.min(s + 1, STEPS.length - 1));
    }, 1800);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const p = setInterval(() => setPulse((v) => !v), 1400);
    return () => clearInterval(p);
  }, []);

  const pct = Math.min(progress || 0, 92);

  return (
    <div className="loading-state">
      <div className="ls-backdrop" />

      <div className="ls-card">
        <div className="ls-header">
          <div className="ls-spinner-wrap">
            <div className={`ls-spinner ${pulse ? 'ls-spinner-pulse' : ''}`} />
            <div className="ls-spinner-ring" />
          </div>
          <div className="ls-title-wrap">
            <Sparkles size={15} className="ls-icon-top" />
            <div className="ls-title">Analyzing conversation</div>
            <div className="ls-sub">{message || 'Processing your audio file'}</div>
          </div>
        </div>

        <div className="ls-progress-wrap">
          <div className="ls-track">
            <div
              className="ls-fill"
              style={{
                width: `${pct}%`,
                background:
                  'linear-gradient(90deg, var(--accent-dim) 0%, var(--accent) 40%, var(--accent-2) 100%)',
              }}
            />
            <div className="ls-fill-glow" style={{ left: `${Math.min(pct, 92)}%` }} />
          </div>
          <div className="ls-pct">{pct}%</div>
        </div>

        <div className="ls-steps">
          {STEPS.map((s, i) => {
            const done = i < step;
            const active = i === step;
            return (
              <div
                key={s.label}
                className="ls-step"
                style={{
                  opacity: i <= step ? 1 : 0.35,
                  transform: i <= step ? 'translateY(0)' : 'translateY(4px)',
                  transition: 'all 0.35s ease',
                }}
              >
                <div
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: 10,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: done
                      ? 'rgba(108,144,195,0.15)'
                      : active
                      ? 'rgba(108,144,195,0.08)'
                      : 'rgba(138,168,200,0.04)',
                    border: `1px solid ${done ? 'rgba(108,144,195,0.3)' : active ? 'rgba(108,144,195,0.15)' : 'rgba(138,168,200,0.06)'}`,
                    color: done ? 'var(--accent-soft)' : active ? 'var(--accent)' : 'var(--fg-dim)',
                    transition: 'all 0.4s ease',
                    flexShrink: 0,
                  }}
                >
                  <s.icon size={16} strokeWidth={done ? 2.2 : 1.8} />
                </div>
                <span
                  className="ls-step-label"
                  style={{
                    color: i === step ? 'var(--accent-soft)' : i < step ? 'var(--fg-muted)' : 'var(--fg-dim)',
                    transition: 'color 0.3s',
                  }}
                >
                  {s.label}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
