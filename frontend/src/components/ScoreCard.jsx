import React from 'react';

export default function ScoreCard({ icon, label, value, color, description }) {
  const colorClass = `score-${color}`;
  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--muted-foreground)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{label}</span>
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, lineHeight: 1 }}>
        {value}<span style={{ fontSize: 14, fontWeight: 400, color: 'var(--muted-foreground)' }}>/100</span>
      </div>
      <div className={`score-badge ${colorClass}`} style={{ marginTop: 6, alignSelf: 'flex-start' }}>
        {color === 'stress' ? 'Stress signal' : color === 'distress' ? 'Distress signal' : `${value}`}
      </div>
      <div className="text-xs text-dim mt-2" style={{ lineHeight: 1.4 }}>{description}</div>
    </div>
  );
}
