import React from 'react';
import { Target } from 'lucide-react';

export default function SVIIndicator({ value, description }) {
  const level = value <= 24 ? 'LOW' : value <= 49 ? 'MODERATE' : value <= 74 ? 'HIGH' : 'CRITICAL';
  const levelClass = `risk-${level.toLowerCase()}`;

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-2">
        <Target size={16} color="#FBBF24" />
        <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--muted-foreground)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>SVI</span>
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, lineHeight: 1 }}>
        {value}<span style={{ fontSize: 14, fontWeight: 400, color: 'var(--muted-foreground)' }}>/100</span>
      </div>
      <div className={`risk-badge ${levelClass}`} style={{ marginTop: 6, alignSelf: 'flex-start' }}>
        {level} SVI
      </div>
      <div className="text-xs text-dim mt-2" style={{ lineHeight: 1.4 }}>
        {description}. Not a clinically validated index.
      </div>
    </div>
  );
}
