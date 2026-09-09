import React from 'react';
import { TrendingUp, AlertTriangle } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Legend, ZAxis
} from 'recharts';

export default function TrendChart({ segments }) {
  if (!segments || segments.length === 0) {
    return <div className="text-xs text-dim" style={{ padding: 20 }}>No trend data available.</div>;
  }

  const data = segments.map((seg, i) => ({
    segment: i + 1,
    stress: seg.stress_score,
    distress: seg.distress_score,
    svi: seg.svi_score,
    risk: seg.risk_level === 'CRITICAL' ? 100 : seg.risk_level === 'HIGH' ? 75 : seg.risk_level === 'MODERATE' ? 50 : 25,
  }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload) return null;
    return (
      <div className="card" style={{ padding: '8px 10px', marginTop: 8, pointerEvents: 'none' }}>
        <div className="text-xs font-bold text-dim mb-1">Segment {label}</div>
        {payload.map((p) => (
          <div key={p.name} className="text-xs" style={{ color: p.color, fontWeight: 600 }}>
            {p.name}: {p.value}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
          <XAxis
            dataKey="segment"
            stroke="rgba(255,255,255,0.15)"
            fontSize={10}
            tick={{ fill: 'var(--dim)', fontSize: 10 }}
            label={{ value: 'Segment', position: 'bottom', fill: 'var(--dim)', fontSize: 10 }}
          />
          <YAxis
            domain={[0, 100]}
            stroke="rgba(255,255,255,0.15)"
            fontSize={10}
            tick={{ fill: 'var(--dim)', fontSize: 10 }}
            label={{ value: 'Score', angle: -90, position: 'insideLeft', fill: 'var(--dim)', fontSize: 10 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 10, paddingTop: 4 }}
            formatter={(value) => <span style={{ color: 'var(--subtle)', textTransform: 'capitalize' }}>{value}</span>}
          />
          <ReferenceLine y={50} stroke="rgba(248,113,113,0.2)" strokeDasharray="5 5" />
          <ReferenceLine y={75} stroke="rgba(239,68,68,0.2)" strokeDasharray="5 5" />
          <Line
            type="monotone"
            dataKey="stress"
            stroke="#F87171"
            strokeWidth={2}
            dot={{ fill: '#F87171', strokeWidth: 0, r: 3 }}
            activeDot={{ r: 5, fill: '#F87171' }}
            name="Stress"
          />
          <Line
            type="monotone"
            dataKey="distress"
            stroke="#FBBF24"
            strokeWidth={2}
            dot={{ fill: '#FBBF24', strokeWidth: 0, r: 3 }}
            activeDot={{ r: 5, fill: '#FBBF24' }}
            name="Distress"
          />
          <Line
            type="monotone"
            dataKey="svi"
            stroke="#8B9CF5"
            strokeWidth={2}
            dot={{ fill: '#8B9CF5', strokeWidth: 0, r: 3 }}
            activeDot={{ r: 5, fill: '#8B9CF5' }}
            name="SVI"
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="flex items-center gap-3 mt-2 text-xs text-dim flex-wrap">
        <span><span style={{ display: 'inline-block', width: 10, height: 3, background: '#F87171', borderRadius: 1, marginRight: 4, verticalAlign: 'middle' }}></span> Stress</span>
        <span><span style={{ display: 'inline-block', width: 10, height: 3, background: '#FBBF24', borderRadius: 1, marginRight: 4, verticalAlign: 'middle' }}></span> Distress</span>
        <span><span style={{ display: 'inline-block', width: 10, height: 3, background: '#8B9CF5', borderRadius: 1, marginRight: 4, verticalAlign: 'middle' }}></span> SVI</span>
        <span style={{ marginLeft: 'auto' }}>Dashed lines: threshold markers (50 / 75)</span>
      </div>
    </div>
  );
}

function AlertTriangleIcon({ size }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>;
}
