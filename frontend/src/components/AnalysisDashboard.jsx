import React, { useState, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Legend
} from 'recharts';
import { Activity, TrendingUp, Heart, Brain, BarChart3 } from 'lucide-react';
import ScoreCard from './ScoreCard';
import SVIIndicator from './SVIIndicator';
import RiskCard from './RiskCard';
import Transcript from './Transcript';
import Indicators from './Indicators';
import TrendChart from './TrendChart';
import RiskExplanation from './RiskExplanation';
import RecommendationCard from './RecommendationCard';

export default function AnalysisDashboard({ caseData, currentSegment, isPlaying }) {
  const [expandedSeg, setExpandedSeg] = useState(null);
  const audioRef = useRef(null);

  if (!caseData) return null;

  const segs = caseData.transcript;

  return (
    <div>
      {/* Top row: trend chart + assessment + recommendation (matches reference) */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
        {/* Trend chart — full width on narrow, left col on wide */}
        <div className="card" style={{ gridColumn: '1 / -1' }}>
          <div className="section-title"><TrendingUp size={13} /> Trend Analysis</div>
          <TrendChart segments={segs} />
        </div>

        {/* Why this assessment? */}
        <RiskExplanation
          level={caseData.overall_risk_level}
          score={caseData.overall_risk_score}
          confidence={caseData.overall_confidence}
          explanation={caseData.risk_explanation}
          safety={caseData.immediate_safety_indicators}
        />

        {/* Assistive recommendation */}
        <RecommendationCard recommendation={caseData.recommendation} />
      </div>

      {/* Score overview row */}
      <div className="grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10, marginBottom: 12 }}>
        <ScoreCard
          icon={<Activity size={16} />}
          label="Stress"
          value={caseData.overall_stress_score}
          color="stress"
          description="Prototype stress indicator"
        />
        <ScoreCard
          icon={<Heart size={16} />}
          label="Distress"
          value={caseData.overall_distress_score}
          color="distress"
          description="Prototype distress indicator"
        />
        <SVIIndicator
          value={caseData.overall_svi_score}
          description="Prototype composite assessment"
        />
        <RiskCard
          level={caseData.overall_risk_level}
          score={caseData.overall_risk_score}
          confidence={caseData.overall_confidence}
          safety={caseData.immediate_safety_indicators}
        />
      </div>

      {/* Calibration note */}
      <div className="card" style={{ background: 'rgba(75,110,245,0.04)', border: '1px solid rgba(75,110,245,0.1)', marginBottom: 12 }}>
        <div className="flex items-center gap-2 text-xs text-muted">
          <Brain size={12} />
          Prototype conversational analysis engine · Transparent scoring · Configurable provider architecture ·
          Designed for Indian-language and code-mixed conversations
        </div>
      </div>

      {/* Main grid: transcript + indicators */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 12, marginBottom: 12 }}>
        <div>
          <div className="section-title"><Activity size={13} /> Transcript & Segment Analysis</div>
          <Transcript
            segments={segs}
            currentSegment={currentSegment}
            isPlaying={isPlaying}
            audioRef={audioRef}
            onSegmentClick={setExpandedSeg}
          />
        </div>
        <div>
          <div className="section-title"><BarChart3 size={13} /> Conversational Indicators</div>
          <Indicators indicators={caseData.overall_indicators} />

          {expandedSeg !== null && segs[expandedSeg] && (
            <div className="card mt-3" style={{ background: 'rgba(255,255,255,0.02)' }}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-dim" style={{ fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Segment {expandedSeg + 1} detail
                </span>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => setExpandedSeg(null)}
                >
                  Close
                </button>
              </div>
              <div className="text-xs text-subtle mb-2" style={{ lineHeight: 1.5 }}>
                “{segs[expandedSeg].text}”
              </div>
              <div className="flex gap-2 flex-wrap">
                <span className="text-xs text-dim" style={{ fontWeight: 600 }}>Stress:</span>
                <span className="text-xs text-subtle">{segs[expandedSeg].stress_score}/100</span>
                <span className="text-xs text-dim" style={{ fontWeight: 600, marginLeft: 8 }}>Distress:</span>
                <span className="text-xs text-subtle">{segs[expandedSeg].distress_score}/100</span>
                <span className="text-xs text-dim" style={{ fontWeight: 600, marginLeft: 8 }}>Emotion:</span>
                <span className="emotion-tag">{segs[expandedSeg].emotion}</span>
                <span className="text-xs text-dim" style={{ fontWeight: 600, marginLeft: 8 }}>Confidence:</span>
                <span className="text-xs text-subtle">{Math.round(segs[expandedSeg].confidence * 100)}%</span>
              </div>
              {segs[expandedSeg].indicators.length > 0 && (
                <div className="indicators mt-2">
                  {segs[expandedSeg].indicators.map((ind, i) => (
                    <span key={i} className="ind-pill">{ind}</span>
                  ))}
                </div>
              )}
              {caseData.immediate_safety_indicators && segs[expandedSeg].indicators.some(i =>
                i.toLowerCase().includes('immediate') || i.toLowerCase().includes('safety')
              ) && (
                <div className="mt-2" style={{ padding: '6px 8px', background: 'rgba(239,68,68,0.1)', borderRadius: 4, fontSize: 11, color: '#FCA5A5' }}>
                  <AlertTriangle size={11} style={{ marginRight: 4 }} />
                  Immediate safety indicators detected — trained human review required.
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="disclaimer mt-3">
        <strong>Assistive, not diagnostic:</strong> {caseData.disclaimer}
      </div>
    </div>
  );
}

function Grid({ children, style }) {
  return <div style={{ ...style, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 }}>{children}</div>;
}
