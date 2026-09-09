import React from 'react';

export default function Transcript({ segments, currentSegment, isPlaying, audioRef, onSegmentClick }) {
  const fmt = (s) => {
    if (!s || isNaN(s)) return '0:00';
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  return (
    <div className="transcript-list">
      {segments.length === 0 ? (
        <div className="text-xs text-dim" style={{ padding: 10 }}>No transcript segments available.</div>
      ) : (
        segments.map((seg, i) => {
          const isActive = i === currentSegment && isPlaying;
          return (
            <div
              key={i}
              className={`segment ${isActive ? 'active' : ''}`}
              onClick={() => {
                if (audioRef?.current) {
                  audioRef.currentTime = seg.start;
                  if (!isPlaying) {
                    audioRef.play().catch(() => {});
                  }
                }
                onSegmentClick(i);
              }}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSegmentClick(i); } }}
            >
              <div className="flex items-center justify-between">
                <span className="segment-time">{fmt(seg.start)} — {fmt(seg.end)}</span>
                <span className="emotion-tag" style={{ fontSize: 10 }}>{seg.emotion}</span>
              </div>
              <div className="segment-text">{seg.text}</div>
              <div className="segment-meta">
                <span className="score-badge score-low" style={{ fontSize: 10, padding: '1px 6px' }}>
                  S{seg.stress_score}
                </span>
                <span className="score-badge score-moderate" style={{ fontSize: 10, padding: '1px 6px', backgroundColor: seg.distress_score >= 50 ? 'rgba(248,113,113,0.15)' : seg.distress_score >= 25 ? 'rgba(251,191,36,0.15)' : '', color: seg.distress_score >= 50 ? '#F87171' : seg.distress_score >= 25 ? '#FBBF24' : '#34D399' }}>
                  D{seg.distress_score}
                </span>
                <span className="text-xs text-dim" style={{ fontWeight: 600 }}>SVI:</span>
                <span className="text-xs text-subtle">{seg.svi_score}</span>
                <span className="text-xs text-dim" style={{ fontWeight: 600, marginLeft: 4 }}>Risk:</span>
                <span className={`risk-badge risk-${seg.risk_level.toLowerCase()}`} style={{ fontSize: 10, padding: '1px 6px' }}>
                  {seg.risk_level}
                </span>
                <span className="text-xs text-dim" style={{ fontWeight: 600, marginLeft: 4 }}>C:</span>
                <span className="text-xs text-subtle">{Math.round(seg.confidence * 100)}%</span>
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}
