import React from 'react';

export default function Indicators({ indicators }) {
  if (!indicators || indicators.length === 0) {
    return <div className="text-xs text-dim">No indicators identified.</div>;
  }
  return (
    <div className="indicators">
      {indicators.map((ind, i) => (
        <span key={i} className="ind-pill">{ind}</span>
      ))}
    </div>
  );
}
