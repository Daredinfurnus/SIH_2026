import React from 'react';

export default function RecommendationCard({ recommendation }) {
  return (
    <div className="recommendation-box">
      <div className="recommendation-title">Assistive Recommendation</div>
      <div className="recommendation-text">{recommendation}</div>
      <div className="text-xs text-dim mt-3" style={{ lineHeight: 1.4 }}>
        These are assistive suggestions only. The system does not independently execute any action.
        Trained human operators retain decision authority.
      </div>
    </div>
  );
}
