import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export default function ErrorState({ message, onRetry, onReset }) {
  return (
    <div className="error-state">
      <AlertTriangle className="error-icon" />
      <div className="error-title">Something went wrong</div>
      <div className="error-detail">{message || 'An unexpected error occurred.'}</div>
      <div className="flex gap-2" style={{ marginTop: 4 }}>
        {onRetry && (
          <button className="btn btn-primary" onClick={onRetry}>
            <RefreshCw size={13} /> Try again
          </button>
        )}
        {onReset && (
          <button className="btn btn-secondary" onClick={onReset}>
            Go back
          </button>
        )}
      </div>
    </div>
  );
}
