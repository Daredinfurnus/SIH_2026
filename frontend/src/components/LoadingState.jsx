import React from 'react';
import { RefreshCw, Activity } from 'lucide-react';

export default function LoadingState({ progress, message }) {
  return (
    <div className="loading-state">
      <div className="spinner" style={{ width: 40, height: 40, borderWidth: 4 }} />
      <div style={{ fontSize: 15, fontWeight: 600 }}>Analyzing conversation…</div>
      <div className="text-xs text-muted" style={{ maxWidth: 320, lineHeight: 1.5 }}>{message}</div>
      {typeof progress === 'number' && progress > 0 && (
        <div style={{ width: 200, marginTop: 8 }}>
          <div style={{ height: 4, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ width: `${Math.min(progress, 90)}%`, height: '100%', background: 'var(--accent)', borderRadius: 2, transition: 'width 0.3s' }} />
          </div>
          <div className="text-xs text-dim mt-1" style={{ textAlign: 'center' }}>{Math.min(progress, 90)}% complete</div>
        </div>
      )}
      <div className="text-xs text-dim mt-2" style={{ lineHeight: 1.4 }}>
        Processing audio · Generating transcript · Analyzing segments
      </div>
    </div>
  );
}
