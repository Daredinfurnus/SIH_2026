import React from 'react';
import { Upload, FileAudio, Download, X } from 'lucide-react';

export default function UploadPanel({ file, name, size, onReset }) {
  const fmtSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const ext = name?.split('.').pop()?.toUpperCase() || '—';

  return (
    <div className="card">
      <div className="section-title"><Upload size={13} /> Audio Source</div>
      <div className="flex items-center gap-3">
        <div style={{ width: 42, height: 42, borderRadius: 6, background: 'rgba(75,110,245,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
          <FileAudio size={20} color="#8B9CF5" />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="flex items-center gap-2 mb-1">
            <span style={{ fontSize: 13, fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {name || 'No file selected'}
            </span>
            <span className="text-xs text-dim">{ext}</span>
          </div>
          <div className="text-xs text-muted">
            {file ? fmtSize(size) : 'Select a consented helpline recording'}
          </div>
        </div>
        {file && (
          <button className="btn btn-secondary btn-sm" onClick={onReset}>
            <X size={13} />
          </button>
        )}
      </div>
    </div>
  );
}
