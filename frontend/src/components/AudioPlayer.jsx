import React, { useState, useRef, useCallback } from 'react';
import { Play, Square, Clock, FileAudio } from 'lucide-react';

export default function AudioPlayer({
  url, name, size, duration, isPlaying, onPlayToggle,
  onTimeUpdate, onEnded, audioRef: externalRef,
}) {
  const innerRef = useRef(null);
  const audioRef = externalRef || innerRef;

  const fmt = (s) => {
    if (!s || isNaN(s)) return '0:00';
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  const fmtSize = (bytes) => {
    if (!bytes) return '—';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const togglePlay = useCallback(() => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play().catch(() => {});
    }
    onPlayToggle();
  }, [isPlaying, onPlayToggle]);

  return (
    <div className="card" style={{ marginTop: 12 }}>
      <div className="section-title"><FileAudio size={13} /> Audio Preview</div>
      <div className="audio-player">
        <button
          className="btn btn-secondary btn-sm"
          onClick={togglePlay}
          disabled={!url}
          style={{ minWidth: 36 }}
        >
          {isPlaying ? <Square size={14} fill="currentColor" /> : <Play size={14} />}
        </button>
        {url ? (
          <>
            <audio
              ref={audioRef}
              src={url}
              onTimeUpdate={(e) => onTimeUpdate(e.currentTarget.currentTime)}
              onEnded={onEnded}
              preload="metadata"
            />
            <span className="audio-info">
              <span style={{ fontWeight: 600 }}>{name}</span>
              <span style={{ margin: '0 6px' }}>·</span>
              {duration ? fmt(duration) : '—'}
              <span style={{ margin: '0 6px' }}>·</span>
              {fmtSize(size)}
            </span>
          </>
        ) : (
          <span className="text-xs text-dim">No audio loaded — upload a file to begin.</span>
        )}
      </div>
      {duration > 0 && (
        <div className="text-xs text-dim mt-2 flex items-center gap-2">
          <Clock size={11} />
          Duration: {fmt(duration)} · Ready for analysis
        </div>
      )}
    </div>
  );
}
