import React, { useImperativeHandle, forwardRef } from 'react';
import { Play, Square, Clock, FileAudio } from 'lucide-react';

const AudioPlayer = forwardRef(({
  url, name, size, duration, isPlaying, onPlayToggle,
  onTimeUpdate, onEnded, audioRef
}, ref) => {
  useImperativeHandle(ref, () => ({
    get current() { return audioRef?.current; }
  }), [audioRef]);

  const fmt = (s) => {
    if (!s || isNaN(s)) return '0:00';
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  const fmtSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="card" style={{ marginTop: 12 }}>
      <div className="section-title"><FileAudio size={13} /> Audio Preview</div>
      <div className="audio-player">
        <button
          className="btn btn-secondary btn-sm"
          onClick={onPlayToggle}
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
              onTimeUpdate={onTimeUpdate}
              onEnded={onEnded}
              preload="metadata"
            />
            <span className="audio-info">
              <span style={{ fontWeight: 600 }}>{name}</span>
              <span style={{ margin: '0 6px' }}>·</span>
              {duration ? fmt(duration) : '—'}
              <span style={{ margin: '0 6px' }}>·</span>
              {size ? fmtSize(size) : ''}
            </span>
          </>
        ) : (
          <span className="text-xs text-dim">No audio loaded. Run the demo or upload a file.</span>
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
});

AudioPlayer.displayName = 'AudioPlayer';
export default AudioPlayer;
