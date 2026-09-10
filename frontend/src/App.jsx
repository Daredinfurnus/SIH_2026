import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Monitor,
  Upload,
  Play,
  Square,
  AlertTriangle,
  FileAudio,
  Activity,
  Brain,
  CheckCircle2,
  Printer,
  RefreshCw,
  Shield,
  UserCheck,
  X,
} from 'lucide-react';
import { healthCheck, uploadAndAnalyze, getCase } from './services/api';
import Header from './components/Header';
import ConsentNotice from './components/ConsentNotice';
import UploadPanel from './components/UploadPanel';
import AudioPlayer from './components/AudioPlayer';
import CaseHeader from './components/CaseHeader';
import AnalysisDashboard from './components/AnalysisDashboard';
import FinalReport from './components/FinalReport';
import LoadingState from './components/LoadingState';
import ErrorState from './components/ErrorState';
import Disclaimer from './components/Disclaimer';

export default function App() {
  // ---- state ------------------------------------------------------------
  const [mode, setMode] = useState('idle');       // idle | consent | uploaded | analyzing | done
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioName, setAudioName] = useState('');
  const [audioSize, setAudioSize] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const [caseData, setCaseData] = useState(null);
  const [error, setError] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentSegment, setCurrentSegment] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [consentAck, setConsentAck] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard'); // dashboard | report
  const [backendOnline, setBackendOnline] = useState(true);
  const [progress, setProgress] = useState(0);

  const audioRef = useRef(null);
  const progressInterval = useRef(null);

  // ---- check backend on mount ------------------------------------------
  useEffect(() => {
    healthCheck()
      .then(() => setBackendOnline(true))
      .catch(() => setBackendOnline(false));
  }, []);

  // ---- audio time tracking ---------------------------------------------
  const handleTimeUpdate = useCallback((currentTime) => {
    if (!caseData) return;
    const t = currentTime;
    const segs = caseData.transcript;
    for (let i = 0; i < segs.length; i++) {
      if (t >= segs[i].start && t < segs[i].end) {
        if (currentSegment !== i) setCurrentSegment(i);
        break;
      }
    }
  }, [caseData, currentSegment]);

  const handleEnded = useCallback(() => {
    setIsPlaying(false);
    setCurrentSegment(0);
  }, []);

  // ---- file handling ---------------------------------------------------
  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    if (!f.name.match(/\.(wav|mp3|m4a|ogg|webm)$/i)) {
      setError('Unsupported format. Please use WAV, MP3, M4A, OGG, or WEBM.');
      return;
    }
    if (f.size > 50 * 1024 * 1024) {
      setError('File too large. Maximum 50 MB.');
      return;
    }
    if (f.size === 0) {
      setError('The file appears to be empty.');
      return;
    }
    setError(null);
    setFile(f);
    const url = URL.createObjectURL(f);
    setAudioUrl(url);
    setAudioName(f.name);
    setAudioSize(f.size);
    setMode('uploaded');
    setCaseData(null);
    setActiveTab('dashboard');
  };

  const resetAll = () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setFile(null);
    setAudioUrl(null);
    setAudioName('');
    setAudioSize(0);
    setAudioDuration(0);
    setCaseData(null);
    setError(null);
    setMode('idle');
    setCurrentSegment(0);
    setIsPlaying(false);
    setConsentAck(false);
    setActiveTab('dashboard');
    setProgress(0);
    if (progressInterval.current) clearInterval(progressInterval.current);
  };

  // ---- analyze uploaded file -----------------------------------------
  const startAnalysis = async () => {
    if (!consentAck) {
      setError('Please acknowledge the consent notice before analysis.');
      return;
    }
    setError(null);
    setIsAnalyzing(true);
    setMode('analyzing');
    setProgress(0);
    setCurrentSegment(0);
    try {
      const data = await uploadAndAnalyze(file);
      setAudioDuration(data.duration_seconds || 0);
      setCaseData(data);
      setMode('done');
    } catch (err) {
      setError(err.message || 'Analysis failed. Please try again.');
      setMode('uploaded');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ---- simulated progressive progress during analysis -----------------
  useEffect(() => {
    if (mode !== 'analyzing') {
      if (progressInterval.current) clearInterval(progressInterval.current);
      return;
    }
    setProgress(0);
    const iv = setInterval(() => {
      setProgress(p => {
        if (p >= 90) { clearInterval(iv); return 90; }
        return p + Math.floor(Math.random() * 8) + 3;
      });
    }, 400);
    progressInterval.current = iv;
    return () => clearInterval(iv);
  }, [mode]);

  // ---- print report ----------------------------------------------------
  const handlePrint = () => {
    if (!caseData) return;
    const win = window.open('', '_blank');
    if (!win) return;
    win.document.write(`
      <html><head><title>TraumaSense Report — ${caseData.case_id}</title>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 32px; color: #111; }
        h1 { font-size: 20px; margin-bottom: 2px; }
        .sub { color: #666; font-size: 12px; margin-bottom: 16px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; margin-bottom: 16px; }
        .item { background: #f5f5f5; border: 1px solid #ddd; border-radius: 6px; padding: 8px 10px; }
        .item .l { font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
        .item .v { font-size: 15px; font-weight: 700; margin-top: 2px; }
        .section { margin-top: 12px; padding-top: 10px; border-top: 1px solid #ddd; }
        .section h4 { font-size: 12px; color: #4B6EF5; margin-bottom: 4px; text-transform: uppercase; }
        .section li { font-size: 12px; color: #444; padding: 2px 0 2px 12px; position: relative; list-style: none; }
        .section li::before { content: '—'; position: absolute; left: 0; color: #999; }
        .disclaimer { margin-top: 14px; padding: 8px 10px; background: #fef2f2; border: 1px solid #fecaca; border-radius: 4px; font-size: 11px; color: #991b1b; }
        .risk-low { color: #065F46; }
        .risk-moderate { color: #92400E; }
        .risk-high { color: #991B1B; }
        .risk-critical { color: #7F1D1D; }
      </style></head><body>
        <h1>TraumaSense — Final Assessment Report</h1>
        <div class="sub">AI-assisted stress & trauma-related conversational assessment · Assistive decision-support only</div>
        <div class="grid">
          <div class="item"><div class="l">Case ID</div><div class="v">${caseData.case_id}</div></div>
          <div class="item"><div class="l">File</div><div class="v">${caseData.file_name}</div></div>
          <div class="item"><div class="l">Duration</div><div class="v">${caseData.duration_seconds}s</div></div>
          <div class="item"><div class="l">Processing Mode</div><div class="v">${caseData.mode}</div></div>
          <div class="item"><div class="l">Analysis Time</div><div class="v">${caseData.analyzed_at ? caseData.analyzed_at.slice(0,19).replace('T',' ') : '—'}</div></div>
        </div>
        <div class="grid">
          <div class="item"><div class="l">Stress Score</div><div class="v">${caseData.overall_stress_score}/100</div></div>
          <div class="item"><div class="l">Distress Score</div><div class="v">${caseData.overall_distress_score}/100</div></div>
          <div class="item"><div class="l">SVI (Composite)</div><div class="v">${caseData.overall_svi_score}/100</div></div>
          <div class="item"><div class="l">Risk Score</div><div class="v">${caseData.overall_risk_score}/100</div></div>
          <div class="item"><div class="l">Risk Level</div><div class="v risk-${caseData.overall_risk_level.toLowerCase()}">${caseData.overall_risk_level}</div></div>
          <div class="item"><div class="l">Confidence</div><div class="v">${Math.round(caseData.overall_confidence * 100)}%</div></div>
        </div>
        <div class="section">
          <h4>Risk Explanation</h4>
          <ul>${caseData.risk_explanation.map(e => `<li>${e}</li>`).join('')}</ul>
        </div>
        <div class="section">
          <h4>Indicators</h4>
          <div style="display:flex;flex-wrap:wrap;gap:4px;">
            ${caseData.overall_indicators.map(i => `<span style="padding:2px 8px;background:#f0f0f0;border-radius:8px;font-size:11px;">${i}</span>`).join('')}
          </div>
        </div>
        <div class="section">
          <h4>Recommendation</h4>
          <p style="font-size:13px;color:#333;line-height:1.5;">${caseData.recommendation}</p>
        </div>
        <div class="section">
          <h4>Transcript Summary (${caseData.transcript.length} segments)</h4>
          <p style="font-size:12px;color:#555;">The conversation was analyzed in ${caseData.transcript.length} timestamped segments. Scores and indicators are shown in the dashboard.</p>
        </div>
        <div class="disclaimer">
          <strong>Important:</strong> ${caseData.disclaimer}<br>
          AI output should be interpreted together with human judgement and available case context.
        </div>
      </body></html>
    `);
    win.document.close();
    win.focus();
    win.print();
  };

  // ---- render ----------------------------------------------------------
  return (
    <div className="app">
      <Header />

      <main className="main">
        {/* Error */}
        {error && (
          <div className="card" style={{ marginBottom: 12, background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.2)' }}>
            <div className="flex items-center gap-2">
              <AlertTriangle size={16} color="#F87171" />
              <span style={{ fontSize: 12, color: '#F87171' }}>{error}</span>
              <button className="btn btn-secondary btn-sm" onClick={() => setError(null)} style={{ marginLeft: 'auto' }}>Dismiss</button>
            </div>
          </div>
        )}

        {/* IDLE — Home */}
        {mode === 'idle' && (
          <>
            <div className="flex items-center gap-3 mb-4">
              <Brain size={28} color="#4B6EF5" />
              <div>
                <h2 style={{ fontSize: 18, fontWeight: 700 }}>TraumaSense</h2>
                <p className="text-muted text-sm" style={{ marginTop: 2 }}>
                  AI-assisted stress & trauma-related conversational assessment
                </p>
              </div>
            </div>

            <div className="card mb-4">
              <p style={{ fontSize: 13, color: 'var(--subtle)', lineHeight: 1.6 }}>
                TraumaSense analyzes <strong>consented helpline conversations</strong> to identify stress, distress
                and contextual conversational indicators, and provide an <strong>assistive risk assessment</strong>
                for trained human review.
              </p>
              <div className="flex items-center gap-2 mt-3 text-sm text-muted" style={{ flexWrap: 'wrap' }}>
                <span className="flex items-center gap-1"><Shield size={13} /> Privacy-first</span>
                <span className="flex items-center gap-1"><UserCheck size={13} /> Human-in-the-loop</span>
              </div>
            </div>

            <div className="flex gap-3 flex-wrap">
              <label className="btn btn-primary" style={{ cursor: 'pointer' }}>
                <Upload size={15} /> Upload Call
                <input
                  type="file"
                  accept=".wav,.mp3,.m4a,.ogg,.webm"
                  className="input"
                  style={{ display: 'none' }}
                  onChange={handleFileChange}
                />
              </label>
            </div>
            <p className="text-xs text-dim mt-3">
              Drop a consented prerecorded helpline call to see the full analysis flow.
            </p>
          </>
        )}

        {/* UPLOADED — file ready, waiting for consent + analysis */}
        {mode === 'uploaded' && (
          <>
            <ConsentNotice
              acknowledged={consentAck}
              onAcknowledge={() => setConsentAck(true)}
              onBack={resetAll}
            />

            <AudioPlayer
              url={audioUrl}
              name={audioName}
              size={audioSize}
              duration={audioDuration}
              onTimeUpdate={handleTimeUpdate}
              onEnded={handleEnded}
              isPlaying={isPlaying}
              onPlayToggle={() => setIsPlaying(!isPlaying)}
            />

            <div className="flex gap-3 mt-3 flex-wrap">
              <button
                className="btn btn-primary"
                onClick={startAnalysis}
                disabled={!consentAck || isAnalyzing}
              >
                {isAnalyzing ? <><RefreshCw size={14} className="spin" /> Analyzing...</> : <><Activity size={15} /> Start Analysis</>}
              </button>
              <button className="btn btn-secondary" onClick={resetAll}>
                <RefreshCw size={14} /> Reset
              </button>
            </div>
            {!consentAck && (
              <p className="text-xs text-muted mt-2">
                Acknowledge the consent notice above to enable analysis.
              </p>
            )}
          </>
        )}

        {/* ANALYZING */}
        {mode === 'analyzing' && (
          <LoadingState
            progress={progress}
            message="Processing audio and generating analysis..."
          />
        )}

        {/* DONE — show results */}
        {mode === 'done' && caseData && (
          <>
            <CaseHeader
              caseId={caseData.case_id}
              fileName={caseData.file_name}
              duration={caseData.duration_seconds}
              riskLevel={caseData.overall_risk_level}
              riskScore={caseData.overall_risk_score}
              confidence={caseData.overall_confidence}
              onReset={resetAll}
              onPrint={handlePrint}
            />

            <div className="flex gap-2 mb-3 flex-wrap no-print">
              <button
                className={`btn btn-sm ${activeTab === 'dashboard' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setActiveTab('dashboard')}
              >
                <Activity size={13} /> Dashboard
              </button>
              <button
                className={`btn btn-sm ${activeTab === 'report' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setActiveTab('report')}
              >
                <Printer size={13} /> Final Report
              </button>
            </div>

            {activeTab === 'dashboard' && (
              <AnalysisDashboard
                caseData={caseData}
                currentSegment={currentSegment}
                isPlaying={isPlaying}
                audioRef={audioRef}
              />
            )}

            {activeTab === 'report' && (
              <FinalReport
                caseData={caseData}
                onPrint={handlePrint}
              />
            )}

            <Disclaimer />
          </>
        )}
      </main>
    </div>
  );
}
