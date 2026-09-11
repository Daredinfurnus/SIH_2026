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
  ChevronRight,
} from 'lucide-react';
import { healthCheck, uploadAndAnalyze, getCase } from './services/api';
import Header from './components/Header';
import ConsentNotice from './components/ConsentNotice';
import UploadPanel from './components/UploadPanel';
import AudioPlayer from './components/AudioPlayer';
import CaseHeader from './components/CaseHeader';
import AnalysisDashboard from './components/AnalysisDashboard';
import FinalReport from './components/FinalReport';
import TranscriptFormatter from './components/TranscriptFormatter';
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
  const [activeTab, setActiveTab] = useState('dashboard'); // dashboard | report | transcript
  const [backendOnline, setBackendOnline] = useState(true);
  const [progress, setProgress] = useState(0);
  const [transcriptSegments, setTranscriptSegments] = useState('');
  const [transcriptFormatted, setTranscriptFormatted] = useState('');
  const [transcriptCounts, setTranscriptCounts] = useState(null);

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
    if (!f.name.match(/\.(wav|mp3|m4a|aac|ogg|webm|mp4)$/i)) {
    setError('Unsupported format. Please use WAV, MP3, M4A, AAC, OGG, WEBM, or MP4.');
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

  // ---- scenario library ------------------------------------------------
  const DEMO_SCENARIOS = {
    low: {
      case_id: 'CASE-26093-DEMO-LOW-0001',
      file_name: 'demo_low_call.wav',
      duration_seconds: 48,
      transcript: [
        { start: 0, end: 8, text: 'Hello, thank you for calling back. How are you doing today?', speaker: 'counselor', stress_score: 12, distress_score: 10, emotion: 'Calm', confidence: 0.88, indicators: ['calm'], svi_score: 14, risk_level: 'LOW', risk_explanation: ['Caller is relaxed and engaged. No distress indicators present.'], emotion_explanation: ['Text-based emotion: Calm (IndicBERT similarity 0.88)'], accent_signals: null },
        { start: 8, end: 16, text: 'I am doing much better, thank you. The situation has settled down a lot.', speaker: 'caller', stress_score: 18, distress_score: 14, emotion: 'Calm', confidence: 0.85, indicators: ['calm'], svi_score: 16, risk_level: 'LOW', risk_explanation: ['Caller reports improvement. Low stress and distress levels.'], emotion_explanation: ['Text-based emotion: Calm (IndicBERT similarity 0.85)'], accent_signals: null },
        { start: 16, end: 24, text: 'That is good to hear. Are you still in touch with your support contact?', speaker: 'counselor', stress_score: 8, distress_score: 6, emotion: 'Calm', confidence: 0.90, indicators: ['calm'], svi_score: 10, risk_level: 'LOW', risk_explanation: ['Counselor tone is measured and supportive.'], emotion_explanation: ['Text-based emotion: Calm (IndicBERT similarity 0.90)'], accent_signals: null },
        { start: 24, end: 34, text: 'Yes, I have been speaking to them every day. I feel a lot more grounded now.', speaker: 'caller', stress_score: 20, distress_score: 16, emotion: 'Calm', confidence: 0.82, indicators: ['calm'], svi_score: 18, risk_level: 'LOW', risk_explanation: ['Caller reports strong support network. Grounding language used.'], emotion_explanation: ['Text-based emotion: Calm (IndicBERT similarity 0.82)'], accent_signals: null },
        { start: 34, end: 48, text: 'That is exactly what we want to hear. Keep that routine going. We will check in again next week.', speaker: 'counselor', stress_score: 10, distress_score: 8, emotion: 'Calm', confidence: 0.89, indicators: ['calm'], svi_score: 12, risk_level: 'LOW', risk_explanation: ['Counselor provides reinforcing, low-pressure guidance.'], emotion_explanation: ['Text-based emotion: Calm (IndicBERT similarity 0.89)'], accent_signals: null },
      ],
      overall_stress_score: 14,
      overall_distress_score: 11,
      overall_svi_score: 18,
      overall_risk_score: 18,
      overall_risk_level: 'LOW',
      overall_confidence: 0.86,
      overall_indicators: ['calm'],
      risk_explanation: [
        'SVI score 18/100 — well within the Low threshold (0–25).',
        'Stress and distress components both very low across all segments.',
        'Caller reports feeling grounded with active support contact.',
        'No safety concerns or distress indicators detected.',
        'Routine follow-up recommended.',
      ],
      recommendation: 'Low stress and distress indicators detected. Continue routine supportive follow-up and maintain existing support contacts.',
      mode: 'demo',
      disclaimer: 'Assistive risk indicator, not a clinical diagnosis. Low-risk results should still be reviewed in context.',
      immediate_safety_indicators: false,
      language: 'en',
      analyzed_at: '2026-09-11T10:00:00+00:00',
      model_status: { asr: 'success', text_emotion: 'success', acoustic_emotion: 'success', fusion: 'multimodal' },
      detected_language: 'en',
    },
    moderate: {
      case_id: 'CASE-26093-DEMO-MODERATE-0001',
      file_name: 'demo_moderate_call.wav',
      duration_seconds: 55,
      transcript: [
        { start: 0, end: 10, text: 'Hi, I wanted to follow up on the complaint I filed last month. Nothing seems to have moved.', speaker: 'caller', stress_score: 42, distress_score: 38, emotion: 'Anxiety', confidence: 0.78, indicators: ['anxiety', 'uncertainty'], svi_score: 40, risk_level: 'MODERATE', risk_explanation: ['Caller expresses frustration about lack of progress. Moderate anxiety indicators.'], emotion_explanation: ['Text-based emotion: Anxiety (IndicBERT similarity 0.78)'], accent_signals: null },
        { start: 10, end: 20, text: 'I understand your concern. Can you tell me what outcome you were expecting?', speaker: 'counselor', stress_score: 18, distress_score: 14, emotion: 'Calm', confidence: 0.86, indicators: ['calm'], svi_score: 16, risk_level: 'LOW', risk_explanation: ['Counselor uses open, non-leading question.'], emotion_explanation: ['Text-based emotion: Calm (IndicBERT similarity 0.86)'], accent_signals: null },
        { start: 20, end: 32, text: 'I expected at least some update by now. Every time I call, I am told it is still being processed.', speaker: 'caller', stress_score: 55, distress_score: 48, emotion: 'Frustration', confidence: 0.74, indicators: ['anxiety', 'uncertainty', 'helplessness'], svi_score: 51, risk_level: 'HIGH', risk_explanation: ['Frustration language detected. Helplessness indicator present.'], emotion_explanation: ['Fused emotion: Frustration (text 0.74/0.52)'], accent_signals: null },
        { start: 32, end: 42, text: 'That sounds exhausting. Let me check the current status and see what I can clarify for you.', speaker: 'counselor', stress_score: 22, distress_score: 18, emotion: 'Calm', confidence: 0.84, indicators: ['calm'], svi_score: 20, risk_level: 'LOW', risk_explanation: ['Counselor acknowledges caller frustration and offers concrete next step.'], emotion_explanation: ['Text-based emotion: Calm (IndicBERT similarity 0.84)'], accent_signals: null },
        { start: 42, end: 55, text: 'Thank you. I just want to know where things stand. I have been worrying about this for weeks.', speaker: 'caller', stress_score: 48, distress_score: 42, emotion: 'Anxiety', confidence: 0.76, indicators: ['anxiety', 'uncertainty'], svi_score: 45, risk_level: 'MODERATE', risk_explanation: ['Caller expresses ongoing worry. Moderate distress indicators persist.'], emotion_explanation: ['Text-based emotion: Anxiety (IndicBERT similarity 0.76)'], accent_signals: null },
      ],
      overall_stress_score: 37,
      overall_distress_score: 32,
      overall_svi_score: 42,
      overall_risk_score: 42,
      overall_risk_level: 'MODERATE',
      overall_confidence: 0.79,
      overall_indicators: ['anxiety', 'uncertainty', 'helplessness'],
      risk_explanation: [
        'SVI score 42/100 — within the Moderate threshold (26–50).',
        'Stress and distress components elevated but not at critical levels.',
        'Uncertainty and anxiety indicators present throughout the conversation.',
        'Caller reports ongoing worry about case progress.',
        'Active case status clarification recommended.',
      ],
      recommendation: 'Moderate stress and uncertainty indicators detected. Provide case status clarification and maintain supportive contact to reduce uncertainty.',
      mode: 'demo',
      disclaimer: 'Assistive risk indicator, not a clinical diagnosis. Moderate-risk results warrant attentive human review.',
      immediate_safety_indicators: false,
      language: 'en',
      analyzed_at: '2026-09-11T11:00:00+00:00',
      model_status: { asr: 'success', text_emotion: 'success', acoustic_emotion: 'success', fusion: 'multimodal' },
      detected_language: 'en',
    },
    high: {
      case_id: 'CASE-26093-DEMO-HIGH-0001',
      file_name: 'demo_high_call.wav',
      duration_seconds: 65,
      transcript: [
        { start: 0, end: 12, text: 'I do not know what to do anymore. They have been calling me constantly, telling me to drop the case.', speaker: 'caller', stress_score: 72, distress_score: 68, emotion: 'Fear', confidence: 0.88, indicators: ['fear', 'threat-related context', 'safety concern'], svi_score: 70, risk_level: 'HIGH', risk_explanation: ['Fear and threat-related context detected. Safety concern flagged.'], emotion_explanation: ['Fused emotion: Fear (text 0.88/0.62, acoustic 58/100/0.44)'], accent_signals: null },
        { start: 12, end: 24, text: 'Are you safe right now? Can you tell me more about what they are saying?', speaker: 'counselor', stress_score: 35, distress_score: 28, emotion: 'Calm', confidence: 0.82, indicators: ['calm'], svi_score: 32, risk_level: 'MODERATE', risk_explanation: ['Counselor assesses immediate safety. Measured, investigative tone.'], emotion_explanation: ['Text-based emotion: Calm (IndicBERT similarity 0.82)'], accent_signals: null },
        { start: 24, end: 36, text: 'I am at home but I do not feel safe. They know where I live. My phone has been ringing all day.', speaker: 'caller', stress_score: 82, distress_score: 78, emotion: 'Fear', confidence: 0.92, indicators: ['fear', 'safety concern', 'threat-related context', 'helplessness'], svi_score: 80, risk_level: 'HIGH', risk_explanation: ['High fear and safety concern. Helplessness indicator present. Caller reports direct knowledge of address.'], emotion_explanation: ['Fused emotion: Fear (text 0.92/0.70, acoustic 68/100/0.50)'], accent_signals: null },
        { start: 36, end: 48, text: 'I want to support you through this. Have you been able to reach anyone in your family or community?', speaker: 'counselor', stress_score: 28, distress_score: 22, emotion: 'Calm', confidence: 0.83, indicators: ['calm'], svi_score: 26, risk_level: 'LOW', risk_explanation: ['Counselor offers supportive, grounding question.'], emotion_explanation: ['Text-based emotion: Calm (IndicBERT similarity 0.83)'], accent_signals: null },
        { start: 48, end: 58, text: 'My sister is with me tonight but I do not want to put her in the middle of this. She is scared too.', speaker: 'caller', stress_score: 68, distress_score: 62, emotion: 'Fear', confidence: 0.86, indicators: ['fear', 'safety concern', 'helplessness', 'isolation'], svi_score: 66, risk_level: 'HIGH', risk_explanation: ['Caller reports family member present but reluctant to involve them. Isolation indicator present.'], emotion_explanation: ['Fused emotion: Fear (text 0.86/0.64, acoustic 55/100/0.42)'], accent_signals: null },
        { start: 58, end: 65, text: 'We will make a note of everything you have shared. Can you describe what they said specifically?', speaker: 'counselor', stress_score: 30, distress_score: 24, emotion: 'Calm', confidence: 0.81, indicators: ['calm'], svi_score: 28, risk_level: 'LOW', risk_explanation: ['Counselor moves to structured information gathering.'], emotion_explanation: ['Text-based emotion: Calm (IndicBERT similarity 0.81)'], accent_signals: null },
      ],
      overall_stress_score: 53,
      overall_distress_score: 47,
      overall_svi_score: 68,
      overall_risk_score: 68,
      overall_risk_level: 'HIGH',
      overall_confidence: 0.85,
      overall_indicators: ['fear', 'safety concern', 'threat-related context', 'helplessness', 'isolation'],
      risk_explanation: [
        'SVI score 68/100 — within the High threshold (51–75).',
        'Stress component elevated across multiple segments.',
        'Fear, safety concern, and threat-related context detected.',
        'Caller reports ongoing contact and knowledge of home address.',
        'Family member present but caller reluctant to involve them.',
        'Priority human review and safety planning recommended.',
      ],
      recommendation: 'Elevated stress and fear indicators detected. Priority human review recommended. Document specific threat language and assess immediate safety needs with the caller.',
      mode: 'demo',
      disclaimer: 'Assistive risk indicator, not a clinical diagnosis. High-risk results require trained human review.',
      immediate_safety_indicators: false,
      language: 'en',
      analyzed_at: '2026-09-11T12:00:00+00:00',
      model_status: { asr: 'success', text_emotion: 'success', acoustic_emotion: 'success', fusion: 'multimodal' },
      detected_language: 'en',
    },
    critical: {
      case_id: 'CASE-26093-DEMO-CRITICAL-0001',
      file_name: 'demo_critical_call.wav',
      duration_seconds: 72,
      transcript: [
        { start: 0, end: 14, text: 'Please help me. They came to my house tonight. They broke the door and they were inside.', speaker: 'caller', stress_score: 94, distress_score: 92, emotion: 'Fear', confidence: 0.96, indicators: ['fear', 'safety concern', 'immediate safety indicators', 'threat-related context'], svi_score: 93, risk_level: 'CRITICAL', risk_explanation: ['Immediate safety indicators detected. Caller reports break-in and intrusion.'], emotion_explanation: ['Fused emotion: Fear (text 0.96/0.78, acoustic 82/100/0.58)'], accent_signals: null },
        { start: 14, end: 26, text: 'Are you in a safe place right now? Is anyone with you?', speaker: 'counselor', stress_score: 42, distress_score: 35, emotion: 'Calm', confidence: 0.84, indicators: ['calm'], svi_score: 38, risk_level: 'MODERATE', risk_explanation: ['Counselor prioritizes immediate safety assessment.'], emotion_explanation: ['Text-based emotion: Calm (IndicBERT similarity 0.84)'], accent_signals: null },
        { start: 26, end: 38, text: 'I am hiding in my room with my daughter. She is only six. They threatened to come back and hurt us both.', speaker: 'caller', stress_score: 96, distress_score: 97, emotion: 'Fear', confidence: 0.97, indicators: ['fear', 'safety concern', 'immediate safety indicators', 'threat-related context', 'helplessness', 'isolation'], svi_score: 96, risk_level: 'CRITICAL', risk_explanation: ['Immediate safety indicators detected. Caller and child hiding. Direct threats against both.'], emotion_explanation: ['Fused emotion: Fear (text 0.97/0.80, acoustic 88/100/0.62)'], accent_signals: null },
        { start: 38, end: 50, text: 'Listen to me carefully. You are not alone. Can you stay on the line with me while we work out what to do?', speaker: 'counselor', stress_score: 55, distress_score: 48, emotion: 'Anxiety', confidence: 0.79, indicators: ['anxiety', 'safety concern'], svi_score: 52, risk_level: 'HIGH', risk_explanation: ['Counselor maintains contact and offers grounding support. Moderate anxiety reflects situation gravity.'], emotion_explanation: ['Fused emotion: Anxiety (text 0.79/0.58)'], accent_signals: null },
        { start: 50, end: 62, text: 'I am shaking so much I cannot think straight. My daughter is crying. I thought they were going to kill us.', speaker: 'caller', stress_score: 95, distress_score: 96, emotion: 'Fear', confidence: 0.96, indicators: ['fear', 'safety concern', 'immediate safety indicators', 'helplessness', 'isolation', 'sleep disturbance'], svi_score: 95, risk_level: 'CRITICAL', risk_explanation: ['Critical distress indicators. Caller in acute fear response. Child present and distressed.'], emotion_explanation: ['Fused emotion: Fear (text 0.96/0.78, acoustic 85/100/0.60)'], accent_signals: null },
        { start: 62, end: 72, text: 'You are doing the right thing by staying on the line. What is the safest room you can get to, away from the door?', speaker: 'counselor', stress_score: 48, distress_score: 42, emotion: 'Anxiety', confidence: 0.77, indicators: ['anxiety', 'safety concern'], svi_score: 46, risk_level: 'MODERATE', risk_explanation: ['Counselor provides concrete safety guidance while maintaining contact.'], emotion_explanation: ['Fused emotion: Anxiety (text 0.77/0.55)'], accent_signals: null },
      ],
      overall_stress_score: 73,
      overall_distress_score: 72,
      overall_svi_score: 88,
      overall_risk_score: 88,
      overall_risk_level: 'CRITICAL',
      overall_confidence: 0.91,
      overall_indicators: ['fear', 'safety concern', 'immediate safety indicators', 'threat-related context', 'helplessness', 'isolation', 'sleep disturbance'],
      risk_explanation: [
        'SVI score 88/100 — at or above the Critical threshold (76–100).',
        'Stress component 73 and distress component 72 both critically elevated.',
        'Immediate safety indicators detected at multiple points.',
        'Caller and child hiding after reported break-in and intrusion.',
        'Direct threats against both caller and child reported.',
        'Caller in acute fear response with physical symptoms.',
        'Trained human review and emergency safety planning required immediately.',
      ],
      recommendation: 'Critical distress and immediate safety indicators detected. Escalate immediately for trained human review and emergency safety planning. Caller and child require urgent support.',
      mode: 'demo',
      disclaimer: 'Assistive risk indicator, not a clinical diagnosis. Critical-risk results require immediate trained human review.',
      immediate_safety_indicators: true,
      language: 'en',
      analyzed_at: '2026-09-11T13:00:00+00:00',
      model_status: { asr: 'success', text_emotion: 'success', acoustic_emotion: 'success', fusion: 'multimodal' },
      detected_language: 'en',
    },
  };

  const startScenario = (level) => {
    setError(null);
    setIsAnalyzing(false);
    setProgress(0);
    setCurrentSegment(0);
    const data = DEMO_SCENARIOS[level];
    if (data) {
      setCaseData(data);
      setMode('done');
      setActiveTab('dashboard');
    }
  };
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
        if (p >= 99) { clearInterval(iv); return 99; }
        return p + Math.floor(Math.random() * 8) + 3;
      });
    }, 400);
    progressInterval.current = iv;
    return () => clearInterval(iv);
  }, [mode]);

  // ---- transcript formatter helpers ------------------------------------
  function formatTranscriptFromLines(lines) {
    const LABEL_MAP = {
      'voice 1': 'Voice 1 (NHAA Official)',
      'voice 2': 'Voice 2 (Victim)',
    };
    function resolveLabel(raw, prevKey) {
      const key = raw.trim().toLowerCase();
      if (key in LABEL_MAP) return LABEL_MAP[key];
      if (!prevKey) return 'Voice 1 (NHAA Official) [assumed]';
      if (prevKey.startsWith('Voice 1')) return 'Voice 2 (Victim) [assumed]';
      return 'Voice 1 (NHAA Official) [assumed]';
    }
    function parseOne(text) {
      const t = text.trim();
      if (!t) return null;
      let m = t.match(/^\[?\s*(Voice\s*\d+)\s*\]?\s*:?\s*(.*)$/i);
      if (m) return { label: resolveLabel(m[1], null), body: m[2].trim() };
      m = t.match(/^(Voice\s*\d+)\s*:?\s*(.*)$/i);
      if (m) return { label: resolveLabel(m[1], null), body: m[2].trim() };
      return { label: resolveLabel('', null), body: t };
    }
    const blocks = [];
    let prevKey = null;
    for (const raw of lines) {
      const p = parseOne(raw);
      if (!p || !p.body) continue;
      const speakerKey = p.label.split(' (')[0];
      if (blocks.length > 0 && blocks[blocks.length - 1].label.startsWith(speakerKey)) {
        blocks[blocks.length - 1].text += ' ' + p.body;
      } else {
        blocks.push({ label: p.label, text: p.body });
      }
      prevKey = speakerKey;
    }
    return blocks.map(b => `${b.label}: ${b.text}`).join(' ');
  }

  function countTranscriptBlocks(formatted) {
    if (!formatted) return { voice1: 0, voice2: 0, total: 0 };
    return {
      voice1: (formatted.match(/Voice 1 \(/g) || []).length,
      voice2: (formatted.match(/Voice 2 \(/g) || []).length,
      total: ((formatted.match(/Voice 1 \(/g) || []).length) + ((formatted.match(/Voice 2 \(/g) || []).length),
    };
  }

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
                to support trained human review.
              </p>
              <div className="flex items-center gap-2 mt-3 text-sm text-muted" style={{ flexWrap: 'wrap' }}>
                <span className="flex items-center gap-1"><Shield size={13} /> Privacy-first</span>
                <span className="flex items-center gap-1"><UserCheck size={13} /> Human-in-the-loop</span>
              </div>
            </div>

            <div className="card mb-4" style={{ background: 'rgba(75,110,245,0.05)', border: '1px solid rgba(75,110,245,0.15)' }}>
              <div className="flex items-center gap-2 mb-2">
                <Brain size={14} color="#4B6EF5" />
                <span style={{ fontSize: 11, fontWeight: 700, color: '#4B6EF5', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Demo Mode — SVI Scenarios</span>
              </div>
              <p className="text-xs text-muted mb-3" style={{ lineHeight: 1.5 }}>
                Explore four complete, self-contained TraumaSense conversation analyses across the SVI spectrum.
              </p>
              <div className="grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10 }}>
                <div className="card scenario-card" style={{ borderColor: 'rgba(52,211,153,0.4)', cursor: 'pointer' }} onClick={() => startScenario('low')}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="risk-badge risk-low" style={{ fontSize: 13, padding: '3px 10px' }}>LOW</span>
                    <span className="text-xs text-dim">0–25</span>
                  </div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: '#34D399', lineHeight: 1 }}>SVI 18<span style={{ fontSize: 11, fontWeight: 400, color: 'var(--muted-foreground)' }}>/100</span></div>
                  <div className="text-xs text-dim mt-1" style={{ lineHeight: 1.4 }}>
                    Calm, cooperative conversation. Low stress and distress. Active support contact in place.
                  </div>
                  <div className="text-xs text-muted mt-2" style={{ color: '#34D399', fontWeight: 600 }}>
                    View Scenario <ChevronRight size={12} style={{ marginLeft: 4, verticalAlign: ' middle' }} />
                  </div>
                </div>
                <div className="card scenario-card" style={{ borderColor: 'rgba(251,191,36,0.4)', cursor: 'pointer' }} onClick={() => startScenario('moderate')}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="risk-badge risk-moderate" style={{ fontSize: 13, padding: '3px 10px' }}>MODERATE</span>
                    <span className="text-xs text-dim">26–50</span>
                  </div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: '#FBBF24', lineHeight: 1 }}>SVI 42<span style={{ fontSize: 11, fontWeight: 400, color: 'var(--muted-foreground)' }}>/100</span></div>
                  <div className="text-xs text-dim mt-1" style={{ lineHeight: 1.4 }}>
                    Ongoing case uncertainty. Frustration about lack of progress. Moderate anxiety and helplessness indicators.
                  </div>
                  <div className="text-xs text-muted mt-2" style={{ color: '#FBBF24', fontWeight: 600 }}>
                    View Scenario <ChevronRight size={12} style={{ marginLeft: 4, verticalAlign: ' middle' }} />
                  </div>
                </div>
                <div className="card scenario-card" style={{ borderColor: 'rgba(248,113,113,0.4)', cursor: 'pointer' }} onClick={() => startScenario('high')}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="risk-badge risk-high" style={{ fontSize: 13, padding: '3px 10px' }}>HIGH</span>
                    <span className="text-xs text-dim">51–75</span>
                  </div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: '#F87171', lineHeight: 1 }}>SVI 68<span style={{ fontSize: 11, fontWeight: 400, color: 'var(--muted-foreground)' }}>/100</span></div>
                  <div className="text-xs text-dim mt-1" style={{ lineHeight: 1.4 }}>
                    Intimidation and repeated threats to drop a case. Caller reports fear, safety concern, and isolation.
                  </div>
                  <div className="text-xs text-muted mt-2" style={{ color: '#F87171', fontWeight: 600 }}>
                    View Scenario <ChevronRight size={12} style={{ marginLeft: 4, verticalAlign: ' middle' }} />
                  </div>
                </div>
                <div className="card scenario-card" style={{ borderColor: 'rgba(239,68,68,0.4)', cursor: 'pointer' }} onClick={() => startScenario('critical')}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="risk-badge risk-critical" style={{ fontSize: 13, padding: '3px 10px' }}>CRITICAL</span>
                    <span className="text-xs text-dim">76–100</span>
                  </div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: '#EF4444', lineHeight: 1 }}>SVI 88<span style={{ fontSize: 11, fontWeight: 400, color: 'var(--muted-foreground)' }}>/100</span></div>
                  <div className="text-xs text-dim mt-1" style={{ lineHeight: 1.4 }}>
                    Reported break-in and intrusion. Caller and child hiding. Direct threats. Immediate safety indicators detected.
                  </div>
                  <div className="text-xs text-muted mt-2" style={{ color: '#EF4444', fontWeight: 600 }}>
                    View Scenario <ChevronRight size={12} style={{ marginLeft: 4, verticalAlign: ' middle' }} />
                  </div>
                </div>
              </div>

              <div className="card" style={{ marginTop: 16, borderColor: 'rgba(138,168,200,0.12)' }}>
                <div className="flex items-center gap-2 mb-3">
                  <Activity size={13} color="#8aa8c8" />
                  <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted-foreground)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Live Analysis</span>
                </div>
                <p className="text-xs text-dim mb-3" style={{ lineHeight: 1.5 }}>
                  Upload a consented prerecorded helpline call for real AI-powered analysis via the backend pipeline.
                </p>
                <div className="flex gap-3 flex-wrap">
                  <label className="btn btn-primary" style={{ cursor: 'pointer' }}>
                    <Upload size={15} /> Upload Call
                    <input
                      type="file"
                      accept=".wav,.mp3,.m4a,.aac,.ogg,.webm,.mp4"
                      className="input"
                      style={{ display: 'none' }}
                      onChange={handleFileChange}
                    />
                  </label>
                </div>
              </div>
            </div>
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
              <button
                className={`btn btn-sm ${activeTab === 'transcript' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setActiveTab('transcript')}
              >
                <svg width={13} height={13} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
                Transcript Formatter
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

            {activeTab === 'transcript' && (
              <TranscriptFormatter
                segments={transcriptSegments}
                formatted={transcriptFormatted}
                counts={transcriptCounts}
                onFormat={({ segments, formatted }) => {
                  setTranscriptSegments(segments);
                  if (formatted == null) {
                    // Compute the formatted output from the raw segments
                    const lines = segments.split('\n').map(l => l.trim()).filter(Boolean);
                    const formattedOut = formatTranscriptFromLines(lines);
                    const counts = countTranscriptBlocks(formattedOut);
                    setTranscriptFormatted(formattedOut);
                    setTranscriptCounts(counts);
                  } else {
                    setTranscriptFormatted(formatted);
                    setTranscriptCounts(null);
                  }
                }}
                onClear={() => {
                  setTranscriptSegments('');
                  setTranscriptFormatted('');
                  setTranscriptCounts(null);
                }}
              />
            )}

            <Disclaimer />
          </>
        )}
      </main>
    </div>
  );
}
