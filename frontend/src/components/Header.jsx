import React from 'react';
import { Monitor, UserCheck } from 'lucide-react';

export default function Header() {
  return (
    <header className="header">
      <div className="header-brand">
        <div className="header-logo">TS</div>
        <div className="header-title">
          <h1>SMART INDIA HACKATHON 2026</h1>
          <div className="subtitle">
            PS 26093 · TraumaSense · AI-assisted stress & trauma-related conversational assessment
          </div>
        </div>
      </div>
      <div className="header-meta">
        <span className="badge badge-ps"><Monitor size={10} /> PS 26093</span>
        <span className="badge badge-hil"><UserCheck size={10} /> Human-in-the-Loop</span>
      </div>
    </header>
  );
}
