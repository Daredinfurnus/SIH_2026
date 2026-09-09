const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const url = `${BASE_URL}/api${path}`;
  const init = {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  };
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export function healthCheck() {
  return request('/health');
}

export function getDemoAnalysis() {
  return request('/demo');
}

export function uploadAndAnalyze(file) {
  const form = new FormData();
  form.append('file', file);
  return fetch(`${BASE_URL}/api/analyze`, {
    method: 'POST',
    body: form,
  }).then(r => {
    if (!r.ok) return r.json().then(j => { throw new Error(j.detail || 'Upload failed'); });
    return r.json();
  });
}

export function getCase(caseId) {
  return request(`/cases/${caseId}`);
}
