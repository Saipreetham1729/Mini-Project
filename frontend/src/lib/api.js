// Thin fetch wrappers matching the backend's real endpoints and Pydantic
// schemas field-for-field (see backend/app/schemas/*.py). Keeping this in
// one file means every request shape is defined in exactly one place.

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function handleResponse(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // response wasn't JSON -- fall back to statusText
    }
    throw new Error(detail);
  }
  return res.json();
}

// Wraps fetch so a dropped connection (backend not running, wrong URL, no
// network) reads as a clear, actionable message instead of a raw
// "Failed to fetch" -- the message that matters most live, mid-demo.
async function safeFetch(url, options) {
  try {
    return await fetch(url, options);
  } catch {
    throw new Error(`Could not reach the backend at ${API_BASE}. Is it running?`);
  }
}

export async function getHealth() {
  const res = await safeFetch(`${API_BASE}/health`);
  return handleResponse(res);
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await safeFetch(`${API_BASE}/documents/upload`, { method: "POST", body: formData });
  return handleResponse(res);
}

export async function processDocument(docId) {
  const res = await safeFetch(`${API_BASE}/documents/${docId}/process`, { method: "POST" });
  return handleResponse(res);
}

export async function embedDocument(docId) {
  const res = await safeFetch(`${API_BASE}/documents/${docId}/embed`, { method: "POST" });
  return handleResponse(res);
}

export async function runQuery(docId, question, topK = 5) {
  const res = await safeFetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id: docId, question, top_k: topK }),
  });
  return handleResponse(res);
}

export async function verifyAnswer({ docId, question, answer, retrievedChunks }) {
  const res = await safeFetch(`${API_BASE}/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      doc_id: docId,
      question,
      answer,
      retrieved_chunks: retrievedChunks,
    }),
  });
  return handleResponse(res);
}

export async function mitigate({ docId, question, originalAnswer, claims, topK = 5 }) {
  const res = await safeFetch(`${API_BASE}/mitigate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      doc_id: docId,
      question,
      original_answer: originalAnswer,
      claims,
      top_k: topK,
    }),
  });
  return handleResponse(res);
}

export async function evaluate({ docId, question, answer, retrievedChunks }) {
  const res = await safeFetch(`${API_BASE}/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      doc_id: docId,
      question,
      answer,
      retrieved_chunks: retrievedChunks,
    }),
  });
  return handleResponse(res);
}

export async function getDashboardSummary() {
  const res = await safeFetch(`${API_BASE}/dashboard/summary`);
  return handleResponse(res);
}
