// API client for MerchantShield AI.
//
// DECOUPLING RULE: this file (and everything downstream of it) only ever speaks
// HTTP + JSON against the documented contract in docs/api.md. It never imports
// Python, model files, policy thresholds, or feature-engineering code -- those
// live entirely on the server. If the backend's decision logic changes, this file
// does not need to change unless the response SHAPE changes.

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    const { headers, ...rest } = options;
    response = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json", ...(headers || {}) },
      ...rest,
    });
  } catch (networkErr) {
    // fetch itself threw -- backend unreachable (down, CORS, DNS, etc.)
    throw new ApiError("Cannot reach the MerchantShield API.", 0, String(networkErr.message || networkErr));
  }

  let body = null;
  try {
    body = await response.json();
  } catch {
    // response wasn't JSON (e.g. a 500 from an unrelated proxy) -- fall through,
    // body stays null and we still surface the status code cleanly.
  }

  if (!response.ok) {
    const detail = body?.detail || body?.error || `Request failed with status ${response.status}`;
    throw new ApiError(detail, response.status, body);
  }

  return body;
}

// Analyst identity lives in this browser only. The token is sent only with actions that change
// something, and only if the person typed one in.
const ANALYST_TOKEN_KEY = "ms-analyst-token";
function analystHeaders() {
  try {
    const t = window.localStorage.getItem(ANALYST_TOKEN_KEY);
    return t ? { "X-Analyst-Token": t } : {};
  } catch {
    return {};
  }
}
const act = (path, body) =>
  request(path, { method: "POST", headers: analystHeaders(), body: JSON.stringify(body) });

export const api = {
  getHealth: () => request("/health"),

  getModelInfo: () => request("/model/info"),

  scoreTransaction: (payload) =>
    request("/risk/score", { method: "POST", body: JSON.stringify(payload) }),

  explainTransaction: (payload) =>
    request("/risk/explain", { method: "POST", body: JSON.stringify(payload) }),

  evaluateTransaction: (payload) =>
    request("/risk/evaluate", { method: "POST", body: JSON.stringify(payload) }),

  getRegistryChampion: () => request("/registry/champion"),

  getRegistryRuns: () => request("/registry/runs"),

  getRegistryEvents: () => request("/registry/events"),

  getRetrainStatus: () => request("/retrain/status"),

  getRetrainLog: (limit = 30) => request(`/retrain/log?limit=${limit}`),

  getDriftLatest: () => request("/monitoring/drift/latest"),

  getDriftHistory: (limit = 30) => request(`/monitoring/drift/history?limit=${limit}`),

  getAnomalies: (severities = []) => {
    const q = severities.map((s) => `severity=${s}`).join("&");
    return request(`/monitoring/anomalies/latest${q ? `?${q}` : ""}`);
  },

  getCaseSummary: () => request("/cases/summary"),

  getCaseAlerts: (status) => request(`/cases/alerts${status ? `?status=${status}` : ""}`),

  getCaseAlert: (id) => request(`/cases/alerts/${id}`),

  getCaseAccounts: (status) => request(`/cases/accounts${status ? `?status=${status}` : ""}`),

  getCaseEvents: (limit = 40) => request(`/cases/events?limit=${limit}`),

  holdAlert: (id, body) => act(`/cases/alerts/${id}/hold`, body),

  dismissAlert: (id, body) => act(`/cases/alerts/${id}/dismiss`, body),

  releaseAccount: (customerId, body) => act(`/cases/accounts/${encodeURIComponent(customerId)}/release`, body),

  closeAccount: (customerId, body) => act(`/cases/accounts/${encodeURIComponent(customerId)}/close`, body),

  getAuditLog: (limit = 25) => request(`/audit-log?limit=${limit}`),
};

export { ApiError };
