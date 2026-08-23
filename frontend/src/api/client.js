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
    response = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
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

export const api = {
  getHealth: () => request("/health"),

  getModelInfo: () => request("/model/info"),

  scoreTransaction: (payload) =>
    request("/risk/score", { method: "POST", body: JSON.stringify(payload) }),

  explainTransaction: (payload) =>
    request("/risk/explain", { method: "POST", body: JSON.stringify(payload) }),

  evaluateTransaction: (payload) =>
    request("/risk/evaluate", { method: "POST", body: JSON.stringify(payload) }),

  getAuditLog: (limit = 25) => request(`/audit-log?limit=${limit}`),
};

export { ApiError };
