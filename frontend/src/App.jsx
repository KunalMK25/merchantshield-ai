import { useEffect, useState, useCallback } from "react";
import { api, ApiError } from "./api/client";
import TransactionForm from "./components/TransactionForm";
import RiskResultCard from "./components/RiskResultCard";
import ExplanationPanel from "./components/ExplanationPanel";
import AuditLogTable from "./components/AuditLogTable";
import ModelInfoPanel from "./components/ModelInfoPanel";
import ErrorBanner from "./components/ErrorBanner";
import "./styles/App.css";

export default function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);

  const [modelInfo, setModelInfo] = useState(null);
  const [modelInfoError, setModelInfoError] = useState(null);
  const [modelInfoLoading, setModelInfoLoading] = useState(true);

  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [evaluateLoading, setEvaluateLoading] = useState(false);
  const [evaluateError, setEvaluateError] = useState(null);

  const [auditEntries, setAuditEntries] = useState(null);
  const [auditLoading, setAuditLoading] = useState(true);
  const [auditError, setAuditError] = useState(null);

  const loadHealth = useCallback(async () => {
    try {
      const data = await api.getHealth();
      setHealth(data);
      setHealthError(null);
    } catch (err) {
      setHealth(null);
      setHealthError(err instanceof ApiError ? err.message : "Unknown error");
    }
  }, []);

  const loadModelInfo = useCallback(async () => {
    setModelInfoLoading(true);
    try {
      const data = await api.getModelInfo();
      setModelInfo(data);
      setModelInfoError(null);
    } catch (err) {
      setModelInfo(null);
      setModelInfoError(err instanceof ApiError ? err.message : "Unknown error");
    } finally {
      setModelInfoLoading(false);
    }
  }, []);

  const loadAuditLog = useCallback(async () => {
    setAuditLoading(true);
    try {
      const data = await api.getAuditLog(20);
      setAuditEntries(data);
      setAuditError(null);
    } catch (err) {
      setAuditError(err instanceof ApiError ? err.message : "Unknown error");
    } finally {
      setAuditLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHealth();
    loadModelInfo();
    loadAuditLog();
  }, [loadHealth, loadModelInfo, loadAuditLog]);

  async function handleEvaluate(payload) {
    setEvaluateLoading(true);
    setEvaluateError(null);
    setResult(null);
    setExplanation(null);
    try {
      // /risk/evaluate: the authoritative decision (runs the decision engine + audit write)
      // /risk/explain: same underlying model/SHAP computation, used to obtain raw
      //   per-feature contributions for the visual breakdown — both call real endpoints;
      //   nothing is duplicated or recomputed client-side.
      const [evaluateResult, explainResult] = await Promise.all([
        api.evaluateTransaction(payload),
        api.explainTransaction(payload),
      ]);
      setResult(evaluateResult);
      setExplanation({
        header: evaluateResult.explanation_header,
        reasons: evaluateResult.reasons,
        contributions: explainResult.contributions,
      });
      loadAuditLog();
    } catch (err) {
      setEvaluateError(
        err instanceof ApiError
          ? err.message
          : "Unexpected error while evaluating the transaction."
      );
    } finally {
      setEvaluateLoading(false);
    }
  }

  const isOk = health?.status === "ok" && health?.model_loaded;
  const modelVersion = health?.model_version || modelInfo?.model_version || "—";
  const threshold = modelInfo?.decision_threshold ?? "—";

  return (
    <div className="app-shell">
      {/* ── Header ───────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="header-brand">
          <div className="app-title">MerchantShield AI</div>
          <div className="app-tagline">Fraud risk decision support</div>
          <div className="app-event-badge" aria-label="Razorpay AI Buildathon 2026, Track 02">
            Razorpay AI Buildathon 2026 · Track 02
          </div>
        </div>

        <div className="header-status">
          <div className="header-status-pill" role="status">
            <span
              className={`header-status-dot ${
                healthError ? "degraded" : isOk ? "ok" : "degraded"
              }`}
              aria-hidden="true"
            />
            <span className="header-status-label">
              {healthError ? "Unreachable" : isOk ? "Operational" : "Degraded"}
            </span>
          </div>
          <div className="header-status-meta" aria-label="System metadata">
            <span className="header-status-item">
              Model: <strong className="mono">{modelVersion}</strong>
            </span>
            <span className="header-status-item">
              Threshold: <strong className="mono">{threshold}</strong>
            </span>
            {modelInfo?.test_metrics_at_frozen_threshold && (
              <>
                <span className="header-status-item">
                  Precision:{" "}
                  <strong className="mono">
                    {(modelInfo.test_metrics_at_frozen_threshold.precision * 100).toFixed(1)}%
                  </strong>
                </span>
                <span className="header-status-item">
                  Recall:{" "}
                  <strong className="mono">
                    {(modelInfo.test_metrics_at_frozen_threshold.recall * 100).toFixed(1)}%
                  </strong>
                </span>
              </>
            )}
          </div>
        </div>
      </header>

      <main>
        {/* ── Context bar ──────────────────────────────────────────── */}
        <p className="context-bar">
          Evaluate transaction risk, understand why the model flagged it, and record the decision.
        </p>

        {/* ── Evaluation grid ──────────────────────────────────────── */}
        <div className="main-grid">
          <TransactionForm onSubmit={handleEvaluate} isSubmitting={evaluateLoading} />

          <div className="stack">
            {evaluateError && (
              <ErrorBanner
                message="Could not evaluate this transaction."
                detail={evaluateError}
              />
            )}
            <RiskResultCard result={result} isLoading={evaluateLoading} />
            <ExplanationPanel explanation={explanation} isLoading={evaluateLoading} />
          </div>
        </div>

        {/* ── Bottom panels ────────────────────────────────────────── */}
        <div className="stack" style={{ marginTop: 24 }}>
          <ModelInfoPanel
            modelInfo={modelInfo}
            isLoading={modelInfoLoading}
            error={modelInfoError}
          />
          <AuditLogTable
            entries={auditEntries}
            isLoading={auditLoading}
            error={auditError}
            onRefresh={loadAuditLog}
          />
        </div>
      </main>
    </div>
  );
}
