import { useEffect, useState, useCallback } from "react";
import { api, ApiError } from "./api/client";
import TransactionForm from "./components/TransactionForm";
import RiskResultCard from "./components/RiskResultCard";
import ExplanationPanel from "./components/ExplanationPanel";
import AuditLogTable from "./components/AuditLogTable";
import ModelInfoPanel from "./components/ModelInfoPanel";
import ErrorBanner from "./components/ErrorBanner";
import Sidebar from "./components/Sidebar";
import StatCard from "./components/StatCard";
import { useTheme } from "./hooks/useTheme";
import "./styles/App.css";

export default function App() {
  const { theme, toggle } = useTheme();
  const [view, setView] = useState("evaluate");
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);

  const [modelInfo, setModelInfo] = useState(null);
  const [modelInfoError, setModelInfoError] = useState(null);
  const [modelInfoLoading, setModelInfoLoading] = useState(true);

  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [currentPayload, setCurrentPayload] = useState(null);  // V2: track payload for history visualization
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
    setCurrentPayload(payload);  // V2: store for history visualization
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
  const m = modelInfo?.test_metrics_at_frozen_threshold;
  const pct = (v) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—");

  const status = healthError
    ? { tone: "degraded", label: "API unreachable" }
    : isOk
      ? { tone: "ok", label: "All systems operational" }
      : { tone: "degraded", label: "Model degraded" };

  return (
    <div className="app-shell">
      <Sidebar view={view} onNavigate={setView} theme={theme} onToggleTheme={toggle} status={status} />

      <main className="main-col">
        <div className="stat-row">
          <StatCard label="Model" value={modelVersion} hint="Active classifier" />
          <StatCard label="Threshold" value={threshold} hint="Cost-optimised cut-off" />
          <StatCard label="Precision" value={pct(m?.precision)} hint="Held-out test set" />
          <StatCard label="Recall" value={pct(m?.recall)} hint="Fraud caught" tone="good" />
        </div>

        {view === "evaluate" && (
          <>
            <header className="page-head">
              <h1 className="page-title">Evaluate a transaction</h1>
              <p className="page-sub">
                Score a payment, see which signals drove the risk, and record the recommended action.
              </p>
            </header>
            <p className="context-note">
              ⓘ Prototype using synthetic data. Not connected to any live payment system.
            </p>
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
                <ExplanationPanel
                  explanation={explanation}
                  isLoading={evaluateLoading}
                  payload={currentPayload}
                />
              </div>
            </div>
          </>
        )}

        {view === "monitor" && (
          <>
            <header className="page-head">
              <h1 className="page-title">Model &amp; audit trail</h1>
              <p className="page-sub">
                What the model was trained on, how it performed, and every decision it has made.
              </p>
            </header>
            <div className="stack">
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
          </>
        )}
      </main>
    </div>
  );
}
