import { useEffect, useState, useCallback } from "react";
import { api, ApiError } from "./api/client";
import StatusBar from "./components/StatusBar";
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
      // /risk/explain: the same underlying model/SHAP computation, used here only to obtain
      //   the raw per-feature contributions for the visual breakdown -- both calls hit the
      //   real, existing endpoints; nothing is duplicated or recomputed client-side.
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
      setEvaluateError(err instanceof ApiError ? err.message : "Unexpected error while evaluating the transaction.");
    } finally {
      setEvaluateLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="app-header">
        <div>
          <div className="app-title">MerchantShield AI</div>
          <div className="app-subtitle">Fraud risk decision support — Razorpay AI Buildathon 2026, Track 02</div>
        </div>
      </div>

      <StatusBar health={health} modelInfo={modelInfo} healthError={healthError} />

      <div className="main-grid">
        <TransactionForm onSubmit={handleEvaluate} isSubmitting={evaluateLoading} />

        <div className="stack">
          {evaluateError && <ErrorBanner message="Could not evaluate this transaction." detail={evaluateError} />}
          <RiskResultCard result={result} isLoading={evaluateLoading} />
          <ExplanationPanel explanation={explanation} isLoading={evaluateLoading} />
        </div>
      </div>

      <div className="stack" style={{ marginTop: 20 }}>
        <ModelInfoPanel modelInfo={modelInfo} isLoading={modelInfoLoading} error={modelInfoError} />
        <AuditLogTable
          entries={auditEntries}
          isLoading={auditLoading}
          error={auditError}
          onRefresh={loadAuditLog}
        />
      </div>
    </div>
  );
}
