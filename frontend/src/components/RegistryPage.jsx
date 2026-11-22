import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import ErrorBanner from "./ErrorBanner";
import { money } from "../utils/currency";

const pct = (v) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—");
const when = (iso) => (iso ? new Date(iso).toLocaleString() : "—");

const EVENT_LABEL = {
  promoted: "Promoted",
  rolled_back: "Rolled back",
  promotion_declined: "Challenger declined",
  run_registered: "Contest run",
  run_imported: "Model imported",
};

function ContestTable({ run, championName }) {
  const rows = [...run.candidates].sort((a, b) => a.val.cost - b.val.cost);
  const worst = Math.max(...rows.map((r) => r.val.cost));
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Threshold</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>Validation cost</th>
            <th>Test cost</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => (
            <tr key={c.name}>
              <td>
                <span className="mono">{c.name}</span>
                {c.name === run.winner && <span className="pill pill-win">contest winner</span>}
                {c.name === championName && <span className="pill pill-champ">serving</span>}
                {!c.tree_based && <span className="pill pill-muted">not explainable</span>}
              </td>
              <td className="mono">{c.val.threshold?.toFixed(2)}</td>
              <td className="mono">{pct(c.val.precision)}</td>
              <td className="mono">{pct(c.val.recall)}</td>
              <td>
                <div className="cost-cell">
                  <span className="mono">{money(c.val.cost)}</span>
                  <span className="cost-bar" aria-hidden="true">
                    <span style={{ width: `${(c.val.cost / worst) * 100}%` }} />
                  </span>
                </div>
              </td>
              <td className="mono">{money(c.test?.cost)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function RegistryPage() {
  const [champion, setChampion] = useState(undefined);
  const [runs, setRuns] = useState(null);
  const [events, setEvents] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const [c, r, e] = await Promise.all([
        api.getRegistryChampion(),
        api.getRegistryRuns(),
        api.getRegistryEvents(),
      ]);
      setChampion(c.champion);
      setRuns(r.runs);
      setEvents(e.events);
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load the registry.");
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const latest = runs?.find((r) => r.candidates.length > 1) ?? runs?.[0];

  return (
    <>
      <header className="page-head">
        <h1 className="page-title">Model registry</h1>
        <p className="page-sub">
          Every candidate is trained and scored on the same data. The one with the lowest expected
          cost wins, and it only replaces the serving model if it clearly beats it.
        </p>
      </header>

      {error && <ErrorBanner message="Registry unavailable." detail={error} />}

      {runs && runs.length === 0 && (
        <div className="panel">
          <div className="empty-state">
            No contest has been run yet. Run <span className="mono">python -m ml.contest.run</span> to
            train all candidates and fill the registry.
          </div>
        </div>
      )}

      {champion && (
        <div className="stat-row">
          <div className="stat-card tone-good">
            <div className="stat-label">Serving model</div>
            <div className="stat-value mono">{champion.model_name}</div>
            <div className="stat-hint">Since {when(champion.promoted_at)}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Threshold</div>
            <div className="stat-value mono">{champion.threshold?.toFixed(2)}</div>
            <div className="stat-hint">Picked on validation</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Validation cost</div>
            <div className="stat-value mono">{money(champion.val_cost)}</div>
            <div className="stat-hint">Lower is better</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Contest runs</div>
            <div className="stat-value mono">{runs?.length ?? "—"}</div>
            <div className="stat-hint">Kept for rollback</div>
          </div>
        </div>
      )}

      <div className="stack">
        {latest && (
          <section className="panel">
            <div className="panel-header">
              <span className="panel-title">Latest contest</span>
              <span className="small text-tertiary mono">{latest.run_id}</span>
            </div>
            <div className="panel-body">
              <ContestTable run={latest} championName={champion?.model_name} />
              <p className="context-note" style={{ marginTop: 12 }}>
                Cost uses assumed figures, not measured ones: a flat cost per false alarm and a share of
                the amount lost per missed fraud. Test cost is shown for reference only and never
                influences the choice.
              </p>
            </div>
          </section>
        )}

        {events && events.length > 0 && (
          <section className="panel">
            <div className="panel-header">
              <span className="panel-title">History</span>
            </div>
            <ul className="timeline">
              {events.slice(0, 12).map((e, i) => (
                <li key={`${e.at}-${i}`} className={`timeline-item ev-${e.event}`}>
                  <div className="timeline-head">
                    <strong>{EVENT_LABEL[e.event] || e.event}</strong>
                    <span className="small text-tertiary">{when(e.at)}</span>
                  </div>
                  <div className="small text-secondary">
                    {e.reason || e.model_name || e.model || (e.winner ? `Winner: ${e.winner}` : e.run_id)}
                  </div>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </>
  );
}
