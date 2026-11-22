export default function StatCard({ label, value, hint, tone }) {
  return (
    <div className={`stat-card${tone ? ` tone-${tone}` : ""}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value mono">{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}
