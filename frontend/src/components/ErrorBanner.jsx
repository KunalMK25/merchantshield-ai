export default function ErrorBanner({ message, detail }) {
  if (!message) return null;
  return (
    <div className="error-banner">
      <span aria-hidden="true">⚠</span>
      <span>
        {message}
        {detail && typeof detail === "string" && <div className="small" style={{ marginTop: 4 }}>{detail}</div>}
      </span>
    </div>
  );
}
