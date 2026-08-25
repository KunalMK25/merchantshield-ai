export default function ErrorBanner({ message, detail }) {
  return (
    <div
      className="error-banner"
      role="alert"
      aria-live="assertive"
    >
      <span aria-hidden="true" style={{ fontSize: 16, flexShrink: 0, marginTop: 1 }}>⚠</span>
      <div>
        <strong>{message}</strong>
        {detail && (
          <div style={{ marginTop: 4, fontSize: 12, opacity: 0.8 }}>{detail}</div>
        )}
      </div>
    </div>
  );
}
