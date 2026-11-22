const ICONS = {
  evaluate: "M12 3l8 3v6c0 4.4-3.2 8.3-8 9-4.8-.7-8-4.6-8-9V6l8-3z M9 12l2 2 4-4",
  monitor: "M3 17l5-6 4 3 5-8 4 5 M3 21h18",
};

function Icon({ name }) {
  return (
    <svg className="nav-icon" viewBox="0 0 24 24" aria-hidden="true" fill="none"
         stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d={ICONS[name]} />
    </svg>
  );
}

const NAV = [
  { id: "evaluate", label: "Evaluate", icon: "evaluate" },
  { id: "monitor", label: "Model & audit", icon: "monitor" },
];

export default function Sidebar({ view, onNavigate, theme, onToggleTheme, status }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor"
               strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 3l8 3v6c0 4.4-3.2 8.3-8 9-4.8-.7-8-4.6-8-9V6l8-3z" />
          </svg>
        </div>
        <div>
          <div className="brand-name">MerchantShield</div>
          <div className="brand-sub">Risk console</div>
        </div>
      </div>

      <nav className="nav" aria-label="Main">
        {NAV.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`nav-item${view === item.id ? " active" : ""}`}
            aria-current={view === item.id ? "page" : undefined}
            onClick={() => onNavigate(item.id)}
          >
            <Icon name={item.icon} />
            {item.label}
          </button>
        ))}
      </nav>

      <div className="sidebar-foot">
        <div className={`sys-status ${status.tone}`} role="status">
          <span className="sys-dot" aria-hidden="true" />
          {status.label}
        </div>
        <div className="demo-chip">Demo · synthetic data</div>
        <button type="button" className="theme-toggle" onClick={onToggleTheme}
                aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
          {theme === "dark" ? "☀ Light theme" : "☾ Dark theme"}
        </button>
      </div>
    </aside>
  );
}
