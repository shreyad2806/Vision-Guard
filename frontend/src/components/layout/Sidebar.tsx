import { NavLink } from "react-router-dom";
import { ShieldCheck, LayoutDashboard, History } from "lucide-react";

export default function Sidebar() {
  return (
    <aside className="app-sidebar">
      <div className="sidebar__brand">
        <div className="sidebar__brand-icon">
          <ShieldCheck size={18} />
        </div>
        <div className="sidebar__brand-text">
          <span className="sidebar__brand-name">VisionGuard</span>
          <span className="sidebar__brand-subtitle">
            Visual Data Quality Intelligence
          </span>
        </div>
      </div>

      <nav className="sidebar__nav">
        <span className="sidebar__section-label">Navigation</span>
        <NavLink
          to="/"
          end
          className={({ isActive }) =>
            `sidebar__link ${isActive ? "sidebar__link--active" : ""}`
          }
        >
          <LayoutDashboard size={16} />
          Dashboard
        </NavLink>
        <NavLink
          to="/history"
          className={({ isActive }) =>
            `sidebar__link ${isActive ? "sidebar__link--active" : ""}`
          }
        >
          <History size={16} />
          Analysis History
        </NavLink>

        <span className="sidebar__section-label">System</span>
        <span className="sidebar__link" style={{ cursor: "default" }}>
          <ShieldCheck size={16} />
          ML Pipeline
        </span>
      </nav>

      <div className="sidebar__footer">
        <div className="sidebar__status">
          <span className="sidebar__status-dot" />
          <span className="sidebar__status-text">System Ready</span>
        </div>
      </div>
    </aside>
  );
}
