import { NavLink } from "react-router-dom";
import { LayoutDashboard, History } from "lucide-react";

export default function Header() {
  return (
    <header className="app-header">
      <nav className="app-header__nav">
        <NavLink
          to="/"
          end
          className={({ isActive }) =>
            `app-header__link ${isActive ? "app-header__link--active" : ""}`
          }
        >
          <LayoutDashboard size={15} />
          Dashboard
        </NavLink>
        <NavLink
          to="/history"
          className={({ isActive }) =>
            `app-header__link ${isActive ? "app-header__link--active" : ""}`
          }
        >
          <History size={15} />
          Analysis History
        </NavLink>
      </nav>

      <div className="app-header__right">
        <div className="app-header__model-pill">Model: Ready</div>
        <div className="app-header__status-pill">
          <span className="app-header__status-dot" />
          System Ready
        </div>
      </div>
    </header>
  );
}
