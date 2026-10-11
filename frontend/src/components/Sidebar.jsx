import { NavLink } from "react-router-dom";

const PIPELINE_LINKS = [
  { to: "/", label: "Home", end: true },
  { to: "/experiment", label: "Experiment" },
  { to: "/verification", label: "Verification" },
  { to: "/mitigation", label: "Mitigation" },
  { to: "/evaluation", label: "Evaluation" },
  { to: "/dashboard", label: "Dashboard" },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="wordmark">
        <span className="dot" aria-hidden="true" />
        ContextGuard
      </div>
      <nav className="nav">
        {PIPELINE_LINKS.map((link, i) => (
          <NavLink
            key={link.to}
            to={link.to}
            end={link.end}
            data-idx={String(i + 1).padStart(2, "0")}
            className={({ isActive }) => (isActive ? "active" : "")}
          >
            {link.label}
          </NavLink>
        ))}
      </nav>
      <nav className="nav nav-secondary">
        <NavLink to="/methodology" className={({ isActive }) => (isActive ? "active" : "")}>
          Methodology
        </NavLink>
      </nav>
    </aside>
  );
}
