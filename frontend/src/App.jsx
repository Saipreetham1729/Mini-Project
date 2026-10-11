import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar.jsx";
import StatusBanner from "./components/StatusBanner.jsx";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import Home from "./pages/Home.jsx";
import Experiment from "./pages/Experiment.jsx";
import Verification from "./pages/Verification.jsx";
import Mitigation from "./pages/Mitigation.jsx";
import Evaluation from "./pages/Evaluation.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Methodology from "./pages/Methodology.jsx";

export default function App() {
  return (
    <div className="app-shell">
      <Sidebar />
      <main className="content">
        <StatusBanner />
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/experiment" element={<Experiment />} />
            <Route path="/verification" element={<Verification />} />
            <Route path="/mitigation" element={<Mitigation />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/methodology" element={<Methodology />} />
          </Routes>
        </ErrorBoundary>
      </main>
    </div>
  );
}
