import { useState } from "react";
import { Link } from "react-router-dom";
import { useExperiment } from "../context/ExperimentContext.jsx";
import { mitigate } from "../lib/api.js";

const STRATEGY_LABELS = {
  retrieve_additional_chunks: "Retrieve additional chunks",
  expand_evidence_context: "Expand evidence context",
  stricter_grounded_prompt: "Stricter grounded prompt",
  require_claim_support: "Require claim support",
  surface_conflicts: "Surface conflicts",
};

function SupportSummaryRow({ label, summary }) {
  return (
    <div className="stat">
      <span className="num">{(summary.support_ratio * 100).toFixed(0)}%</span>
      <span className="label">
        {label} -- {summary.supported_count} supported, {summary.partially_supported_count} partial,{" "}
        {summary.unsupported_count} unsupported
      </span>
    </div>
  );
}

function DeltaBadge({ before, after }) {
  const delta = Math.round((after - before) * 100);
  const cls = delta > 0 ? "up" : delta < 0 ? "down" : "flat";
  const sign = delta > 0 ? "+" : "";
  return <span className={`delta-badge ${cls}`}>{sign}{delta}pp support ratio</span>;
}

export default function Mitigation() {
  const { docId, queryResult, verifyResult, mitigateResult, setMitigateResult } = useExperiment();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const needsMitigation =
    verifyResult && (verifyResult.unsupported_count > 0 || verifyResult.partially_supported_count > 0);

  async function handleMitigate() {
    setLoading(true);
    setError(null);
    try {
      const result = await mitigate({
        docId,
        question: queryResult.question,
        originalAnswer: queryResult.answer,
        claims: verifyResult.claims,
      });
      setMitigateResult(result);
    } catch (err) {
      setError(err.message || "Mitigation failed.");
    } finally {
      setLoading(false);
    }
  }

  if (!verifyResult) {
    return (
      <div>
        <div className="page-header">
          <div className="eyebrow">Fixing what's unsupported</div>
          <h1>Mitigation</h1>
        </div>
        <div className="card">
          <p style={{ margin: 0 }}>
            No verification results yet. <Link to="/verification">Verify an answer</Link> first.
          </p>
        </div>
      </div>
    );
  }

  if (!needsMitigation && !mitigateResult) {
    return (
      <div>
        <div className="page-header">
          <div className="eyebrow">Fixing what's unsupported</div>
          <h1>Mitigation</h1>
        </div>
        <div className="card">
          <p style={{ margin: 0 }}>Every claim was already fully supported -- nothing to mitigate.</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <div className="eyebrow">Fixing what's unsupported</div>
        <h1>Mitigation</h1>
        <p>Some claims weren't fully supported. Apply a matched mitigation strategy and compare.</p>
      </div>

      {!mitigateResult && (
        <div className="card">
          {error && <p className="error-text">{error}</p>}
          <button className="btn" onClick={handleMitigate} disabled={loading}>
            {loading && <span className="spinner" aria-hidden="true" />}
            {loading ? "Applying" : "Apply mitigation"}
          </button>
        </div>
      )}

      {mitigateResult && (
        <div className="results-reveal">
          <p>
            Detected problem: <strong>{mitigateResult.dominant_failure_type?.replaceAll("_", " ")}</strong> --
            strategy applied:{" "}
            <strong>{STRATEGY_LABELS[mitigateResult.strategy_used] || mitigateResult.strategy_used}</strong>
          </p>

          <div className="two-col">
            <div className="card">
              <h3>Original answer</h3>
              <p style={{ color: "var(--text)" }}>{mitigateResult.original_answer}</p>
            </div>
            <div className="card">
              <h3>Revised answer</h3>
              <p style={{ color: "var(--text)" }}>{mitigateResult.revised_answer}</p>
            </div>
          </div>

          <div className="card section-gap">
            <div className="delta-row">
              <h3 style={{ margin: 0 }}>Before vs. after</h3>
              <DeltaBadge before={mitigateResult.before.support_ratio} after={mitigateResult.after.support_ratio} />
            </div>
            <div className="two-col section-gap">
              <SupportSummaryRow label="Before" summary={mitigateResult.before} />
              <SupportSummaryRow label="After" summary={mitigateResult.after} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
