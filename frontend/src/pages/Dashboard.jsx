import { useEffect, useState } from "react";
import CountBarChart from "../components/CountBarChart.jsx";
import { getDashboardSummary } from "../lib/api.js";

const STATUS_COLORS = {
  supported: "var(--supported)",
  partially_supported: "var(--partial)",
  unsupported: "var(--unsupported)",
};

const METHOD_LABELS = {
  baseline_embedding_similarity: "Baseline (embedding similarity)",
  rouge_l: "ROUGE-L",
  nli_entailment: "NLI entailment",
  llm_judge: "LLM judge",
  ragas_faithfulness: "RAGAS faithfulness",
  self_consistency: "Self-consistency",
};

export default function Dashboard() {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getDashboardSummary()
      .then(setSummary)
      .catch((err) => setError(err.message || "Could not load dashboard data."))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <div className="page-header">
        <div className="eyebrow">Across all benchmark runs</div>
        <h1>Research dashboard</h1>
        <p>
          Aggregated results from every question run through{" "}
          <code className="mono">/benchmark/questions/{"{id}"}/run</code>. Curating questions and
          labeling claims still happens via the API docs (<code className="mono">/docs</code>) --
          this page only visualizes what's already been recorded and labeled.
        </p>
      </div>

      {loading && <p className="not-evaluated">Loading…</p>}
      {error && <p className="error-text">{error}</p>}

      {summary && summary.total_claims === 0 && (
        <div className="card">
          <p style={{ margin: 0 }}>
            No benchmark data recorded yet. Add questions and run them via{" "}
            <code className="mono">POST /benchmark/questions</code> and{" "}
            <code className="mono">POST /benchmark/questions/{"{id}"}/run</code>, then label the
            resulting claims before this dashboard has anything to show.
          </p>
        </div>
      )}

      {summary && summary.total_claims > 0 && (
        <div className="results-reveal">
          <div className="delta-row">
            <h3 style={{ margin: 0 }}>Results</h3>
            <button type="button" className="btn secondary" onClick={() => window.print()}>
              Print / export
            </button>
          </div>
          <div className="card section-gap">
            <div className="readouts">
              <div className="stat">
                <span className="num">{summary.total_questions}</span>
                <span className="label">Benchmark questions</span>
              </div>
              <div className="stat">
                <span className="num">{summary.total_claims}</span>
                <span className="label">Claims recorded</span>
              </div>
              <div className="stat">
                <span className="num">{summary.labeled_claims}</span>
                <span className="label">Labeled</span>
              </div>
              <div className="stat">
                <span className="num">{summary.unlabeled_claims}</span>
                <span className="label">Awaiting label</span>
              </div>
            </div>
            {summary.avg_latency_seconds !== null && (
              <p className="reasoning">Average pipeline run latency: {summary.avg_latency_seconds}s</p>
            )}
          </div>

          <div className="card section-gap">
            <h3>Baseline status distribution</h3>
            <CountBarChart data={summary.status_counts} colors={STATUS_COLORS} />
          </div>

          <div className="card section-gap">
            <h3>Likely failure types</h3>
            <CountBarChart data={summary.failure_type_counts} colors={{}} />
          </div>

          <div className="card section-gap">
            <h3>Method comparison (Precision / Recall / F1)</h3>
            {summary.labeled_claims === 0 ? (
              <p className="not-evaluated">
                Not evaluated yet -- no claims have been labeled. Label some via{" "}
                <code className="mono">POST /benchmark/claims/{"{id}"}/label</code> to see this
                comparison.
              </p>
            ) : (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Method</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>Labeled examples</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(summary.prf_by_method).map(([key, prf]) => (
                    <tr key={key}>
                      <td>{METHOD_LABELS[key] || key}</td>
                      <td>
                        {prf.precision !== null ? prf.precision.toFixed(2) : <span className="not-evaluated">--</span>}
                      </td>
                      <td>
                        {prf.recall !== null ? prf.recall.toFixed(2) : <span className="not-evaluated">--</span>}
                      </td>
                      <td>
                        {prf.f1 !== null ? prf.f1.toFixed(2) : <span className="not-evaluated">--</span>}
                      </td>
                      <td>{prf.labeled_examples}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
