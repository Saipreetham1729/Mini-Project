import { useState } from "react";
import { Link } from "react-router-dom";
import ScoreBar from "../components/ScoreBar.jsx";
import { useExperiment } from "../context/ExperimentContext.jsx";
import { evaluate } from "../lib/api.js";

const METRICS = [
  { key: "rouge_l", label: "ROUGE-L", tone: "partial" },
  { key: "nli_entailment", label: "NLI entailment", tone: "supported" },
  { key: "llm_judge", label: "LLM judge", tone: "accent" },
  { key: "ragas_faithfulness", label: "RAGAS faithfulness", tone: "accent" },
  { key: "self_consistency", label: "Self-consistency", tone: "supported" },
];

export default function Evaluation() {
  const { docId, queryResult, evaluateResult, setEvaluateResult } = useExperiment();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleEvaluate() {
    setLoading(true);
    setError(null);
    try {
      const result = await evaluate({
        docId,
        question: queryResult.question,
        answer: queryResult.answer,
        retrievedChunks: queryResult.retrieved_chunks,
      });
      setEvaluateResult(result);
    } catch (err) {
      setError(err.message || "Evaluation failed.");
    } finally {
      setLoading(false);
    }
  }

  if (!queryResult) {
    return (
      <div>
        <div className="page-header">
          <div className="eyebrow">Comparing the methods</div>
          <h1>Evaluation</h1>
        </div>
        <div className="card">
          <p style={{ margin: 0 }}>
            No answer to evaluate yet. <Link to="/experiment">Run an experiment</Link> first.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <div className="eyebrow">Comparing the methods</div>
        <h1>Evaluation</h1>
        <p>
          Aggregate faithfulness score per method. A method that couldn't run reports that plainly
          instead of a guessed number.
        </p>
      </div>

      {!evaluateResult && (
        <div className="card">
          {error && <p className="error-text">{error}</p>}
          <button className="btn" onClick={handleEvaluate} disabled={loading}>
            {loading && <span className="spinner" aria-hidden="true" />}
            {loading ? "Running 6 checks -- this takes longer than the other steps" : "Run evaluation"}
          </button>
        </div>
      )}

      {evaluateResult && (
        <div className="results-reveal">
          <div className="delta-row">
            <h3 style={{ margin: 0 }}>Results</h3>
            <button type="button" className="btn secondary" onClick={() => window.print()}>
              Print / export
            </button>
          </div>
          <div className="card section-gap">
            {METRICS.map((m) => (
              <ScoreBar
                key={m.key}
                label={m.label}
                score={evaluateResult[m.key].aggregate_score}
                tone={m.tone}
              />
            ))}
          </div>

          {METRICS.some((m) => evaluateResult[m.key].error) && (
            <div className="card section-gap">
              <h3>Notes</h3>
              {METRICS.filter((m) => evaluateResult[m.key].error).map((m) => (
                <p key={m.key} className="reasoning">
                  <strong>{m.label}:</strong> {evaluateResult[m.key].error}
                </p>
              ))}
            </div>
          )}

          <div className="card section-gap">
            <h3>Per-claim scores</h3>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Claim</th>
                  {METRICS.map((m) => (
                    <th key={m.key}>{m.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {evaluateResult.rouge_l.per_claim.map((row) => {
                  const nli = evaluateResult.nli_entailment.per_claim.find(
                    (r) => r.claim_id === row.claim_id
                  );
                  const judge = evaluateResult.llm_judge.per_claim.find(
                    (r) => r.claim_id === row.claim_id
                  );
                  const ragas = evaluateResult.ragas_faithfulness.per_claim.find(
                    (r) => r.claim_id === row.claim_id
                  );
                  const selfConsistency = evaluateResult.self_consistency.per_claim.find(
                    (r) => r.claim_id === row.claim_id
                  );
                  return (
                    <tr key={row.claim_id}>
                      <td className="mono">{row.claim_id}</td>
                      <td>{row.score.toFixed(2)}</td>
                      <td>
                        {nli && nli.score !== null ? (
                          nli.score.toFixed(2)
                        ) : (
                          <span className="not-evaluated">--</span>
                        )}
                      </td>
                      <td>
                        {judge && judge.score !== null ? (
                          judge.score.toFixed(2)
                        ) : (
                          <span className="not-evaluated">--</span>
                        )}
                      </td>
                      <td>
                        {ragas && ragas.score !== null ? (
                          ragas.score.toFixed(2)
                        ) : (
                          <span className="not-evaluated">--</span>
                        )}
                      </td>
                      <td>
                        {selfConsistency && selfConsistency.score !== null ? (
                          selfConsistency.score.toFixed(2)
                        ) : (
                          <span className="not-evaluated">--</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
