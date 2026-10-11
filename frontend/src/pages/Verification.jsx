import { useState } from "react";
import { Link } from "react-router-dom";
import StatusChip from "../components/StatusChip.jsx";
import { useExperiment } from "../context/ExperimentContext.jsx";
import { verifyAnswer } from "../lib/api.js";

export default function Verification() {
  const { docId, queryResult, verifyResult, setVerifyResult } = useExperiment();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleVerify() {
    setLoading(true);
    setError(null);
    try {
      const result = await verifyAnswer({
        docId,
        question: queryResult.question,
        answer: queryResult.answer,
        retrievedChunks: queryResult.retrieved_chunks,
      });
      setVerifyResult(result);
    } catch (err) {
      setError(err.message || "Verification failed.");
    } finally {
      setLoading(false);
    }
  }

  if (!queryResult) {
    return (
      <div>
        <div className="page-header">
          <div className="eyebrow">Claim-level check</div>
          <h1>Verification</h1>
        </div>
        <div className="card">
          <p style={{ margin: 0 }}>
            No answer to verify yet. <Link to="/experiment">Run an experiment</Link> first.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <div className="eyebrow">Claim-level check</div>
        <h1>Verification</h1>
        <p>Each claim in the answer, checked against the retrieved evidence individually.</p>
      </div>

      {!verifyResult && (
        <div className="card">
          <p>Ready to verify the answer from your last experiment.</p>
          {error && <p className="error-text">{error}</p>}
          <button className="btn" onClick={handleVerify} disabled={loading}>
            {loading ? "Verifying..." : "Verify claims"}
          </button>
        </div>
      )}

      {verifyResult && (
        <div className="results-reveal">
          <div className="card">
            <div className="readouts">
              <div className="stat">
                <span className="num">{verifyResult.supported_count}</span>
                <span className="label">Supported</span>
              </div>
              <div className="stat">
                <span className="num">{verifyResult.partially_supported_count}</span>
                <span className="label">Partially supported</span>
              </div>
              <div className="stat">
                <span className="num">{verifyResult.unsupported_count}</span>
                <span className="label">Unsupported</span>
              </div>
              <div className="stat">
                <span className="num">{(verifyResult.support_ratio * 100).toFixed(0)}%</span>
                <span className="label">Support ratio</span>
              </div>
            </div>
          </div>

          <div className="card section-gap">
            <h3>Claims</h3>
            {verifyResult.claims.map((claim) => (
              <div key={claim.claim_id} className={`claim-row status-${claim.status}`}>
                <div className="claim-meta">
                  <StatusChip status={claim.status} />
                  <span className="mono">similarity {claim.similarity_score.toFixed(2)}</span>
                  {claim.best_evidence_page && <span className="mono">page {claim.best_evidence_page}</span>}
                </div>
                <p style={{ margin: 0, color: "var(--text)" }}>{claim.text}</p>
                {claim.likely_failure_type && (
                  <p className="reasoning">
                    Likely failure: <strong>{claim.likely_failure_type.replaceAll("_", " ")}</strong>
                    {" -- "}
                    {claim.failure_reasoning}
                  </p>
                )}
              </div>
            ))}
          </div>

          <p className="reasoning section-gap">
            Continue to <strong>Mitigation</strong> to attempt fixing any unsupported claims.
          </p>
        </div>
      )}
    </div>
  );
}
