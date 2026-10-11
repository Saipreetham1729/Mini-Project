const COLORS = {
  supported: "var(--supported)",
  partial: "var(--partial)",
  unsupported: "var(--unsupported)",
  accent: "var(--accent)",
};

/**
 * score: 0-1 or null. Per the project's "never fabricate a result" rule,
 * a null score renders as "Not evaluated yet" instead of an empty/zero bar,
 * which would misleadingly look like a real measured zero.
 */
export default function ScoreBar({ label, score, tone = "accent" }) {
  return (
    <div className="score-bar-row">
      <div className="score-bar-label">
        <span>{label}</span>
        {score === null || score === undefined ? (
          <span className="value not-evaluated">Not evaluated yet</span>
        ) : (
          <span className="value">{score.toFixed(2)}</span>
        )}
      </div>
      <div className="score-bar-track">
        {score !== null && score !== undefined && (
          <div
            className="score-bar-fill"
            style={{ width: `${Math.round(score * 100)}%`, background: COLORS[tone] }}
          />
        )}
      </div>
    </div>
  );
}
