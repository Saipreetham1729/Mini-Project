const DEFAULT_COLOR = "var(--accent)";

/**
 * data: { label: count } -- e.g. { supported: 12, unsupported: 4 }
 * colors: optional { label: cssColorVar } override per bar
 * Bars scale relative to the max value in this dataset, not a fixed 0-1
 * range (unlike ScoreBar, which is for 0-1 metric scores).
 */
export default function CountBarChart({ data, colors = {} }) {
  const entries = Object.entries(data || {});
  const max = Math.max(1, ...entries.map(([, v]) => v));

  if (entries.length === 0) {
    return <p className="not-evaluated">No data recorded yet.</p>;
  }

  return (
    <div>
      {entries.map(([label, value]) => (
        <div key={label} className="score-bar-row">
          <div className="score-bar-label">
            <span>{label.replaceAll("_", " ")}</span>
            <span className="value">{value}</span>
          </div>
          <div className="score-bar-track">
            <div
              className="score-bar-fill"
              style={{
                width: `${Math.round((value / max) * 100)}%`,
                background: colors[label] || DEFAULT_COLOR,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
