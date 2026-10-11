const DEFAULT_STEPS = ["Upload", "Retrieve", "Generate", "Verify", "Mitigate", "Evaluate"];

/**
 * activeIndex: index of the currently-running step (0-based). Steps before
 * it render as "done", the step itself as "active", the rest as pending.
 */
export default function PipelineSteps({ steps = DEFAULT_STEPS, activeIndex = -1 }) {
  return (
    <div className="pipeline-steps">
      {steps.map((step, i) => (
        <div key={step} style={{ display: "flex", alignItems: "center" }}>
          <div
            className={`pipeline-step ${
              i === activeIndex ? "active" : i < activeIndex ? "done" : ""
            }`}
          >
            {step}
          </div>
          {i < steps.length - 1 && <span className="pipeline-arrow" aria-hidden="true" />}
        </div>
      ))}
    </div>
  );
}
