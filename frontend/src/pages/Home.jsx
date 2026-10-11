import PipelineSteps from "../components/PipelineSteps.jsx";

const METHODS = [
  { tone: "supported", name: "NLI entailment", note: "does the evidence logically entail the claim" },
  { tone: "partial", name: "ROUGE-L overlap", note: "lexical overlap between claim and evidence" },
  { tone: "unsupported", name: "LLM-as-judge", note: "a second model scores support directly" },
];

export default function Home() {
  return (
    <div>
      <div className="hero-grid">
        <div>
          <div className="eyebrow">Research project</div>
          <h1 className="hero-title">Evaluating hallucination detection in RAG</h1>
          <p>
            ContextGuard is a controlled experimental environment for comparing how well
            different faithfulness metrics catch hallucinations in retrieval-augmented
            generation — not a general-purpose document chatbot.
          </p>
        </div>
        <div className="board">
          <div className="board-label">pipeline trace</div>
          <PipelineSteps activeIndex={-1} />
          <div className="methods-list">
            {METHODS.map((m) => (
              <div className="method-row" key={m.name}>
                <span className={`chip ${m.tone === "unsupported" ? "unsupported" : m.tone === "partial" ? "partially_supported" : "supported"}`} />
                <div>
                  <div className="method-name">{m.name}</div>
                  <div className="method-note">{m.note}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card section-gap">
        <h2>The problem</h2>
        <p>
          RAG systems reduce hallucination by grounding answers in retrieved evidence, but they
          don't eliminate it. A generated answer can still introduce claims that aren't actually
          supported by what was retrieved — and it isn't obvious, from the answer text alone,
          which claims are trustworthy.
        </p>
      </div>

      <div className="card section-gap">
        <h2>The approach</h2>
        <p>
          Every answer is broken into individual claims, and each claim is checked against the
          retrieved evidence using several independent methods. Comparing how these methods
          agree and disagree with human judgment is the actual research question.
        </p>
      </div>
    </div>
  );
}
