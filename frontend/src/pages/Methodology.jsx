const METHODS = [
  {
    name: "Baseline (embedding similarity)",
    checkedAgainst: "Retrieved evidence",
    how: "Cosine similarity between the claim and its best-matching retrieved chunk.",
    cost: "Cheapest",
  },
  {
    name: "ROUGE-L",
    checkedAgainst: "Retrieved evidence",
    how: "Lexical overlap (longest common subsequence F1) between claim and evidence.",
    cost: "Cheap, pure Python",
  },
  {
    name: "NLI entailment",
    checkedAgainst: "Retrieved evidence",
    how: "A DeBERTa cross-encoder estimates the probability the evidence entails the claim.",
    cost: "Moderate (local model)",
  },
  {
    name: "LLM-judge",
    checkedAgainst: "Retrieved evidence",
    how: "An LLM verdicts every claim in one call: supported / partially supported / unsupported.",
    cost: "API cost, expensive",
  },
  {
    name: "RAGAS faithfulness",
    checkedAgainst: "Retrieved evidence",
    how: "An LLM gives a strictly binary yes/no per claim, following RAGAS's own methodology.",
    cost: "API cost, expensive",
  },
  {
    name: "Self-consistency",
    checkedAgainst: "The model's own sampling variance",
    how: "The question is re-answered independently N times; each claim's score is its average lexical overlap against those samples -- no retrieved evidence needed at all.",
    cost: "Highest (N extra LLM calls)",
  },
];

const FAILURE_TYPES = [
  { name: "Retrieval failure", note: "Nothing relevant was retrieved for the question at all." },
  { name: "Insufficient context", note: "Retrieval found something relevant, but not enough to fully answer." },
  { name: "Context misinterpretation", note: "Relevant evidence was retrieved, but the answer drew the wrong conclusion from it." },
  { name: "Unsupported generation", note: "The model introduced a claim with no basis in the retrieved evidence at all." },
  { name: "Conflicting evidence", note: "Retrieved chunks disagree with each other; embedding divergence flags this, but doesn't prove contradiction." },
];

export default function Methodology() {
  return (
    <div>
      <div className="page-header">
        <div className="eyebrow">Research methodology</div>
        <h1>How this evaluation works</h1>
        <p>
          The research question: <em>which faithfulness/grounding metrics best detect actual
          hallucinations in RAG-generated answers?</em> Everything below exists to answer that,
          not to build a general-purpose chatbot.
        </p>
      </div>

      <div className="card">
        <h2>Claim-level checking</h2>
        <p>
          Every generated answer is broken into individual, checkable claims. Each claim is then
          scored independently by six methods -- five check it against the retrieved evidence,
          one checks it against the model's own sampling behavior instead.
        </p>
      </div>

      <div className="card section-gap">
        <h2>The six methods</h2>
        <table className="data-table">
          <thead>
            <tr>
              <th>Method</th>
              <th>Checked against</th>
              <th>How it works</th>
              <th>Cost</th>
            </tr>
          </thead>
          <tbody>
            {METHODS.map((m) => (
              <tr key={m.name}>
                <td>{m.name}</td>
                <td className="mono">{m.checkedAgainst}</td>
                <td style={{ color: "var(--text-muted)" }}>{m.how}</td>
                <td className="mono">{m.cost}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="reasoning section-gap">
          Comparing how these six agree and disagree with human judgment (Precision / Recall /
          F1, on the Dashboard page) is the actual research contribution -- not any single
          method's raw accuracy.
        </p>
      </div>

      <div className="card section-gap">
        <h2>Likely failure types</h2>
        <p>
          For claims that aren't fully supported, a likely failure type is suggested from
          pipeline signals only (retrieval scores, claim-evidence similarity, inter-chunk
          divergence) -- always framed as "likely," never claimed as the verified internal cause.
        </p>
        <table className="data-table">
          <thead>
            <tr>
              <th>Failure type</th>
              <th>What it means</th>
            </tr>
          </thead>
          <tbody>
            {FAILURE_TYPES.map((f) => (
              <tr key={f.name}>
                <td>{f.name}</td>
                <td style={{ color: "var(--text-muted)" }}>{f.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card section-gap">
        <h2>Mitigation</h2>
        <p>
          A matched mitigation strategy is applied per detected failure type -- retrieving more
          context, a stricter evidence-grounded prompt, a claim-support-required prompt, or a
          conflict-aware prompt -- and before/after support ratios are compared on the Mitigation
          page.
        </p>
      </div>

      <div className="card section-gap">
        <h2>No fabricated results</h2>
        <p>
          Any metric that can't run -- no network, unparseable model output, no labeled examples
          yet -- reports "Not evaluated yet" rather than a guessed number. This is enforced in
          code, not just stated in documentation: every metric module returns <code className="mono">None</code>{" "}
          for a claim it couldn't score, and the UI renders that as <code className="mono">--</code>, never a zero.
        </p>
      </div>
    </div>
  );
}
