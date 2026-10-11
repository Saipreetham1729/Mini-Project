import { useEffect, useState } from "react";
import PipelineSteps from "../components/PipelineSteps.jsx";
import { useExperiment } from "../context/ExperimentContext.jsx";
import { uploadDocument, processDocument, embedDocument, runQuery } from "../lib/api.js";

const STEPS = ["Upload", "Process", "Embed", "Retrieve & generate"];

export default function Experiment() {
  const { setDocId, setDocName, queryResult, setQueryResult } = useExperiment();
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [activeStep, setActiveStep] = useState(-1);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [sampleQuestions, setSampleQuestions] = useState([]);
  const [loadingSample, setLoadingSample] = useState(false);

  useEffect(() => {
    fetch("/sample/sample_questions.json")
      .then((res) => (res.ok ? res.json() : []))
      .then(setSampleQuestions)
      .catch(() => setSampleQuestions([]));
  }, []);

  async function handleUseSample(sampleQuestion) {
    setError(null);
    setLoadingSample(true);
    try {
      const res = await fetch("/sample/rag_overview.pdf");
      if (!res.ok) throw new Error("Sample document not found.");
      const blob = await res.blob();
      setFile(new File([blob], "rag_overview.pdf", { type: "application/pdf" }));
      setQuestion(sampleQuestion);
    } catch (err) {
      setError(err.message || "Could not load the sample document.");
    } finally {
      setLoadingSample(false);
    }
  }

  async function handleRun(e) {
    e.preventDefault();
    if (!file) {
      setError("Choose a PDF first.");
      return;
    }
    if (!question.trim()) {
      setError("Enter a question.");
      return;
    }

    setError(null);
    setRunning(true);
    setQueryResult(null);

    try {
      setActiveStep(0);
      const uploadRes = await uploadDocument(file);
      setDocId(uploadRes.doc_id);
      setDocName(uploadRes.filename);

      setActiveStep(1);
      await processDocument(uploadRes.doc_id);

      setActiveStep(2);
      await embedDocument(uploadRes.doc_id);

      setActiveStep(3);
      const result = await runQuery(uploadRes.doc_id, question);
      setQueryResult(result);
      setActiveStep(4);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div>
      <div className="page-header">
        <div className="eyebrow">Run an experiment</div>
        <h1>Ask a question</h1>
        <p>Upload a document, ask a question about it, and run it through the pipeline.</p>
      </div>

      {sampleQuestions.length > 0 && (
        <div className="card sample-card">
          <h3>Try a sample -- no upload needed</h3>
          <p>
            Loads a bundled 3-page document about RAG and hallucination detection, plus one of
            the questions below. Useful for a live demo where you don't want to depend on a file
            upload working perfectly in front of an audience.
          </p>
          <div className="sample-chips">
            {sampleQuestions.map((sq) => (
              <button
                key={sq.question}
                type="button"
                className="sample-chip"
                disabled={loadingSample}
                onClick={() => handleUseSample(sq.question)}
              >
                {sq.question}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="card section-gap">
        <form onSubmit={handleRun}>
          <div className="field">
            <label htmlFor="doc-upload">Document (PDF)</label>
            <label className={`dropzone${file ? " has-file" : ""}`} htmlFor="doc-upload">
              <span className="dropzone-icon" aria-hidden="true" />
              <span className="dropzone-text">
                {file ? file.name : "Choose a PDF, or drop one here"}
              </span>
              <input
                id="doc-upload"
                type="file"
                accept="application/pdf"
                onChange={(e) => setFile(e.target.files[0] || null)}
              />
            </label>
          </div>
          <div className="field">
            <label htmlFor="question">Question</label>
            <textarea
              id="question"
              placeholder="e.g. How does RAG reduce hallucinations?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
          </div>
          {error && <p className="error-text">{error}</p>}
          <button type="submit" className="btn" disabled={running}>
            {running && <span className="spinner" aria-hidden="true" />}
            {running ? "Running" : "Run experiment"}
          </button>
        </form>
      </div>

      {activeStep >= 0 && (
        <div className="section-gap">
          <PipelineSteps steps={STEPS} activeIndex={running ? activeStep : STEPS.length} />
        </div>
      )}

      {queryResult && (
        <div className="section-gap">
          <div className="card">
            <h3>Generated answer</h3>
            <p style={{ color: "var(--text)" }}>{queryResult.answer}</p>
            <p className="mono">latency {queryResult.latency_seconds}s</p>
          </div>

          <div className="card section-gap">
            <h3>Retrieved evidence</h3>
            {queryResult.retrieved_chunks.map((chunk) => (
              <div key={chunk.chunk_id} className="claim-row">
                <div className="claim-meta">
                  <span className="mono">Page {chunk.page_number}</span>
                  <span className="mono">score {chunk.score.toFixed(2)}</span>
                </div>
                <p style={{ margin: 0, color: "var(--text)" }}>{chunk.text}</p>
              </div>
            ))}
          </div>

          <p className="reasoning section-gap">
            Continue to <strong>Verification</strong> to check each claim against this evidence.
          </p>
        </div>
      )}
    </div>
  );
}
