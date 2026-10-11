import { createContext, useContext, useState } from "react";

// Holds the results of the last experiment run so Verification, Mitigation,
// and Evaluation (separate routes) can all read the same doc_id / answer /
// claims without re-fetching or prop-drilling through App.jsx.
const ExperimentContext = createContext(null);

export function ExperimentProvider({ children }) {
  const [docId, setDocId] = useState(null);
  const [docName, setDocName] = useState(null);
  const [queryResult, setQueryResult] = useState(null);
  const [verifyResult, setVerifyResult] = useState(null);
  const [mitigateResult, setMitigateResult] = useState(null);
  const [evaluateResult, setEvaluateResult] = useState(null);

  const value = {
    docId,
    setDocId,
    docName,
    setDocName,
    queryResult,
    setQueryResult,
    verifyResult,
    setVerifyResult,
    mitigateResult,
    setMitigateResult,
    evaluateResult,
    setEvaluateResult,
  };

  return <ExperimentContext.Provider value={value}>{children}</ExperimentContext.Provider>;
}

export function useExperiment() {
  const ctx = useContext(ExperimentContext);
  if (!ctx) {
    throw new Error("useExperiment must be used within an ExperimentProvider");
  }
  return ctx;
}
