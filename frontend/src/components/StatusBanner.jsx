import { useEffect, useState } from "react";
import { getHealth } from "../lib/api.js";

export default function StatusBanner() {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="status-banner status-banner-bad">
        <span className="chip unsupported" /> {error}
      </div>
    );
  }

  if (health && health.is_mock) {
    return (
      <div className="status-banner status-banner-warn">
        <span className="chip partially_supported" /> Running in MOCK mode -- answers are
        placeholders, not real model output. Set <code className="mono">LLM_PROVIDER=openai</code>{" "}
        and <code className="mono">LLM_API_KEY</code> in <code className="mono">backend/.env</code> for
        real results.
      </div>
    );
  }

  if (health) {
    return (
      <div className="status-banner status-banner-ok">
        <span className="chip supported" /> Connected -- {health.llm_model_name || health.llm_provider}
      </div>
    );
  }

  return null;
}
