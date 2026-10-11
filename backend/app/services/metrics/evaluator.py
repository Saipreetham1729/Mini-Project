"""
Runs all faithfulness metrics over a set of claims and aggregates each into
one answer-level score: ROUGE-L, NLI entailment, LLM-judge, and RAGAS
faithfulness check claims against retrieved evidence; self-consistency
checks claims against independently re-sampled answers instead, so it takes
the question and context rather than evidence_chunks.

Any metric that fails to run (e.g. no network to download the NLI model)
degrades to "Not evaluated yet" for that metric only -- other metrics keep
running. Per the project's rule against fabricating results, a missing
measurement is never silently replaced with a guessed number.
"""
import logging
import time
from typing import List, Optional, TypedDict

from app.services.metrics import llm_judge, nli_entailment, ragas_metric, rouge_l, self_consistency

logger = logging.getLogger(__name__)


class MetricRunResult(TypedDict):
    per_claim: List[dict]
    aggregate_score: Optional[float]
    error: Optional[str]


def _run_metric(name: str, fn, claims: List[dict], evidence_chunks: List[dict]) -> MetricRunResult:
    try:
        per_claim = fn(claims, evidence_chunks)
        scored = [c["score"] for c in per_claim if c.get("score") is not None]
        aggregate = round(sum(scored) / len(scored), 4) if scored else None
        return {"per_claim": per_claim, "aggregate_score": aggregate, "error": None}
    except Exception as exc:  # noqa: BLE001 -- each metric backend can fail independently (missing model, network, etc.)
        logger.warning("Metric '%s' failed to run: %s", name, exc)
        return {
            "per_claim": [{"claim_id": c["claim_id"], "score": None} for c in claims],
            "aggregate_score": None,
            "error": str(exc),
        }


def _run_self_consistency(claims: List[dict], question: str, context: str) -> MetricRunResult:
    try:
        per_claim = self_consistency.score_claims(claims, question, context)
        scored = [c["score"] for c in per_claim if c.get("score") is not None]
        aggregate = round(sum(scored) / len(scored), 4) if scored else None
        return {"per_claim": per_claim, "aggregate_score": aggregate, "error": None}
    except Exception as exc:  # noqa: BLE001 -- sampling can fail independently (network, rate limits, etc.)
        logger.warning("Metric 'self_consistency' failed to run: %s", exc)
        return {
            "per_claim": [{"claim_id": c["claim_id"], "score": None} for c in claims],
            "aggregate_score": None,
            "error": str(exc),
        }


def run_all_metrics(claims: List[dict], evidence_chunks: List[dict], question: str = "", context: str = "") -> dict:
    start = time.perf_counter()
    results = {
        "rouge_l": _run_metric("rouge_l", rouge_l.score_claims, claims, evidence_chunks),
        "nli_entailment": _run_metric("nli_entailment", nli_entailment.score_claims, claims, evidence_chunks),
        "llm_judge": _run_metric("llm_judge", llm_judge.score_claims, claims, evidence_chunks),
        "ragas_faithfulness": _run_metric("ragas_faithfulness", ragas_metric.score_claims, claims, evidence_chunks),
        "self_consistency": _run_self_consistency(claims, question, context),
    }
    results["latency_seconds"] = round(time.perf_counter() - start, 3)
    return results
