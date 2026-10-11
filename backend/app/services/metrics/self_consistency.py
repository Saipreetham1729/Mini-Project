"""
Self-consistency faithfulness signal.

Unlike the other four metrics, this one doesn't check a claim against
*retrieved* evidence -- it checks whether the claim keeps showing up when
the model is asked the same question again, independently, at a higher
temperature. The idea (from the SelfCheckGPT line of work): a claim that's
grounded in the context tends to reappear consistently across independent
samples, while a hallucinated claim is comparatively fragile and drifts or
disappears from sample to sample. It's a useful signal specifically because
it needs no evidence/ground-truth at all -- only the model's own
sampling variance.

Implementation choice: agreement between a claim and a sample is measured
with the same ROUGE-L scorer already used elsewhere in this project (lexical
overlap), applied once per independent sample and averaged, rather than
adding a second NLI pass per sample. This mirrors the n-gram/lexical variant
of SelfCheckGPT specifically, and keeps this metric cheap enough to run
n_samples extra LLM calls without also running n_samples extra NLI passes.
Comparing this lexical self-consistency score against the semantic metrics
(NLI, LLM-judge, RAGAS faithfulness) elsewhere in the evaluation is itself
part of the comparison this project is making.
"""
import logging
from typing import List, Optional, TypedDict

from app.config import settings
from app.services.llm.factory import get_llm_provider
from app.services.metrics import rouge_l
from app.services.prompts import RAG_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ClaimMetricScore(TypedDict):
    claim_id: str
    score: Optional[float]
    sample_count: int


def score_claims(
    claims: List[dict],
    question: str,
    context: str,
    n_samples: Optional[int] = None,
    temperature: Optional[float] = None,
) -> List[ClaimMetricScore]:
    if not claims:
        return []

    n_samples = n_samples or settings.self_consistency_samples
    temperature = settings.self_consistency_temperature if temperature is None else temperature
    llm = get_llm_provider()
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    samples: List[str] = []
    for i in range(n_samples):
        try:
            samples.append(llm.generate(RAG_SYSTEM_PROMPT, user_prompt, temperature=temperature))
        except Exception as exc:  # noqa: BLE001 -- one failed sample shouldn't sink the whole metric
            logger.warning("Self-consistency sample %d/%d failed: %s", i + 1, n_samples, exc)

    if not samples:
        return [{"claim_id": c["claim_id"], "score": None, "sample_count": 0} for c in claims]

    totals = {c["claim_id"]: 0.0 for c in claims}
    for i, sample_text in enumerate(samples):
        pseudo_evidence = [{"chunk_id": f"sample_{i}", "text": sample_text}]
        for scored in rouge_l.score_claims(claims, pseudo_evidence):
            totals[scored["claim_id"]] += scored["score"]

    return [
        {
            "claim_id": c["claim_id"],
            "score": round(totals[c["claim_id"]] / len(samples), 4),
            "sample_count": len(samples),
        }
        for c in claims
    ]
