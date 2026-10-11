"""
NLI-based entailment faithfulness metric using a DeBERTa cross-encoder.

For each claim, scores it as the hypothesis against every retrieved chunk
as the premise, and keeps the highest entailment probability (same
max-over-chunks convention as the other metrics). This is the closest of
the three core metrics to what RAGAS-style faithfulness approximates
internally.

Model: cross-encoder/nli-deberta-v3-xsmall (~140MB) -- chosen for a light
download footprint over larger DeBERTa NLI variants, appropriate for a
student laptop; this is a real accuracy/size tradeoff worth naming in the
report. Downloads from Hugging Face on first use.
"""
import logging
from functools import lru_cache
from typing import List, Optional, TypedDict

import numpy as np
from sentence_transformers import CrossEncoder

from app.config import settings

logger = logging.getLogger(__name__)

# Label order for cross-encoder/nli-deberta-v3-xsmall, per its model card:
# [contradiction, entailment, neutral]. If you swap in a different NLI
# cross-encoder, verify its label order before trusting this index.
_ENTAILMENT_INDEX = 1


class ClaimMetricScore(TypedDict):
    claim_id: str
    score: float
    best_evidence_chunk_id: Optional[str]


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    logger.info("Loading NLI model: %s (first load may take a moment)", settings.nli_model_name)
    return CrossEncoder(settings.nli_model_name)


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


def score_claims(claims: List[dict], evidence_chunks: List[dict]) -> List[ClaimMetricScore]:
    if not evidence_chunks:
        return [{"claim_id": c["claim_id"], "score": 0.0, "best_evidence_chunk_id": None} for c in claims]

    model = _get_model()
    results: List[ClaimMetricScore] = []

    for claim in claims:
        pairs = [(chunk["text"], claim["text"]) for chunk in evidence_chunks]
        logits = model.predict(pairs)  # shape (n_chunks, 3)
        entailment_probs = [float(_softmax(np.array(l))[_ENTAILMENT_INDEX]) for l in logits]
        best_idx = max(range(len(entailment_probs)), key=lambda i: entailment_probs[i])
        results.append(
            {
                "claim_id": claim["claim_id"],
                "score": round(entailment_probs[best_idx], 4),
                "best_evidence_chunk_id": evidence_chunks[best_idx]["chunk_id"],
            }
        )
    return results
