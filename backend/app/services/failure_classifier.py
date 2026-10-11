"""
Heuristic "likely failure type" classification for non-supported claims.

IMPORTANT: this never claims to know the actual internal cause of a
hallucination. It only uses signals already available from the pipeline --
retrieval scores, claim-to-evidence similarity, and how much retrieved
chunks agree with each other -- to suggest the most plausible explanation.
Every result is phrased as "likely" and comes with the reasoning that
produced it, so it can be inspected and disagreed with, not taken as fact.
"""
import logging
from itertools import combinations
from collections import Counter
from typing import List, Optional, TypedDict

import numpy as np

from app.config import settings
from app.services.embedding_service import embed_texts

logger = logging.getLogger(__name__)

RETRIEVAL_FAILURE = "retrieval_failure"
INSUFFICIENT_CONTEXT = "insufficient_context"
CONTEXT_MISINTERPRETATION = "context_misinterpretation"
UNSUPPORTED_GENERATION = "unsupported_generation"
CONFLICTING_EVIDENCE = "conflicting_evidence"


class FailureClassification(TypedDict):
    likely_failure_type: Optional[str]
    reasoning: str


def _cosine_sim(a: List[float], b: List[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return float(np.dot(a_arr, b_arr) / denom) if denom else 0.0


def _detect_conflicting_evidence(retrieved_chunks: List[dict]) -> Optional[str]:
    """
    Weak heuristic: among chunks that individually looked relevant to the
    query, do any two diverge sharply from each other? This is a proxy for
    "sources might disagree" -- embedding dissimilarity signals topical
    divergence, NOT proven contradiction. Treat this as the lowest-confidence
    of the five classifications; a real contradiction check needs NLI
    (see Phase 8).
    """
    relevant = [c for c in retrieved_chunks if c.get("score", 0) >= settings.retrieval_relevance_threshold]
    if len(relevant) < 2:
        return None

    embeddings = embed_texts([c["text"] for c in relevant])

    min_pairwise = 1.0
    for i, j in combinations(range(len(embeddings)), 2):
        min_pairwise = min(min_pairwise, _cosine_sim(embeddings[i], embeddings[j]))

    if min_pairwise < settings.conflict_dissimilarity_threshold:
        return (
            f"Multiple retrieved chunks were individually relevant to the question, but diverge "
            f"substantially from each other (lowest pairwise similarity {round(min_pairwise, 3)}). "
            f"This may indicate the sources contain differing information -- worth a manual look, "
            f"since this signal measures topical divergence, not confirmed contradiction."
        )
    return None


def classify_claim_failure(
    claim_status: str,
    claim_similarity_score: float,
    retrieved_chunks: List[dict],
) -> FailureClassification:
    """
    Suggests a likely failure type for one non-supported claim.
    Returns likely_failure_type=None for supported claims (nothing to explain).
    """
    if claim_status == "supported":
        return {"likely_failure_type": None, "reasoning": "Claim was supported by evidence; no failure to classify."}

    best_retrieval_score = max((c.get("score", 0.0) for c in retrieved_chunks), default=0.0)

    # 1. Nothing retrieved was even topically relevant to the question.
    if not retrieved_chunks or best_retrieval_score < settings.retrieval_relevance_threshold:
        return {
            "likely_failure_type": RETRIEVAL_FAILURE,
            "reasoning": (
                f"No retrieved chunk was strongly relevant to the question "
                f"(best retrieval score {round(best_retrieval_score, 3)} < threshold "
                f"{settings.retrieval_relevance_threshold}). The relevant information was likely "
                f"never retrieved from the document."
            ),
        }

    # 2. Retrieved chunks look relevant but sharply diverge from each other.
    conflict_reasoning = _detect_conflicting_evidence(retrieved_chunks)
    if conflict_reasoning:
        return {"likely_failure_type": CONFLICTING_EVIDENCE, "reasoning": conflict_reasoning}

    # 3. Evidence is topically related to this claim, but not a close enough match.
    if settings.partial_support_threshold <= claim_similarity_score < settings.support_threshold:
        return {
            "likely_failure_type": CONTEXT_MISINTERPRETATION,
            "reasoning": (
                f"Relevant evidence was retrieved and is topically related to this claim "
                f"(claim-evidence similarity {round(claim_similarity_score, 3)}), but not closely "
                f"enough to be considered fully supported. The answer may be interpreting the "
                f"evidence incorrectly rather than inventing new information."
            ),
        }

    # 4. Claim similarity is low despite retrieval looking okay -- split by retrieval strength.
    if best_retrieval_score >= settings.strong_retrieval_threshold:
        return {
            "likely_failure_type": UNSUPPORTED_GENERATION,
            "reasoning": (
                f"Retrieval found strongly relevant material for the question overall (best score "
                f"{round(best_retrieval_score, 3)}), but this specific claim doesn't match any "
                f"retrieved evidence well (similarity {round(claim_similarity_score, 3)}). The model "
                f"likely introduced information not present in the retrieved context."
            ),
        }

    return {
        "likely_failure_type": INSUFFICIENT_CONTEXT,
        "reasoning": (
            f"Retrieval found only moderately relevant material (best score "
            f"{round(best_retrieval_score, 3)}), and this claim doesn't match it well "
            f"(similarity {round(claim_similarity_score, 3)}). The retrieved context likely didn't "
            f"contain enough information to answer this part of the question."
        ),
    }


# Priority order used only to break ties when multiple failure types are
# equally frequent across a set of claims. This is a deterministic tie-break
# for reproducibility, not a claim about which failure type is "worse."
_TIE_BREAK_PRIORITY = [
    RETRIEVAL_FAILURE,
    CONFLICTING_EVIDENCE,
    INSUFFICIENT_CONTEXT,
    CONTEXT_MISINTERPRETATION,
    UNSUPPORTED_GENERATION,
]


def determine_dominant_failure(claims: List[dict]) -> Optional[str]:
    """
    Picks a single failure type to drive mitigation, from the failure types
    already attached to non-supported claims. Uses whichever type is most
    frequent; ties are broken by _TIE_BREAK_PRIORITY.
    """
    types = [c["likely_failure_type"] for c in claims if c.get("likely_failure_type")]
    if not types:
        return None

    counts = Counter(types)
    max_count = max(counts.values())
    candidates = {t for t, n in counts.items() if n == max_count}

    for p in _TIE_BREAK_PRIORITY:
        if p in candidates:
            return p
    return next(iter(candidates))
