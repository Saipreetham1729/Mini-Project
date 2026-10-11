"""
ROUGE-L faithfulness baseline: lexical overlap between a claim and its
retrieved evidence, using longest-common-subsequence F1.

Included specifically as the cheap baseline in the comparison -- part of
the research question is whether cheap lexical overlap correlates with
human judgment as well as the more expensive semantic methods (NLI,
LLM-judge) do, or falls short.
"""
import logging
from typing import List, Optional, TypedDict

from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


class ClaimMetricScore(TypedDict):
    claim_id: str
    score: float
    best_evidence_chunk_id: Optional[str]


def score_claims(claims: List[dict], evidence_chunks: List[dict]) -> List[ClaimMetricScore]:
    """
    For each claim, computes ROUGE-L F1 against every retrieved chunk and
    keeps the best (max) score -- same max-over-chunks convention used by
    the other metrics in this phase, so results are directly comparable.
    """
    if not evidence_chunks:
        return [{"claim_id": c["claim_id"], "score": 0.0, "best_evidence_chunk_id": None} for c in claims]

    results: List[ClaimMetricScore] = []
    for claim in claims:
        best_f1 = 0.0
        best_chunk_id: Optional[str] = None
        for chunk in evidence_chunks:
            scored = _scorer.score(chunk["text"], claim["text"])
            f1 = scored["rougeL"].fmeasure
            if f1 > best_f1:
                best_f1 = f1
                best_chunk_id = chunk["chunk_id"]
        results.append(
            {"claim_id": claim["claim_id"], "score": round(best_f1, 4), "best_evidence_chunk_id": best_chunk_id}
        )
    return results
