"""
Baseline evidence verification using embedding cosine similarity.

For each claim, find the evidence chunk it's most semantically similar to
and classify support level by threshold. This is a simple, explainable
baseline standing in for real faithfulness metrics -- Phase 8 adds NLI
entailment, ROUGE-L, RAGAS, self-consistency, and LLM verification, and
those are what the research comparison is actually built on. This phase
just needs claim -> evidence -> label working end-to-end.
"""
import logging
from typing import List, Optional, TypedDict

import numpy as np

from app.config import settings
from app.services.claim_extractor import Claim
from app.services.embedding_service import embed_texts

logger = logging.getLogger(__name__)

STATUS_SUPPORTED = "supported"
STATUS_PARTIAL = "partially_supported"
STATUS_UNSUPPORTED = "unsupported"


class ClaimVerification(TypedDict):
    claim_id: str
    text: str
    status: str
    best_evidence_chunk_id: Optional[str]
    best_evidence_page: Optional[int]
    similarity_score: float


def _cosine_sim(a: List[float], b: List[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def _classify(score: float) -> str:
    if score >= settings.support_threshold:
        return STATUS_SUPPORTED
    if score >= settings.partial_support_threshold:
        return STATUS_PARTIAL
    return STATUS_UNSUPPORTED


def verify_claims(claims: List[Claim], evidence_chunks: List[dict]) -> List[ClaimVerification]:
    if not claims:
        return []

    if not evidence_chunks:
        # No evidence at all -> every claim is trivially unsupported.
        return [
            {
                "claim_id": c["claim_id"],
                "text": c["text"],
                "status": STATUS_UNSUPPORTED,
                "best_evidence_chunk_id": None,
                "best_evidence_page": None,
                "similarity_score": 0.0,
            }
            for c in claims
        ]

    claim_embeddings = embed_texts([c["text"] for c in claims])
    evidence_embeddings = embed_texts([e["text"] for e in evidence_chunks])

    results: List[ClaimVerification] = []
    for claim, claim_emb in zip(claims, claim_embeddings):
        best_score = -1.0
        best_chunk = None
        for chunk, chunk_emb in zip(evidence_chunks, evidence_embeddings):
            score = _cosine_sim(claim_emb, chunk_emb)
            if score > best_score:
                best_score = score
                best_chunk = chunk

        results.append(
            {
                "claim_id": claim["claim_id"],
                "text": claim["text"],
                "status": _classify(best_score),
                "best_evidence_chunk_id": best_chunk["chunk_id"] if best_chunk else None,
                "best_evidence_page": best_chunk["page_number"] if best_chunk else None,
                "similarity_score": round(best_score, 4),
            }
        )

    return results
