import logging
from typing import List

from fastapi import APIRouter, HTTPException

from app.schemas.mitigation import MitigateRequest, MitigateResponse, SupportSummary
from app.schemas.verification import ClaimVerificationOut
from app.services.claim_extractor import extract_claims
from app.services.evidence_verifier import verify_claims
from app.services.failure_classifier import classify_claim_failure, determine_dominant_failure
from app.services.mitigation import apply_mitigation

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mitigation"])


def _summarize(claims: List[dict]) -> SupportSummary:
    supported = sum(1 for c in claims if c["status"] == "supported")
    partial = sum(1 for c in claims if c["status"] == "partially_supported")
    unsupported = sum(1 for c in claims if c["status"] == "unsupported")
    total = len(claims)
    ratio = round(supported / total, 4) if total else 0.0
    return SupportSummary(
        supported_count=supported,
        partially_supported_count=partial,
        unsupported_count=unsupported,
        support_ratio=ratio,
    )


@router.post("/mitigate", response_model=MitigateResponse)
def mitigate(body: MitigateRequest):
    if not body.claims:
        raise HTTPException(status_code=400, detail="No claims provided -- run /verify first.")

    before_claims = [c.model_dump() for c in body.claims]
    before_summary = _summarize(before_claims)

    dominant = determine_dominant_failure(before_claims)
    if dominant is None:
        raise HTTPException(
            status_code=422,
            detail="No unsupported or partially supported claims found -- nothing to mitigate.",
        )

    mitigation_result = apply_mitigation(body.doc_id, body.question, dominant, base_top_k=body.top_k)

    revised_claims = extract_claims(mitigation_result["revised_answer"])
    after_verifications = verify_claims(revised_claims, mitigation_result["retrieved_chunks"])
    for v in after_verifications:
        failure = classify_claim_failure(
            v["status"], v["similarity_score"], mitigation_result["retrieved_chunks"]
        )
        v["likely_failure_type"] = failure["likely_failure_type"]
        v["failure_reasoning"] = failure["reasoning"]

    after_summary = _summarize(after_verifications)

    logger.info(
        "Mitigation '%s' for doc_id=%s: support_ratio %.3f -> %.3f",
        mitigation_result["strategy_used"], body.doc_id,
        before_summary.support_ratio, after_summary.support_ratio,
    )

    return MitigateResponse(
        doc_id=body.doc_id,
        question=body.question,
        dominant_failure_type=dominant,
        strategy_used=mitigation_result["strategy_used"],
        original_answer=body.original_answer,
        revised_answer=mitigation_result["revised_answer"],
        before=before_summary,
        after=after_summary,
        after_claims=[ClaimVerificationOut(**v) for v in after_verifications],
        latency_seconds=mitigation_result["latency_seconds"],
    )
