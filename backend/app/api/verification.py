import logging

from fastapi import APIRouter, HTTPException

from app.schemas.verification import ClaimVerificationOut, VerifyRequest, VerifyResponse
from app.services.claim_extractor import extract_claims
from app.services.evidence_verifier import verify_claims
from app.services.failure_classifier import classify_claim_failure

logger = logging.getLogger(__name__)

router = APIRouter(tags=["verification"])


@router.post("/verify", response_model=VerifyResponse)
def verify(body: VerifyRequest):
    if not body.answer.strip():
        raise HTTPException(status_code=400, detail="Answer cannot be empty.")

    claims = extract_claims(body.answer)
    evidence_chunks = [c.model_dump() for c in body.retrieved_chunks]
    verifications = verify_claims(claims, evidence_chunks)

    for v in verifications:
        failure = classify_claim_failure(v["status"], v["similarity_score"], evidence_chunks)
        v["likely_failure_type"] = failure["likely_failure_type"]
        v["failure_reasoning"] = failure["reasoning"]

    supported = sum(1 for v in verifications if v["status"] == "supported")
    partial = sum(1 for v in verifications if v["status"] == "partially_supported")
    unsupported = sum(1 for v in verifications if v["status"] == "unsupported")
    total = len(verifications)
    ratio = round(supported / total, 4) if total else 0.0

    logger.info(
        "Verified %d claims (doc_id=%s): %d supported, %d partial, %d unsupported",
        total, body.doc_id, supported, partial, unsupported,
    )

    return VerifyResponse(
        doc_id=body.doc_id,
        question=body.question,
        answer=body.answer,
        claims=[ClaimVerificationOut(**v) for v in verifications],
        supported_count=supported,
        partially_supported_count=partial,
        unsupported_count=unsupported,
        support_ratio=ratio,
    )
