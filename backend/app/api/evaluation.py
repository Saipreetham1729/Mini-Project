import logging

from fastapi import APIRouter, HTTPException

from app.schemas.evaluation import EvaluateRequest, EvaluateResponse, MetricResultOut
from app.services.claim_extractor import extract_claims
from app.services.metrics.evaluator import run_all_metrics
from app.services.rag_pipeline import build_context

logger = logging.getLogger(__name__)

router = APIRouter(tags=["evaluation"])


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(body: EvaluateRequest):
    if not body.answer.strip():
        raise HTTPException(status_code=400, detail="Answer cannot be empty.")

    claims = extract_claims(body.answer)
    if not claims:
        raise HTTPException(status_code=422, detail="No checkable claims were extracted from this answer.")

    evidence_chunks = [c.model_dump() for c in body.retrieved_chunks]
    context = build_context(evidence_chunks)
    results = run_all_metrics(claims, evidence_chunks, question=body.question, context=context)

    return EvaluateResponse(
        doc_id=body.doc_id,
        question=body.question,
        answer=body.answer,
        claim_count=len(claims),
        rouge_l=MetricResultOut(**results["rouge_l"]),
        nli_entailment=MetricResultOut(**results["nli_entailment"]),
        llm_judge=MetricResultOut(**results["llm_judge"]),
        ragas_faithfulness=MetricResultOut(**results["ragas_faithfulness"]),
        self_consistency=MetricResultOut(**results["self_consistency"]),
        latency_seconds=results["latency_seconds"],
    )
