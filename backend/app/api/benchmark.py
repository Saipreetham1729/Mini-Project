import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.db import repository
from app.schemas.benchmark import (
    BenchmarkClaimOut,
    BenchmarkMetricsResponse,
    BenchmarkQuestionCreate,
    BenchmarkQuestionOut,
    BenchmarkRunResponse,
    LabelRequest,
    PrfResultOut,
)
from app.services.benchmark_metrics import compute_prf_for_all_methods
from app.services.benchmark_runner import run_benchmark_question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


@router.post("/questions", response_model=BenchmarkQuestionOut)
def add_question(body: BenchmarkQuestionCreate):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    row = repository.add_benchmark_question(body.doc_id, body.question, body.notes)
    return BenchmarkQuestionOut(**row)


@router.get("/questions", response_model=List[BenchmarkQuestionOut])
def list_questions():
    return [BenchmarkQuestionOut(**r) for r in repository.list_benchmark_questions()]


@router.post("/questions/{question_id}/run", response_model=BenchmarkRunResponse)
def run_question(question_id: int, top_k: int = 5):
    question_row = repository.get_benchmark_question(question_id)
    if not question_row:
        raise HTTPException(status_code=404, detail=f"No benchmark question with id={question_id}")
    result = run_benchmark_question(question_row, top_k=top_k)
    return BenchmarkRunResponse(**result)


@router.get("/claims", response_model=List[BenchmarkClaimOut])
def list_claims(
    labeled_only: Optional[bool] = Query(default=None),
    run_id: Optional[str] = Query(default=None),
):
    rows = repository.list_benchmark_claims(labeled_only=labeled_only, run_id=run_id)
    return [BenchmarkClaimOut(**r) for r in rows]


@router.post("/claims/{claim_row_id}/label", response_model=BenchmarkClaimOut)
def label_claim(claim_row_id: int, body: LabelRequest):
    if body.human_label not in ("supported", "unsupported"):
        raise HTTPException(status_code=400, detail="human_label must be 'supported' or 'unsupported'.")
    row = repository.set_human_label(claim_row_id, body.human_label)
    if not row:
        raise HTTPException(status_code=404, detail=f"No benchmark claim with id={claim_row_id}")
    return BenchmarkClaimOut(**row)


@router.get("/metrics", response_model=BenchmarkMetricsResponse)
def get_metrics():
    rows = repository.list_benchmark_claims(labeled_only=True)
    results = compute_prf_for_all_methods(rows)
    return BenchmarkMetricsResponse(
        baseline_embedding_similarity=PrfResultOut(**results["baseline_embedding_similarity"]),
        rouge_l=PrfResultOut(**results["rouge_l"]),
        nli_entailment=PrfResultOut(**results["nli_entailment"]),
        llm_judge=PrfResultOut(**results["llm_judge"]),
        ragas_faithfulness=PrfResultOut(**results["ragas_faithfulness"]),
        self_consistency=PrfResultOut(**results["self_consistency"]),
    )
