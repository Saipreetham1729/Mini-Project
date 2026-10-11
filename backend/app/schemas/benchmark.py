from typing import List, Optional

from pydantic import BaseModel


class BenchmarkQuestionCreate(BaseModel):
    doc_id: str
    question: str
    notes: Optional[str] = None


class BenchmarkQuestionOut(BaseModel):
    id: int
    doc_id: str
    question: str
    notes: Optional[str] = None
    created_at: str


class BenchmarkRunResponse(BaseModel):
    run_id: str
    claims_recorded: int
    message: str


class BenchmarkClaimOut(BaseModel):
    id: int
    run_id: str
    doc_id: str
    question: str
    claim_id: str
    claim_text: str
    answer_text: str
    baseline_status: Optional[str] = None
    baseline_score: Optional[float] = None
    likely_failure_type: Optional[str] = None
    rouge_l_score: Optional[float] = None
    nli_score: Optional[float] = None
    llm_judge_score: Optional[float] = None
    llm_judge_verdict: Optional[str] = None
    ragas_score: Optional[float] = None
    self_consistency_score: Optional[float] = None
    human_label: Optional[str] = None
    model_name: Optional[str] = None
    latency_seconds: Optional[float] = None
    created_at: str


class LabelRequest(BaseModel):
    human_label: str  # "supported" or "unsupported"


class PrfResultOut(BaseModel):
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    labeled_examples: int
    note: Optional[str] = None


class BenchmarkMetricsResponse(BaseModel):
    baseline_embedding_similarity: PrfResultOut
    rouge_l: PrfResultOut
    nli_entailment: PrfResultOut
    llm_judge: PrfResultOut
    ragas_faithfulness: PrfResultOut
    self_consistency: PrfResultOut
