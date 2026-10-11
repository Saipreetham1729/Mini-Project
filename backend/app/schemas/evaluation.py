from typing import List, Optional

from pydantic import BaseModel

from app.schemas.document import RetrievedChunkOut


class EvaluateRequest(BaseModel):
    doc_id: str
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunkOut]


class MetricResultOut(BaseModel):
    per_claim: List[dict]
    aggregate_score: Optional[float] = None
    error: Optional[str] = None


class EvaluateResponse(BaseModel):
    doc_id: str
    question: str
    answer: str
    claim_count: int
    rouge_l: MetricResultOut
    nli_entailment: MetricResultOut
    llm_judge: MetricResultOut
    ragas_faithfulness: MetricResultOut
    self_consistency: MetricResultOut
    latency_seconds: float
