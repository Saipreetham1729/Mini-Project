from typing import List, Optional

from pydantic import BaseModel

from app.schemas.verification import ClaimVerificationOut


class MitigateRequest(BaseModel):
    doc_id: str
    question: str
    original_answer: str
    claims: List[ClaimVerificationOut]  # the claims array from a prior /verify call
    top_k: int = 5


class SupportSummary(BaseModel):
    supported_count: int
    partially_supported_count: int
    unsupported_count: int
    support_ratio: float


class MitigateResponse(BaseModel):
    doc_id: str
    question: str
    dominant_failure_type: Optional[str]
    strategy_used: str
    original_answer: str
    revised_answer: str
    before: SupportSummary
    after: SupportSummary
    after_claims: List[ClaimVerificationOut]
    latency_seconds: float
