from typing import List, Optional

from pydantic import BaseModel

from app.schemas.document import RetrievedChunkOut


class ClaimVerificationOut(BaseModel):
    claim_id: str
    text: str
    status: str  # "supported" | "partially_supported" | "unsupported"
    best_evidence_chunk_id: Optional[str] = None
    best_evidence_page: Optional[int] = None
    similarity_score: float
    likely_failure_type: Optional[str] = None
    failure_reasoning: Optional[str] = None


class VerifyRequest(BaseModel):
    doc_id: str
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunkOut]


class VerifyResponse(BaseModel):
    doc_id: str
    question: str
    answer: str
    claims: List[ClaimVerificationOut]
    supported_count: int
    partially_supported_count: int
    unsupported_count: int
    support_ratio: float  # supported claims / total claims
