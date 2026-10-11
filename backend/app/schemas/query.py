from typing import List

from pydantic import BaseModel

from app.schemas.document import RetrievedChunkOut


class QueryRequest(BaseModel):
    doc_id: str
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    doc_id: str
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunkOut]
    latency_seconds: float
