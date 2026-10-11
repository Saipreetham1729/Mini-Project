from typing import List

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    message: str


class ChunkOut(BaseModel):
    chunk_id: str
    text: str
    page_number: int
    word_count: int


class ProcessResponse(BaseModel):
    doc_id: str
    page_count: int
    chunk_count: int
    message: str


class ChunksListResponse(BaseModel):
    doc_id: str
    chunk_count: int
    chunks: List[ChunkOut]


class EmbedResponse(BaseModel):
    doc_id: str
    embedded_chunk_count: int
    message: str


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    text: str
    page_number: int
    score: float


class RetrieveResponse(BaseModel):
    doc_id: str
    query: str
    results: List[RetrievedChunkOut]
