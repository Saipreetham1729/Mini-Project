"""
Thin wrapper around a single persistent ChromaDB collection.

All documents share one collection; each chunk's metadata carries doc_id and
page_number, so retrieval always filters `where={"doc_id": ...}`. This keeps
things simple for a single-collection mini-project scale while still letting
each document's chunks live in isolation for retrieval purposes.
"""
import logging
from typing import List, TypedDict

import chromadb

from app.config import settings

logger = logging.getLogger(__name__)

_client = chromadb.PersistentClient(path=str(settings.vector_db_dir))
_collection = _client.get_or_create_collection(
    name="contextguard_chunks",
    metadata={"hnsw:space": "cosine"},
)


class RetrievedChunk(TypedDict):
    chunk_id: str
    text: str
    page_number: int
    score: float


def add_chunks(doc_id: str, chunks: List[dict], embeddings: List[List[float]]) -> None:
    if not chunks:
        return
    _collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{"doc_id": doc_id, "page_number": c["page_number"]} for c in chunks],
    )
    logger.info("Stored %d chunk embeddings for doc_id=%s", len(chunks), doc_id)


def delete_doc(doc_id: str) -> None:
    """Remove any existing vectors for this doc_id -- makes re-embedding idempotent."""
    _collection.delete(where={"doc_id": doc_id})


def query(doc_id: str, query_embedding: List[float], top_k: int = 5) -> List[RetrievedChunk]:
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"doc_id": doc_id},
    )

    out: List[RetrievedChunk] = []
    if not results["ids"] or not results["ids"][0]:
        return out

    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = round(1 - distance, 4)  # cosine space: distance = 1 - similarity
        out.append(
            {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "page_number": results["metadatas"][0][i]["page_number"],
                "score": similarity,
            }
        )
    return out
