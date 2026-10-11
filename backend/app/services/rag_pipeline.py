"""
Orchestrates the basic RAG flow: question -> retrieval -> context -> LLM -> answer.

This is intentionally a plain, non-strict prompt -- a "naive" RAG baseline.
Later phases (verification, mitigation) build on top of this and compare a
stricter, evidence-grounded prompt against this baseline. If we made this
prompt maximally strict now, there would be nothing left to demonstrate
improvement over.
"""
import logging
import time
from typing import List, TypedDict

from app.services import vector_store
from app.services.embedding_service import embed_query
from app.services.llm.factory import get_llm_provider
from app.services.prompts import RAG_SYSTEM_PROMPT
from app.services.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)

# Re-exported for backwards compatibility -- benchmark_runner and others
# already import SYSTEM_PROMPT from this module.
SYSTEM_PROMPT = RAG_SYSTEM_PROMPT


class RagResult(TypedDict):
    doc_id: str
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    latency_seconds: float


def _build_context(chunks: List[RetrievedChunk]) -> str:
    parts = [f"[Page {c['page_number']}] {c['text']}" for c in chunks]
    return "\n\n".join(parts)


# Public alias -- Phase 7's mitigation module reuses this exact context-building
# logic so "before" and "after" answers are built the same way, differing only
# in the prompt and/or retrieved chunk set.
build_context = _build_context


def run_query(doc_id: str, question: str, top_k: int = 5) -> RagResult:
    start = time.perf_counter()

    query_embedding = embed_query(question)
    retrieved = vector_store.query(doc_id, query_embedding, top_k=top_k)

    if not retrieved:
        latency = round(time.perf_counter() - start, 3)
        return {
            "doc_id": doc_id,
            "question": question,
            "answer": "No relevant context was found for this document. Has it been embedded yet?",
            "retrieved_chunks": [],
            "latency_seconds": latency,
        }

    context = _build_context(retrieved)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    llm = get_llm_provider()
    answer = llm.generate(SYSTEM_PROMPT, user_prompt)

    latency = round(time.perf_counter() - start, 3)
    logger.info("RAG query answered in %.3fs (doc_id=%s)", latency, doc_id)

    return {
        "doc_id": doc_id,
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved,
        "latency_seconds": latency,
    }
