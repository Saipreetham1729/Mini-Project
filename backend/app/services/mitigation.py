"""
Applies one mitigation strategy based on the dominant likely failure type
across a set of claims, then regenerates an answer so before/after can be
compared.

Design note (stated honestly): "retrieval failure" and "insufficient
context" are mitigated the same way here -- retrieve more chunks. The
brief treats them as conceptually distinct causes, but the only lever
either mitigation has available is giving the model more of the document.
The three prompt-based mitigations (misinterpretation, unsupported
generation, conflicting evidence) are where the strategies genuinely
diverge from each other.
"""
import logging
import time
from typing import List, Optional, Tuple, TypedDict

from app.services import vector_store
from app.services.embedding_service import embed_query
from app.services.failure_classifier import (
    CONFLICTING_EVIDENCE,
    CONTEXT_MISINTERPRETATION,
    INSUFFICIENT_CONTEXT,
    RETRIEVAL_FAILURE,
    UNSUPPORTED_GENERATION,
)
from app.services.llm.factory import get_llm_provider
from app.services.rag_pipeline import build_context

logger = logging.getLogger(__name__)

STRICT_GROUNDED_PROMPT = (
    "You are a careful assistant that answers ONLY using the provided context. "
    "Re-read the context carefully before answering, and make sure your answer accurately "
    "reflects what the context says -- do not draw conclusions the context doesn't support. "
    "If the context does not contain enough information, say so explicitly."
)

CLAIM_SUPPORT_PROMPT = (
    "You are a careful assistant that answers ONLY using the provided context. "
    "Every statement in your answer must be directly supported by the context. "
    "Do not add any information that is not explicitly present in the context. "
    "If the context does not contain enough information to answer the question, "
    "respond with: 'Not enough information is available in the provided context to answer this.'"
)

CONFLICT_AWARE_PROMPT = (
    "You are a careful assistant that answers ONLY using the provided context. "
    "The retrieved context may contain passages that disagree with each other. "
    "If you notice conflicting information, point out the disagreement explicitly rather than "
    "presenting one version as certain. Do not silently pick one side."
)

DEFAULT_PROMPT = (
    "You are a careful assistant that answers ONLY using the provided context. "
    "If the context does not contain enough information, say so explicitly."
)


class MitigationResult(TypedDict):
    strategy_used: str
    dominant_failure_type: Optional[str]
    revised_answer: str
    retrieved_chunks: List[dict]
    latency_seconds: float


def _regenerate(doc_id: str, question: str, top_k: int, system_prompt: str) -> Tuple[str, List[dict]]:
    query_embedding = embed_query(question)
    retrieved = vector_store.query(doc_id, query_embedding, top_k=top_k)
    if not retrieved:
        return "No relevant context was found for this document.", []

    context = build_context(retrieved)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    llm = get_llm_provider()
    answer = llm.generate(system_prompt, user_prompt)
    return answer, retrieved


# Maps each failure type to (strategy_name, top_k_multiplier, system_prompt).
# The multiplier is 2x for the two "give it more context" strategies, 1x
# (unchanged) for the three prompt-only strategies.
_STRATEGY_TABLE = {
    RETRIEVAL_FAILURE: ("retrieve_additional_chunks", 2, STRICT_GROUNDED_PROMPT),
    INSUFFICIENT_CONTEXT: ("expand_evidence_context", 2, STRICT_GROUNDED_PROMPT),
    CONTEXT_MISINTERPRETATION: ("stricter_grounded_prompt", 1, STRICT_GROUNDED_PROMPT),
    UNSUPPORTED_GENERATION: ("require_claim_support", 1, CLAIM_SUPPORT_PROMPT),
    CONFLICTING_EVIDENCE: ("surface_conflicts", 1, CONFLICT_AWARE_PROMPT),
}


def apply_mitigation(
    doc_id: str,
    question: str,
    dominant_failure_type: Optional[str],
    base_top_k: int = 5,
) -> MitigationResult:
    start = time.perf_counter()

    strategy, multiplier, system_prompt = _STRATEGY_TABLE.get(
        dominant_failure_type, ("stricter_grounded_prompt", 1, DEFAULT_PROMPT)
    )

    answer, retrieved = _regenerate(doc_id, question, base_top_k * multiplier, system_prompt)

    latency = round(time.perf_counter() - start, 3)
    logger.info("Applied mitigation '%s' for doc_id=%s in %.3fs", strategy, doc_id, latency)

    return {
        "strategy_used": strategy,
        "dominant_failure_type": dominant_failure_type,
        "revised_answer": answer,
        "retrieved_chunks": retrieved,
        "latency_seconds": latency,
    }
