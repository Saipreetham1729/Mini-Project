"""
Runs the full pipeline (retrieve -> generate -> verify -> classify failure ->
evaluate metrics) for one benchmark question and persists one row per claim.

This is the reproducibility artifact the project's brief requires: every run
records the model used, the question, the generated claims, every metric's
score, the likely failure type, and latency -- with human_label left NULL
until manually annotated in a separate labeling step.
"""
import logging
import time
import uuid

from app.config import settings
from app.db import repository
from app.services import vector_store
from app.services.claim_extractor import extract_claims
from app.services.embedding_service import embed_query
from app.services.evidence_verifier import verify_claims
from app.services.failure_classifier import classify_claim_failure
from app.services.llm.factory import get_llm_provider
from app.services.metrics import llm_judge, nli_entailment, ragas_metric, rouge_l, self_consistency
from app.services.rag_pipeline import SYSTEM_PROMPT, build_context

logger = logging.getLogger(__name__)


def run_benchmark_question(question_row: dict, top_k: int = 5) -> dict:
    doc_id = question_row["doc_id"]
    question = question_row["question"]
    run_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()

    query_embedding = embed_query(question)
    retrieved = vector_store.query(doc_id, query_embedding, top_k=top_k)
    if not retrieved:
        return {"run_id": run_id, "claims_recorded": 0, "message": "No relevant context retrieved -- nothing to record."}

    context = build_context(retrieved)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    llm = get_llm_provider()
    answer = llm.generate(SYSTEM_PROMPT, user_prompt)

    claims = extract_claims(answer)
    if not claims:
        return {"run_id": run_id, "claims_recorded": 0, "message": "No checkable claims extracted from the answer."}

    baseline_verifications = verify_claims(claims, retrieved)
    rouge_scores = {c["claim_id"]: c for c in rouge_l.score_claims(claims, retrieved)}

    try:
        nli_scores = {c["claim_id"]: c for c in nli_entailment.score_claims(claims, retrieved)}
    except Exception as exc:  # noqa: BLE001 -- NLI model may be unavailable (no network, not downloaded yet)
        logger.warning("NLI metric unavailable during benchmark run: %s", exc)
        nli_scores = {}

    llm_scores = {c["claim_id"]: c for c in llm_judge.score_claims(claims, retrieved)}

    try:
        ragas_scores = {c["claim_id"]: c for c in ragas_metric.score_claims(claims, retrieved)}
    except Exception as exc:  # noqa: BLE001 -- depends on the same LLM call llm_judge does, can fail the same ways
        logger.warning("RAGAS faithfulness unavailable during benchmark run: %s", exc)
        ragas_scores = {}

    try:
        self_consistency_scores = {
            c["claim_id"]: c for c in self_consistency.score_claims(claims, question, context)
        }
    except Exception as exc:  # noqa: BLE001 -- extra sampling calls can fail independently (rate limits, network)
        logger.warning("Self-consistency unavailable during benchmark run: %s", exc)
        self_consistency_scores = {}

    chunk_ids = ",".join(c["chunk_id"] for c in retrieved)
    model_name = settings.llm_model_name if settings.llm_provider == "openai" else "mock"
    latency = round(time.perf_counter() - start, 3)

    recorded = 0
    for v in baseline_verifications:
        failure = classify_claim_failure(v["status"], v["similarity_score"], retrieved)
        rouge_entry = rouge_scores.get(v["claim_id"], {})
        nli_entry = nli_scores.get(v["claim_id"], {})
        llm_entry = llm_scores.get(v["claim_id"], {})
        ragas_entry = ragas_scores.get(v["claim_id"], {})
        self_consistency_entry = self_consistency_scores.get(v["claim_id"], {})

        repository.insert_benchmark_claim(
            {
                "run_id": run_id,
                "benchmark_question_id": question_row["id"],
                "doc_id": doc_id,
                "question": question,
                "claim_id": v["claim_id"],
                "claim_text": v["text"],
                "answer_text": answer,
                "retrieved_chunk_ids": chunk_ids,
                "baseline_status": v["status"],
                "baseline_score": v["similarity_score"],
                "likely_failure_type": failure["likely_failure_type"],
                "rouge_l_score": rouge_entry.get("score"),
                "nli_score": nli_entry.get("score"),
                "llm_judge_score": llm_entry.get("score"),
                "llm_judge_verdict": llm_entry.get("verdict"),
                "ragas_score": ragas_entry.get("score"),
                "self_consistency_score": self_consistency_entry.get("score"),
                "model_name": model_name,
                "latency_seconds": latency,
            }
        )
        recorded += 1

    return {"run_id": run_id, "claims_recorded": recorded, "message": f"Recorded {recorded} claims for review."}
