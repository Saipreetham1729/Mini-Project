from app.schemas.benchmark import BenchmarkClaimOut, BenchmarkMetricsResponse, PrfResultOut


def test_benchmark_metrics_response_includes_all_six_methods():
    empty_prf = {"precision": None, "recall": None, "f1": None, "labeled_examples": 0, "note": "Not evaluated yet."}
    response = BenchmarkMetricsResponse(
        baseline_embedding_similarity=PrfResultOut(**empty_prf),
        rouge_l=PrfResultOut(**empty_prf),
        nli_entailment=PrfResultOut(**empty_prf),
        llm_judge=PrfResultOut(**empty_prf),
        ragas_faithfulness=PrfResultOut(**empty_prf),
        self_consistency=PrfResultOut(**empty_prf),
    )
    assert set(response.model_dump().keys()) == {
        "baseline_embedding_similarity", "rouge_l", "nli_entailment",
        "llm_judge", "ragas_faithfulness", "self_consistency",
    }


def test_benchmark_claim_out_includes_ragas_and_self_consistency_scores():
    claim = BenchmarkClaimOut(
        id=1, run_id="r1", doc_id="doc1", question="Q?", claim_id="c0", claim_text="Claim",
        answer_text="Answer", ragas_score=0.75, self_consistency_score=0.6, created_at="2026-01-01",
    )
    dumped = claim.model_dump()
    assert dumped["ragas_score"] == 0.75
    assert dumped["self_consistency_score"] == 0.6
