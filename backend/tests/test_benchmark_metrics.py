from app.services.benchmark_metrics import compute_prf_for_all_methods


def test_perfect_agreement_yields_perfect_scores():
    rows = [
        {
            "human_label": "supported", "baseline_score": 0.9, "rouge_l_score": 0.8, "nli_score": 0.85,
            "llm_judge_score": 1.0, "ragas_score": 1.0, "self_consistency_score": 0.9,
        },
        {
            "human_label": "unsupported", "baseline_score": 0.1, "rouge_l_score": 0.05, "nli_score": 0.1,
            "llm_judge_score": 0.0, "ragas_score": 0.0, "self_consistency_score": 0.05,
        },
    ]
    results = compute_prf_for_all_methods(rows)
    assert set(results.keys()) == {
        "baseline_embedding_similarity", "rouge_l", "nli_entailment", "llm_judge",
        "ragas_faithfulness", "self_consistency",
    }
    for method in results.values():
        assert method["precision"] == 1.0
        assert method["recall"] == 1.0
        assert method["f1"] == 1.0
        assert method["labeled_examples"] == 2


def test_no_labels_reports_not_evaluated():
    results = compute_prf_for_all_methods([])
    for method in results.values():
        assert method["precision"] is None
        assert method["labeled_examples"] == 0
        assert "Not evaluated yet" in method["note"]


def test_missed_hallucination_lowers_recall():
    rows = [
        {"human_label": "unsupported", "baseline_score": 0.9, "rouge_l_score": None, "nli_score": None, "llm_judge_score": None}
    ]
    results = compute_prf_for_all_methods(rows)
    assert results["baseline_embedding_similarity"]["recall"] == 0.0
    assert results["baseline_embedding_similarity"]["precision"] is None


def test_unlabeled_rows_are_excluded():
    rows = [
        {"human_label": None, "baseline_score": 0.1, "rouge_l_score": 0.1, "nli_score": 0.1, "llm_judge_score": 0.1},
        {"human_label": "supported", "baseline_score": 0.9, "rouge_l_score": 0.8, "nli_score": 0.85, "llm_judge_score": 1.0},
    ]
    results = compute_prf_for_all_methods(rows)
    assert results["baseline_embedding_similarity"]["labeled_examples"] == 1


def test_ragas_and_self_consistency_missing_scores_report_not_evaluated():
    # Rows from before the extended-scope phase have no ragas_score/self_consistency_score at all.
    rows = [{"human_label": "supported", "baseline_score": 0.9}]
    results = compute_prf_for_all_methods(rows)
    assert results["ragas_faithfulness"]["labeled_examples"] == 0
    assert results["self_consistency"]["labeled_examples"] == 0
