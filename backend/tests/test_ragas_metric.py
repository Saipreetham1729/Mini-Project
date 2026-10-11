import json
from unittest.mock import MagicMock, patch

from app.services.metrics import ragas_metric

CLAIMS = [
    {"claim_id": "c0", "text": "RAG grounds answers in retrieved evidence."},
    {"claim_id": "c1", "text": "The Eiffel Tower was designed by Alexandre Dumas."},
]
EVIDENCE = [{"chunk_id": "e0", "text": "RAG grounds generated answers in retrieved evidence.", "page_number": 1}]


def _fake_llm(response_text):
    provider = MagicMock()
    provider.generate.return_value = response_text
    return provider


@patch("app.services.metrics.ragas_metric.get_llm_provider")
def test_yes_no_verdicts_map_to_binary_scores(mock_factory):
    response = json.dumps(
        [
            {"claim_id": "c0", "verdict": "yes", "reason": "Directly stated in context."},
            {"claim_id": "c1", "verdict": "no", "reason": "Not mentioned in context."},
        ]
    )
    mock_factory.return_value = _fake_llm(response)

    results = ragas_metric.score_claims(CLAIMS, EVIDENCE)

    assert results[0]["score"] == 1.0
    assert results[0]["verdict"] == "yes"
    assert results[1]["score"] == 0.0
    assert results[1]["verdict"] == "no"


@patch("app.services.metrics.ragas_metric.get_llm_provider")
def test_unparseable_response_returns_none_scores_not_guesses(mock_factory):
    mock_factory.return_value = _fake_llm("not valid json at all")

    results = ragas_metric.score_claims(CLAIMS, EVIDENCE)

    assert all(r["score"] is None for r in results)
    assert all("Not evaluated yet" in r["reason"] for r in results)


def test_no_evidence_marks_all_unsupported():
    results = ragas_metric.score_claims(CLAIMS, [])
    assert all(r["score"] == 0.0 and r["verdict"] == "no" for r in results)


def test_empty_claims_returns_empty_list():
    assert ragas_metric.score_claims([], EVIDENCE) == []
