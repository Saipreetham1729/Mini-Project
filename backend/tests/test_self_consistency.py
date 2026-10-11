from unittest.mock import MagicMock, patch

from app.services.metrics import self_consistency

CLAIMS = [{"claim_id": "c0", "text": "RAG grounds answers in retrieved evidence."}]


def _fake_llm(*responses):
    provider = MagicMock()
    provider.generate.side_effect = list(responses)
    return provider


@patch("app.services.metrics.self_consistency.get_llm_provider")
def test_consistent_samples_score_high(mock_factory):
    # Every independent sample repeats the claim near-verbatim -> high agreement.
    mock_factory.return_value = _fake_llm(
        "RAG grounds answers in retrieved evidence.",
        "RAG grounds answers in retrieved evidence.",
        "RAG grounds answers in retrieved evidence.",
    )

    results = self_consistency.score_claims(CLAIMS, "How does RAG work?", "some context", n_samples=3)

    assert results[0]["sample_count"] == 3
    assert results[0]["score"] > 0.9


@patch("app.services.metrics.self_consistency.get_llm_provider")
def test_inconsistent_samples_score_low(mock_factory):
    # Samples never mention the claim at all -> low agreement.
    mock_factory.return_value = _fake_llm(
        "The weather today is sunny.",
        "Bananas are a good source of potassium.",
        "The stock market closed higher.",
    )

    results = self_consistency.score_claims(CLAIMS, "How does RAG work?", "some context", n_samples=3)

    assert results[0]["score"] < 0.3


@patch("app.services.metrics.self_consistency.get_llm_provider")
def test_all_samples_failing_returns_none_not_a_guess(mock_factory):
    provider = MagicMock()
    provider.generate.side_effect = RuntimeError("network down")
    mock_factory.return_value = provider

    results = self_consistency.score_claims(CLAIMS, "How does RAG work?", "some context", n_samples=2)

    assert results[0]["score"] is None
    assert results[0]["sample_count"] == 0


def test_empty_claims_returns_empty_list():
    assert self_consistency.score_claims([], "q", "ctx") == []
