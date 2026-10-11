from unittest.mock import patch

from app.services.failure_classifier import (
    CONFLICTING_EVIDENCE,
    CONTEXT_MISINTERPRETATION,
    INSUFFICIENT_CONTEXT,
    RETRIEVAL_FAILURE,
    UNSUPPORTED_GENERATION,
    classify_claim_failure,
    determine_dominant_failure,
)


def test_supported_claim_has_no_failure_type():
    chunks = [{"chunk_id": "c1", "text": "x", "page_number": 1, "score": 0.9}]
    result = classify_claim_failure("supported", 0.9, chunks)
    assert result["likely_failure_type"] is None


def test_low_retrieval_score_means_retrieval_failure():
    chunks = [{"chunk_id": "c1", "text": "x", "page_number": 1, "score": 0.1}]
    result = classify_claim_failure("unsupported", 0.1, chunks)
    assert result["likely_failure_type"] == RETRIEVAL_FAILURE


def test_no_retrieved_chunks_means_retrieval_failure():
    result = classify_claim_failure("unsupported", 0.0, [])
    assert result["likely_failure_type"] == RETRIEVAL_FAILURE


def test_partial_similarity_with_strong_retrieval_means_misinterpretation():
    chunks = [{"chunk_id": "c1", "text": "x", "page_number": 1, "score": 0.8}]
    result = classify_claim_failure("partially_supported", 0.45, chunks)
    assert result["likely_failure_type"] == CONTEXT_MISINTERPRETATION


def test_low_similarity_with_strong_retrieval_means_unsupported_generation():
    chunks = [{"chunk_id": "c1", "text": "x", "page_number": 1, "score": 0.8}]
    result = classify_claim_failure("unsupported", 0.1, chunks)
    assert result["likely_failure_type"] == UNSUPPORTED_GENERATION


def test_low_similarity_with_moderate_retrieval_means_insufficient_context():
    chunks = [{"chunk_id": "c1", "text": "x", "page_number": 1, "score": 0.45}]
    result = classify_claim_failure("unsupported", 0.1, chunks)
    assert result["likely_failure_type"] == INSUFFICIENT_CONTEXT


@patch("app.services.failure_classifier.embed_texts")
def test_diverging_relevant_chunks_means_conflicting_evidence(mock_embed):
    mock_embed.side_effect = lambda texts: [[1.0, 0.0], [0.0, 1.0]][: len(texts)]
    chunks = [
        {"chunk_id": "c1", "text": "a", "page_number": 1, "score": 0.7},
        {"chunk_id": "c2", "text": "b", "page_number": 2, "score": 0.7},
    ]
    result = classify_claim_failure("unsupported", 0.1, chunks)
    assert result["likely_failure_type"] == CONFLICTING_EVIDENCE


def test_dominant_failure_picks_majority():
    claims = [
        {"likely_failure_type": UNSUPPORTED_GENERATION},
        {"likely_failure_type": UNSUPPORTED_GENERATION},
        {"likely_failure_type": RETRIEVAL_FAILURE},
    ]
    assert determine_dominant_failure(claims) == UNSUPPORTED_GENERATION


def test_dominant_failure_tie_break_priority():
    claims = [{"likely_failure_type": RETRIEVAL_FAILURE}, {"likely_failure_type": CONTEXT_MISINTERPRETATION}]
    assert determine_dominant_failure(claims) == RETRIEVAL_FAILURE


def test_dominant_failure_none_when_all_supported():
    claims = [{"likely_failure_type": None}, {"likely_failure_type": None}]
    assert determine_dominant_failure(claims) is None
