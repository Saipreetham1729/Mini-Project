from unittest.mock import patch

from app.services.evidence_verifier import verify_claims

FAKE_MAP = {
    "RAG reduces hallucinations by grounding answers in evidence.": [1.0, 0.0, 0.0],
    "RAG grounds generated answers in retrieved evidence to reduce hallucination.": [0.98, 0.05, 0.0],
    "The Eiffel Tower is in Paris.": [0.0, 0.0, 1.0],
    "Chunking splits documents into smaller pieces for retrieval.": [0.0, 1.0, 0.0],
}


def fake_embed_texts(texts):
    return [FAKE_MAP[t] for t in texts]


@patch("app.services.evidence_verifier.embed_texts", side_effect=fake_embed_texts)
def test_high_similarity_claim_is_supported(mock_embed):
    claims = [{"claim_id": "c0", "text": "RAG reduces hallucinations by grounding answers in evidence."}]
    evidence = [
        {
            "chunk_id": "c1",
            "text": "RAG grounds generated answers in retrieved evidence to reduce hallucination.",
            "page_number": 1,
        },
        {"chunk_id": "c2", "text": "Chunking splits documents into smaller pieces for retrieval.", "page_number": 2},
    ]
    result = verify_claims(claims, evidence)
    assert result[0]["status"] == "supported"
    assert result[0]["best_evidence_chunk_id"] == "c1"


@patch("app.services.evidence_verifier.embed_texts", side_effect=fake_embed_texts)
def test_low_similarity_claim_is_unsupported(mock_embed):
    claims = [{"claim_id": "c1", "text": "The Eiffel Tower is in Paris."}]
    evidence = [
        {
            "chunk_id": "c1",
            "text": "RAG grounds generated answers in retrieved evidence to reduce hallucination.",
            "page_number": 1,
        }
    ]
    result = verify_claims(claims, evidence)
    assert result[0]["status"] == "unsupported"


def test_no_evidence_marks_all_unsupported():
    claims = [{"claim_id": "c0", "text": "Any claim."}]
    result = verify_claims(claims, [])
    assert result[0]["status"] == "unsupported"
    assert result[0]["similarity_score"] == 0.0
