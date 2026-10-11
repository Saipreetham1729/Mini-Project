from app.services.claim_extractor import extract_claims


def test_splits_into_sentences():
    answer = "RAG grounds answers in evidence. It reduces hallucination."
    claims = extract_claims(answer)
    assert len(claims) == 2
    assert claims[0]["text"] == "RAG grounds answers in evidence."
    assert claims[1]["text"] == "It reduces hallucination."


def test_skips_trivial_fragments():
    answer = "Yes. RAG grounds answers in evidence."
    claims = extract_claims(answer)
    assert len(claims) == 1
    assert "Yes" not in claims[0]["text"]


def test_empty_answer_returns_no_claims():
    assert extract_claims("") == []
    assert extract_claims("   ") == []


def test_claim_ids_are_unique():
    answer = "First claim here. Second claim here. Third claim here."
    claims = extract_claims(answer)
    ids = [c["claim_id"] for c in claims]
    assert len(ids) == len(set(ids))
