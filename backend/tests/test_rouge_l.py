from app.services.metrics.rouge_l import score_claims


def test_near_verbatim_claim_scores_high():
    claims = [{"claim_id": "c0", "text": "RAG grounds generated answers in retrieved evidence."}]
    evidence = [
        {
            "chunk_id": "e1",
            "text": "RAG grounds generated answers in retrieved evidence to reduce hallucination.",
            "page_number": 1,
        }
    ]
    result = score_claims(claims, evidence)
    assert result[0]["score"] > 0.7
    assert result[0]["best_evidence_chunk_id"] == "e1"


def test_unrelated_claim_scores_low():
    claims = [{"claim_id": "c1", "text": "The moon is made of cheese according to folklore."}]
    evidence = [{"chunk_id": "e1", "text": "RAG grounds generated answers in retrieved evidence.", "page_number": 1}]
    result = score_claims(claims, evidence)
    assert result[0]["score"] < 0.3


def test_no_evidence_returns_zero():
    claims = [{"claim_id": "c0", "text": "Some claim."}]
    result = score_claims(claims, [])
    assert result[0]["score"] == 0.0
    assert result[0]["best_evidence_chunk_id"] is None


def test_picks_best_matching_chunk_among_several():
    claims = [{"claim_id": "c0", "text": "RAG grounds answers in retrieved evidence."}]
    evidence = [
        {"chunk_id": "irrelevant", "text": "Cats are mammals with retractable claws.", "page_number": 1},
        {"chunk_id": "relevant", "text": "RAG grounds generated answers in retrieved evidence.", "page_number": 2},
    ]
    result = score_claims(claims, evidence)
    assert result[0]["best_evidence_chunk_id"] == "relevant"
