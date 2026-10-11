from unittest.mock import MagicMock, patch

from app.services.failure_classifier import CONTEXT_MISINTERPRETATION, RETRIEVAL_FAILURE, UNSUPPORTED_GENERATION
from app.services.mitigation import apply_mitigation


class FakeLLM:
    def generate(self, system_prompt, user_prompt):
        return "fake revised answer"


@patch("app.services.mitigation.get_llm_provider", return_value=FakeLLM())
@patch("app.services.mitigation.vector_store")
@patch("app.services.mitigation.embed_query", return_value=[0.1, 0.2])
def test_unsupported_generation_keeps_top_k_unchanged(mock_embed, mock_vs, mock_llm):
    mock_vs.query.return_value = [{"chunk_id": "c1", "text": "x", "page_number": 1, "score": 0.9}]
    result = apply_mitigation("doc1", "question?", UNSUPPORTED_GENERATION, base_top_k=5)
    assert result["strategy_used"] == "require_claim_support"
    mock_vs.query.assert_called_with("doc1", [0.1, 0.2], top_k=5)


@patch("app.services.mitigation.get_llm_provider", return_value=FakeLLM())
@patch("app.services.mitigation.vector_store")
@patch("app.services.mitigation.embed_query", return_value=[0.1, 0.2])
def test_retrieval_failure_doubles_top_k(mock_embed, mock_vs, mock_llm):
    mock_vs.query.return_value = [{"chunk_id": "c1", "text": "x", "page_number": 1, "score": 0.9}]
    result = apply_mitigation("doc1", "question?", RETRIEVAL_FAILURE, base_top_k=5)
    assert result["strategy_used"] == "retrieve_additional_chunks"
    mock_vs.query.assert_called_with("doc1", [0.1, 0.2], top_k=10)


@patch("app.services.mitigation.get_llm_provider", return_value=FakeLLM())
@patch("app.services.mitigation.vector_store")
@patch("app.services.mitigation.embed_query", return_value=[0.1, 0.2])
def test_misinterpretation_uses_stricter_prompt_strategy(mock_embed, mock_vs, mock_llm):
    mock_vs.query.return_value = [{"chunk_id": "c1", "text": "x", "page_number": 1, "score": 0.9}]
    result = apply_mitigation("doc1", "question?", CONTEXT_MISINTERPRETATION, base_top_k=5)
    assert result["strategy_used"] == "stricter_grounded_prompt"


@patch("app.services.mitigation.get_llm_provider", return_value=FakeLLM())
@patch("app.services.mitigation.vector_store")
@patch("app.services.mitigation.embed_query", return_value=[0.1, 0.2])
def test_no_retrieved_chunks_returns_fallback_message(mock_embed, mock_vs, mock_llm):
    mock_vs.query.return_value = []
    result = apply_mitigation("doc1", "question?", UNSUPPORTED_GENERATION, base_top_k=5)
    assert result["retrieved_chunks"] == []
    assert "No relevant context" in result["revised_answer"]
