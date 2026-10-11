from typing import Optional

from app.services.llm.base import LLMProvider


class MockProvider(LLMProvider):
    """
    Deterministic, offline stand-in for a real LLM.

    Useful for testing the RAG pipeline's plumbing (retrieval, context
    building, response shape) without an API key or network access.

    IMPORTANT: this does not reason over the context at all -- never use it
    to produce results that go into the research evaluation. It exists
    purely so the system runs end-to-end before you've wired up a real key.
    """

    def generate(self, system_prompt: str, user_prompt: str, temperature: Optional[float] = None) -> str:
        return (
            "[MOCK RESPONSE -- no real LLM was called] "
            "This is a placeholder answer; it does not reason over the retrieved context. "
            "Set LLM_PROVIDER=openai and LLM_API_KEY in your .env to get real answers."
        )
