from functools import lru_cache

from app.config import settings
from app.services.llm.base import LLMProvider
from app.services.llm.mock_provider import MockProvider


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    provider = settings.llm_provider.lower()

    if provider == "mock":
        return MockProvider()

    if provider == "openai":
        from app.services.llm.openai_provider import OpenAIProvider  # local import: avoid requiring the openai package when unused

        return OpenAIProvider()

    raise ValueError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use 'mock' or 'openai'.")
