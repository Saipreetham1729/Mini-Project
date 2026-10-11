import logging
from typing import Optional

from openai import OpenAI

from app.config import settings
from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    Works with OpenAI's API and any OpenAI-compatible endpoint (Groq,
    Together, OpenRouter, etc.) -- just point LLM_BASE_URL and
    LLM_MODEL_NAME at that provider in your .env, no code changes needed.
    """

    def __init__(self):
        if not settings.llm_api_key:
            raise ValueError(
                "LLM_API_KEY is not set. Add it to backend/.env, or set "
                "LLM_PROVIDER=mock in .env to test without a real API key."
            )
        self._client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

    def generate(self, system_prompt: str, user_prompt: str, temperature: Optional[float] = None) -> str:
        kwargs = {} if temperature is None else {"temperature": temperature}
        response = self._client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=settings.llm_request_timeout,
            **kwargs,
        )
        content = response.choices[0].message.content
        if content is None:
            logger.warning("LLM returned no content for a request.")
            return ""
        return content
