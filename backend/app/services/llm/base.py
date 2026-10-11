from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """
    Common interface every LLM backend implements, so the rest of the
    pipeline never needs to know which provider is behind it.
    """

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: Optional[float] = None) -> str:
        """
        Return the model's text completion for the given prompts.

        temperature is optional and provider-specific: pass None to use the
        provider's own default (deterministic-ish, low-temperature answers
        for the main pipeline), or a value like 0.7-1.0 for the independent,
        higher-variance samples self-consistency scoring needs.
        """
        raise NotImplementedError
