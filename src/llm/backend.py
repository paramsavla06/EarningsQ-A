from abc import ABC, abstractmethod
from src.config import LLM_MAX_TOKENS

class LLMBackend(ABC):
    """Abstract interface for LLM generation backends."""

    @abstractmethod
    def answer_question(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            system_prompt: System context or instructions.
            user_message: The user's query.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated response string.
        """
        pass

    @abstractmethod
    def get_token_count_estimate(self, text: str) -> int:
        """Estimate the token count for a text snippet.

        Args:
            text: The text to estimate.

        Returns:
            Approximate token count.
        """
        pass
