"""Google Gemini client wrapper with retry logic."""

import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

try:
    from google import genai
except ImportError:
    import genai

from src.config import GEMINI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, USE_MOCK_LLM

logger = logging.getLogger(__name__)


class MockLLMClient:
    """Mock LLM client for development without API calls."""

    def __init__(self, api_key: str = None, model: str = LLM_MODEL):
        """Initialize mock client.

        Args:
            api_key: Ignored (mock doesn't use API)
            model: Model name (for logging only)
        """
        self.model = model
        logger.info("⚠️  MOCK MODE ENABLED - No API calls will be made")

    def answer_question(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        """Generate a mock response.

        Args:
            system_prompt: System prompt (ignored in mock)
            user_message: User's question
            max_tokens: Max output tokens (ignored in mock)

        Returns:
            Mock response based on question
        """
        logger.info(f"Mock response to: {user_message[:50]}...")

        # Simple mock responses based on keywords
        question_lower = user_message.lower()

        if "revenue" in question_lower:
            return "[MOCK] Based on the earnings call transcript:\n\nRevenue for the quarter was strong at $119.6 billion, representing 2% year-over-year growth. This reflects solid demand across our product categories and growing services adoption.\n\nKey drivers:\n- iPhone: $69.3B (strong despite market maturity)\n- Services: $22.3B (our highest ever, up 16% YoY)\n- Mac: $8.5B (benefiting from M3 chip adoption)\n- Wearables: $14.1B (excellent response to Apple Watch Series 9)\n\n[This is a MOCK response for development]"
        elif "margin" in question_lower or "profit" in question_lower:
            return "[MOCK] Gross margin came in at 46.2%, up 150 basis points year-over-year. This strong margin expansion was driven by:\n\n1. Operational efficiency improvements\n2. Favorable product mix (high-margin services growing 16% YoY)\n3. Economies of scale\n\nOperating margin remained healthy despite R&D investments in AI capabilities and advanced materials.\n\n[This is a MOCK response for development]"
        elif "guidance" in question_lower or "outlook" in question_lower:
            return "[MOCK] Forward guidance for Q2:\n\n- Expected revenue: $91-95 billion\n- Growth: Modest, driven by new product launches\n- Key catalysts: New product announcements, Services momentum\n- China: Optimistic about momentum despite competitive environment\n\nWe expect mid-single-digit revenue growth for FY2024 with strong Services momentum continuing.\n\n[This is a MOCK response for development]"
        elif "china" in question_lower:
            return "[MOCK] Greater China performance remains resilient despite competitive pressures. The iPhone 15 lineup, particularly Pro models with advanced camera capabilities, is performing well in urban markets. We're seeing strong demand momentum and expect this to continue into calendar 2024.\n\n[This is a MOCK response for development]"
        else:
            return f"[MOCK] Based on the earnings call: {user_message}\n\nThis is a mock response for development and testing. In production mode, this would retrieve relevant sections from the earnings transcript and provide a context-grounded answer.\n\nNote: To enable real Gemini API calls, set USE_MOCK_LLM=false in your .env file.\n\n[This is a MOCK response for development]"

    def get_token_count_estimate(self, text: str) -> int:
        """Rough estimate of token count.

        Args:
            text: Text to estimate

        Returns:
            Approximate token count
        """
        return len(text) // 4


class LLMClient:
    """Wrapper around Google Gemini client with retry logic and error handling."""

    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = LLM_MODEL):
        """Initialize Gemini client.

        Args:
            api_key: Google Gemini API key
            model: Model name (e.g., 'gemini-2.0-flash')
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(lambda e: "500" in str(
            e) or "unavailable" in str(e).lower()),
        reraise=True,
    )
    def _call_api(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        """Call Gemini API with retry logic.

        Args:
            system_prompt: System message to set behavior
            user_message: User question/message
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response

        Returns:
            Response text from LLM

        Raises:
            Exception: If API call fails after retries
        """
        # Combine system prompt with user message for Gemini
        full_message = f"{system_prompt}\n\n{user_message}"

        response = self.client.models.generate_content(
            model=self.model,
            contents=full_message,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

        return response.text

    def answer_question(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        """Answer a question using the Gemini LLM.

        Args:
            system_prompt: System prompt for context/behavior
            user_message: User's question
            max_tokens: Maximum tokens in response

        Returns:
            Answer from Gemini
        """
        try:
            logger.info(f"Calling Gemini: {self.model}")
            response = self._call_api(
                system_prompt=system_prompt,
                user_message=user_message,
                max_tokens=max_tokens,
            )
            logger.info("Gemini call successful")
            return response
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def get_token_count_estimate(self, text: str) -> int:
        """Rough estimate of token count (4 chars ≈ 1 token).

        Args:
            text: Text to estimate

        Returns:
            Approximate token count
        """
        return len(text) // 4


def get_llm_client() -> LLMClient:
    """Factory function to get appropriate LLM client based on config.

    Returns:
        MockLLMClient if USE_MOCK_LLM=true, otherwise LLMClient
    """
    if USE_MOCK_LLM:
        return MockLLMClient()
    return LLMClient()
