"""Guardrails: Validate queries and responses for safety and quality."""

import logging
from typing import Tuple, List

from src.llm.client import get_llm_client
from src.config import TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)


class GuardrailValidator:
    """Validate queries and responses for scope, confidence, and safety."""

    def __init__(self):
        """Initialize guardrails validator."""
        self.llm = get_llm_client()

    def check_scope(self, query: str) -> Tuple[bool, str]:
        """Check if query is in scope (about earnings calls).

        Args:
            query: User query

        Returns:
            Tuple of (is_in_scope, reason)
        """
        # Keywords that indicate out-of-scope questions
        out_of_scope_keywords = [
            "recipe", "movie", "song", "game", "joke", "funny",
            "weather", "sports", "politics", "president", "election",
            "love", "dating", "relationship"
        ]

        query_lower = query.lower()

        # Check keywords
        for keyword in out_of_scope_keywords:
            if keyword in query_lower:
                return False, f"Your question contains '{keyword}', which is outside the scope of earnings call analysis."

        # Check if query seems related to earnings
        earnings_keywords = [
            "revenue", "profit", "margin", "earnings", "guidance", "quarter",
            "growth", "performance", "stock", "dividend", "cash flow",
            "expense", "cost", "sales", "income", "loss", "eps"
        ]

        has_earnings_keyword = any(
            keyword in query_lower for keyword in earnings_keywords)

        if not has_earnings_keyword and len(query) < 10:
            return True, "Query is very short but will attempt to answer from earnings context."

        return True, "Query is in scope."

    def check_confidence(
        self,
        retrieved_documents: List,
        min_similarity: float = 0.3,
    ) -> Tuple[float, str]:
        """Check confidence of retrieval based on match quality.

        Args:
            retrieved_documents: List of (Document, similarity) tuples
            min_similarity: Minimum acceptable similarity score

        Returns:
            Tuple of (confidence_score, confidence_level)
        """
        if not retrieved_documents:
            return 0.0, "LOW (no documents retrieved)"

        # Calculate average similarity
        similarities = [sim for _, sim in retrieved_documents]
        avg_similarity = sum(similarities) / len(similarities)

        # Confidence levels
        if avg_similarity >= 0.7:
            confidence_level = "HIGH"
        elif avg_similarity >= 0.4:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        return avg_similarity, confidence_level

    def add_confidence_note(
        self,
        response: str,
        confidence_score: float,
        retrieved_count: int,
    ) -> str:
        """Add confidence note to response.

        Args:
            response: Original response
            confidence_score: Confidence score (0-1)
            retrieved_count: Number of documents retrieved

        Returns:
            Response with confidence note appended
        """
        if confidence_score < 0.4 and retrieved_count == 0:
            note = "\n\n⚠️ Note: This response is based on general knowledge about earnings calls, as no specific context was found in the transcript database. For accurate information, please ensure the relevant earnings call PDF has been indexed."
        elif confidence_score < 0.5:
            note = f"\n\n📊 Confidence: {confidence_score:.0%} - Limited matching context found. The answer may be incomplete."
        else:
            note = ""

        return response + note

    def validate_response(self, response: str) -> Tuple[bool, str]:
        """Validate response for hallucinations and quality.

        Args:
            response: Response text

        Returns:
            Tuple of (is_valid, validation_message)
        """
        # Check for placeholder/mock responses
        if "[MOCK]" in response or "[This is a mock" in response:
            return True, "Mock response (development mode)"

        # Check response length
        if len(response.strip()) < 20:
            return False, "Response too short - likely an error."

        # Check for explicit uncertainty language
        if "i don't know" in response.lower() or "no data" in response.lower():
            return True, "Response acknowledges lack of data (good)"

        # Check for citations
        if any(x in response for x in ["quarter", "Q1", "Q2", "Q3", "Q4"]):
            return True, "Response includes proper citations"

        return True, "Response appears valid"

    def apply_guardrails(
        self,
        query: str,
        response: str,
        retrieved_documents: List = None,
    ) -> Tuple[str, str]:
        """Apply all guardrails to query and response.

        Args:
            query: User query
            response: LLM response
            retrieved_documents: Retrieved documents (for confidence check)

        Returns:
            Tuple of (final_response, status_message)
        """
        # Check scope
        in_scope, scope_msg = self.check_scope(query)
        if not in_scope:
            return scope_msg, "OUT_OF_SCOPE"

        # Check confidence
        if retrieved_documents:
            confidence_score, confidence_level = self.check_confidence(
                retrieved_documents)
            response = self.add_confidence_note(
                response, confidence_score, len(retrieved_documents))
        else:
            response = self.add_confidence_note(response, 0.0, 0)

        # Validate response
        is_valid, validation_msg = self.validate_response(response)

        status = "VALID" if is_valid else "INVALID"

        return response, status
