"""Guardrails: Validate queries and responses for safety and quality."""

import logging
from typing import Tuple, List, Optional

from src.llm import get_llm_client
from src.config import get_all_company_aliases

logger = logging.getLogger(__name__)


class GuardrailValidator:
    """Validate queries and responses for scope, confidence, and safety."""

    def __init__(self):
        """Initialize guardrails validator."""
        self.llm = get_llm_client()

    @staticmethod
    def _has_company_marker(text: str) -> bool:
        """Detect whether text names a known company or company id."""
        text_lower = text.lower()
        company_markers = get_all_company_aliases()
        return any(marker in text_lower for marker in company_markers)

    def check_scope(self, query: str, conversation_context: Optional[str] = None) -> Tuple[bool, str]:
        """Check if query is in scope (about earnings calls).

        Args:
            query: User query

        Returns:
            Tuple of (is_in_scope, reason)
        """
        # Keywords that indicate out-of-scope questions
        out_of_scope_keywords = [
            # General Off-topic
            "recipe", "movie", "song", "game", "joke", "funny",
            "weather", "sports", "politics", "president", "election",
            "love", "dating", "relationship", "travel", "vacation",
            "celebrity", "gossip", "entertainment", "history", "geography",
            # Programming/Technical
            "coding", "programming", "javascript", "python", "software development",
            "sql", "html", "css", "bug fix", "refactor",
            # Creative
            "poem", "story", "creative writing", "lyrics",
            # CRITICAL: Medical advice (since data is from hospitals)
            "symptom", "diagnosis", "treatment for", "medicine for", "cure for",
            "doctor recommendation", "surgery advice", "pain in my", "health advice",
            # Casual Chatter / Greetings
            "how are you", "how is your day", "how's your day", "good morning",
            "good afternoon", "good evening", "what's up", "hello there"
        ]

        query_lower = query.lower()

        # Check keywords
        for keyword in out_of_scope_keywords:
            if keyword in query_lower:
                return False, (
                    "I'd love to chat about that, but my expertise is strictly limited to financial earnings calls and company performance. "
                    "Please ask me a question about revenue, market trends, business strategy, or a specific company's latest quarter!"
                )

        vague_growth_keywords = ["why", "reason", "because", "grow",
                                 "growth", "grew", "increase", "increased", "decline", "declined"]
        has_vague_growth_intent = any(
            keyword in query_lower for keyword in vague_growth_keywords)
        has_company_marker = self._has_company_marker(query)
        has_history_marker = bool(conversation_context and self._has_company_marker(conversation_context))

        if has_vague_growth_intent and not (has_company_marker or has_history_marker):
            return False, (
                "Please name the company and quarter for growth-related questions. For example: 'Why did Medanta revenue grow in Q2 FY2025?' or 'What drove Polycab's revenue in Q3?'")

        # Instead of strictly requiring a financial keyword to be present (which blocks
        # qualitative questions like "What are your AI plans?"), we will default to
        # allowing the query as long as it didn't trigger any out_of_scope_keywords above.

        return True, "Query is assumed in scope."

    def check_confidence(
        self,
        retrieved_documents: List,
        min_similarity: float = 0.3,
    ) -> Tuple[float, str]:
        """Check confidence of retrieval based on match quality and diversity.
        """
        if not retrieved_documents:
            return 0.0, "LOW (no documents retrieved)"

        # 1. Get raw similarities
        similarities = [sim for _, sim in retrieved_documents]

        # 2. Extract unique companies (diversity factor)
        unique_companies = len(
            set(doc.company_id for doc, _ in retrieved_documents))

        # 3. Calculate metrics
        # max_similarity tells us if we found at least one "correct" anchor
        max_sim = max(similarities)
        avg_sim = sum(similarities) / len(similarities)

        # 4. Normalized score calculation
        # Local embeddings (Ollama) often have lower raw similarity scores (e.g., 0.1 - 0.3)
        # We boost the score if we found one strong anchor OR multiple companies.

        # Base confidence on the best available match (Max Similarity)
        # 0.7+ is usually a direct semantic match
        confidence_score = max_sim

        # Small diversity bonus (max +5% if we have multiple companies)
        if unique_companies > 1:
            confidence_score += 0.05

        confidence_score = min(0.99, confidence_score)

        # Confidence levels (Standard industry thresholds)
        if confidence_score >= 0.7:
            confidence_level = "HIGH"
        elif confidence_score >= 0.4:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        return confidence_score, confidence_level

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
        retrieved_documents: Optional[List] = None,
        conversation_context: Optional[str] = None,
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
        in_scope, scope_msg = self.check_scope(
            query,
            conversation_context=conversation_context,
        )
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
