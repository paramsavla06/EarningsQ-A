from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from earnings_qa.rag.ingestion import Document
from earnings_qa.config import TOP_K_RETRIEVAL

class RetrieverBackend(ABC):
    """Abstract interface for document retrieval backends."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        company_ids: Optional[List[str]] = None,
        quarters: Optional[List[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve top-K documents based on semantic similarity.

        Args:
            query: The search query.
            top_k: Number of documents to retrieve.
            company_ids: Optional list of company filters.
            quarters: Optional list of quarter filters.

        Returns:
            List of tuples containing (Document, similarity_score).
        """
        pass

    @abstractmethod
    async def retrieve_async(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        company_ids: Optional[List[str]] = None,
        quarters: Optional[List[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve top-K documents asynchronously."""
        pass

    @abstractmethod
    def retrieve_by_filters(
        self,
        company_ids: Optional[List[str]] = None,
        quarters: Optional[List[str]] = None,
    ) -> List[Document]:
        """Retrieve documents exactly matching the provided filters.

        Args:
            company_ids: Optional list of company filters.
            quarters: Optional list of quarter filters.

        Returns:
            List of matching documents.
        """
        pass

    @abstractmethod
    def get_all_documents(self) -> List[Document]:
        """Return all documents in the index (useful for listing/debugging)."""
        pass

    @abstractmethod
    def _detect_company_intents(self, query: str) -> List[str]:
        """Detect all companies requested in the query."""
        pass

    @abstractmethod
    def _detect_quarter_intents(self, query: str) -> List[str]:
        """Detect all quarters requested in the query."""
        pass

    def format_context(
        self,
        documents: List[Tuple[Document, float]],
        max_chars: int = 3000,
    ) -> str:
        """Format retrieved documents into a context string for the LLM.

        Args:
            documents: List of (Document, similarity) tuples.
            max_chars: Maximum characters to include.

        Returns:
            Formatted context string.
        """
        if not documents:
            return "No relevant context found."

        context_lines = []
        total_chars = 0

        for i, (doc, similarity) in enumerate(documents):
            header = f"[{doc.company_id} {doc.quarter}{doc.year} - Match: {similarity:.2%}]"
            content = doc.content[:500]

            entry = f"{header}\n{content}\n"

            if total_chars + len(entry) > max_chars:
                break

            context_lines.append(entry)
            total_chars += len(entry)

        return "\n---\n".join(context_lines)
