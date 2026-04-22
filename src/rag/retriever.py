"""Retriever: Query FAISS index and retrieve relevant documents."""

import logging
import numpy as np
from typing import List, Optional, Tuple

from src.rag.ingestion import Document
from src.rag.embeddings import EmbeddingPipeline
from src.config import TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieve documents from FAISS index based on query."""

    def __init__(self, embedding_pipeline: EmbeddingPipeline):
        """Initialize retriever.

        Args:
            embedding_pipeline: Initialized embedding pipeline with loaded index
        """
        self.embedding_pipeline = embedding_pipeline
        self.index = embedding_pipeline.index
        self.documents = embedding_pipeline.documents
        self.embeddings = embedding_pipeline.embeddings

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        company_id: Optional[str] = None,
        quarter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve top-K most similar documents for query.

        Args:
            query: Query text
            top_k: Number of documents to retrieve
            company_id: Optional company filter
            quarter: Optional quarter filter

        Returns:
            List of (Document, similarity_score) tuples, sorted by relevance
        """
        if self.index is None:
            logger.warning("No index loaded")
            return []

        # Embed query
        query_embedding = self.embedding_pipeline.embed_text(query)
        query_embedding = np.array([query_embedding]).astype(np.float32)

        # Search index
        distances, indices = self.index.search(
            query_embedding, top_k * 2)  # Get more to filter

        # Convert distances to similarities (lower distance = higher similarity)
        # For L2 distance, similarity can be 1 / (1 + distance)
        results = []

        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                distance = distances[0][i]
                similarity = 1.0 / (1.0 + distance)

                # Apply filters
                if company_id and doc.company_id != company_id:
                    continue
                if quarter and doc.quarter != quarter:
                    continue

                results.append((doc, similarity))

        # Sort by similarity (descending) and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        logger.info(
            f"Retrieved {len(results)} documents for query: {query[:50]}...")

        return results

    def retrieve_by_filters(
        self,
        company_id: Optional[str] = None,
        quarter: Optional[str] = None,
    ) -> List[Document]:
        """Retrieve documents by company and quarter filters.

        Args:
            company_id: Optional company filter
            quarter: Optional quarter filter

        Returns:
            List of matching documents
        """
        results = []

        for doc in self.documents:
            if company_id and doc.company_id != company_id:
                continue
            if quarter and doc.quarter != quarter:
                continue
            results.append(doc)

        logger.info(
            f"Retrieved {len(results)} documents for filters: company={company_id}, quarter={quarter}")

        return results

    def format_context(
        self,
        documents: List[Tuple[Document, float]],
        max_chars: int = 3000,
    ) -> str:
        """Format retrieved documents into context string for LLM.

        Args:
            documents: List of (Document, similarity) tuples
            max_chars: Maximum characters to include

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant context found."

        context_lines = []
        total_chars = 0

        for i, (doc, similarity) in enumerate(documents):
            # Format each document with metadata
            header = f"[{doc.company_id} {doc.quarter}{doc.year} - Match: {similarity:.2%}]"
            content = doc.content[:500]  # First 500 chars

            entry = f"{header}\n{content}\n"

            if total_chars + len(entry) > max_chars:
                break

            context_lines.append(entry)
            total_chars += len(entry)

        return "\n---\n".join(context_lines)
