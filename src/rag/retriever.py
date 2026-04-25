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

    def _detect_company_intents(self, query: str) -> List[str]:
        """Detect all companies requested."""
        query_lower = query.lower()
        mapping = {
            "532400": ["bsoft", "birlasoft", "birla soft", "birla soft limited", "532400"],
            "542652": ["polycab", "wires", "542652"],
            "543654": ["medanta", "global health", "543654"],
            "544350": ["agarwal", "544350"]
        }
        found = []
        for cid, keywords in mapping.items():
            if any(k in query_lower for k in keywords):
                found.append(cid)
        return found

    def _detect_quarter_intents(self, query: str) -> List[str]:
        """Detect all quarters requested."""
        query_upper = query.upper()
        found = []
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            if q in query_upper:
                found.append(q)
        return found

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        company_ids: Optional[List[str]] = None,
        quarters: Optional[List[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """Retrieve top-K documents with smart intent detection.
        """
        query_lower = query.lower()
        if self.index is None:
            logger.warning("No index loaded")
            return []

        # 1. Intent Detection: Did the user name a company or quarter in the text?
        if not company_ids:
            company_ids = self._detect_company_intents(query)

        if not quarters:
            quarters = self._detect_quarter_intents(query)

        # 2. Embed query
        query_embedding = self.embedding_pipeline.embed_text(query)
        query_embedding = np.array([query_embedding]).astype(np.float32)

        # 3. Search
        search_k = max(100, top_k * 4)
        distances, indices = self.index.search(
            query_embedding, search_k)  # type: ignore[call-arg]

        # 4. Filter and Boost
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                dist_sq = distances[0][i]

                # With normalized vectors, Cosine Similarity = 1 - (squared_l2_dist / 2)
                similarity = max(0.0, 1.0 - (dist_sq / 2.0))

                # --- NEW: Keyword Boosting ---
                # If chunk contains specific keywords the user asked for, boost its similarity
                boost = 0.0
                content_lower = doc.content.lower()

                # Check for financial metrics
                if "revenue" in query_lower and ("revenue" in content_lower or "income" in content_lower):
                    boost += 0.1
                if "ebitda" in query_lower and "ebitda" in content_lower:
                    boost += 0.1
                if "pat " in query_lower or "profit" in query_lower:
                    if "pat " in content_lower or "profit" in content_lower:
                        boost += 0.15

                # CRITICAL: Prioritize 'Consolidated' + Hard Numbers
                if "consolidated" in content_lower:
                    boost += 0.1
                    # Double boost if both 'consolidated' and numbers exist
                    if "%" in content_lower or "₹" in content_lower or "rs" in content_lower:
                        boost += 0.2

                similarity = min(0.99, similarity + boost)
                # -----------------------------

                if company_ids and doc.company_id not in company_ids:
                    continue
                if quarters and doc.quarter not in quarters:
                    continue

                candidates.append((doc, similarity))

        if not candidates:
            return []

        # --- Decision: Focus vs Diversity ---

        # If we have a specific company (via filter or intent), prioritize FOCUS
        if company_ids and len(company_ids) == 1:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:top_k]

        # Otherwise, use Diversity Logic for generic queries
        by_company = {}
        for doc, sim in candidates:
            if doc.company_id not in by_company:
                by_company[doc.company_id] = []
            by_company[doc.company_id].append((doc, sim))

        final_results = []
        seen_indices = set()

        for cid in sorted(by_company.keys()):
            by_company[cid].sort(key=lambda x: x[1], reverse=True)
            best_match = by_company[cid][0]
            final_results.append(best_match)
            for i, (d, s) in enumerate(candidates):
                if d == best_match[0]:
                    seen_indices.add(i)
                    break

        candidates.sort(key=lambda x: x[1], reverse=True)
        for i, (doc, sim) in enumerate(candidates):
            if len(final_results) >= top_k:
                break
            if i not in seen_indices:
                final_results.append((doc, sim))
                seen_indices.add(i)

        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results

    def retrieve_by_filters(
        self,
        company_ids: Optional[List[str]] = None,
        quarters: Optional[List[str]] = None,
    ) -> List[Document]:
        """Retrieve documents by company and quarter filters.

        Args:
            company_ids: Optional list of company filters
            quarters: Optional list of quarter filters

        Returns:
            List of matching documents
        """
        results = []

        for doc in self.documents:
            if company_ids and doc.company_id not in company_ids:
                continue
            if quarters and doc.quarter not in quarters:
                continue
            results.append(doc)

        logger.debug(
            f"Retrieved {len(results)} documents for filters: companies={company_ids}, quarters={quarters}")

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
