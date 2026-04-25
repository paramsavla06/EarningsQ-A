"""Embedding pipeline: Embed documents and store in FAISS."""

import os
import logging
import time
import re
import requests
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple

try:
    import google.genai as genai
except ImportError:
    try:
        from google import genai
    except ImportError:
        import genai

import faiss

from earnings_qa.config import (
    GEMINI_API_KEY, EMBEDDING_MODEL, USE_MOCK_LLM,
    USE_OLLAMA_EMBED, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL,
)
from earnings_qa.rag.ingestion import Document

# Force the SDK to use our configured key, not any stale system GOOGLE_API_KEY
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Embedding dimensions by backend
# nomic-embed-text (Ollama) = 768, gemini-embedding-001 = 3072
EMBEDDING_DIM = 768 if USE_OLLAMA_EMBED else 3072


class EmbeddingPipeline:
    """Pipeline for embedding documents and creating FAISS index."""

    def __init__(self, api_key: str = GEMINI_API_KEY, embedding_model: str = EMBEDDING_MODEL):
        """Initialize embedding pipeline.

        Args:
            api_key: Gemini API key
            embedding_model: Embedding model name
        """
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.index = None
        self.documents = []
        self.embeddings = None

        if USE_OLLAMA_EMBED:
            logger.info(f"Using Ollama embeddings: {OLLAMA_EMBED_MODEL} @ {OLLAMA_BASE_URL}")
        elif not USE_MOCK_LLM:
            self.client = genai.Client(api_key=api_key)

    def _embed_via_ollama(self, text: str) -> np.ndarray:
        """Embed text using the local Ollama server."""
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=120,
        )
        response.raise_for_status()
        vals = response.json()["embedding"]
        vals = np.array(vals, dtype=np.float32)
        # Normalize vector
        faiss.normalize_L2(vals.reshape(1, -1))
        return vals

    def embed_text(self, text: str, max_retries: int = 8) -> np.ndarray:
        """Embed a single text using Gemini embedding model."""
        from earnings_qa.core.cache import cache
        
        if not text:
            return np.array([]).astype(np.float32)

        cached = cache.get_embedding(text)
        if cached is not None:
            return cached

        if USE_MOCK_LLM:
            res = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            cache.set_embedding(text, res)
            return res

        if USE_OLLAMA_EMBED:
            res = self._embed_via_ollama(text)
            cache.set_embedding(text, res)
            return res

        for attempt in range(max_retries + 1):
            try:
                result = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                )

                if hasattr(result, 'embeddings') and result.embeddings:
                    vals = result.embeddings[0].values
                elif isinstance(result, dict) and 'embedding' in result:
                    vals = result['embedding']
                else:
                    raise ValueError(f"Unknown embedding response format: {type(result)}")

                vals = np.array(vals).astype(np.float32)
                # Normalize query vector
                faiss.normalize_L2(vals.reshape(1, -1))
                cache.set_embedding(text, vals)
                return vals

            except Exception as e:
                err_str = str(e)

                # Daily quota exhausted — cannot recover by waiting seconds
                if "PerDay" in err_str:
                    raise RuntimeError(
                        "Gemini free-tier daily quota exhausted (1000 requests/day). "
                        "Wait until midnight Pacific Time and re-run --index, "
                        "or enable billing at https://console.cloud.google.com/billing"
                    ) from e

                # Per-minute / burst rate limit — read retry delay and sleep
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", err_str)
                    wait = float(match.group(1)) + 1.0 if match else (2 ** attempt) * 5
                    logger.warning(
                        f"Rate limited (429). Waiting {wait:.1f}s "
                        f"(retry {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait)
                    continue

                logger.error(f"Embedding error (attempt {attempt + 1}): {e}")
                raise

        raise RuntimeError(f"Failed to embed text after {max_retries} retries.")

    def embed_texts(self, texts: List[str], max_retries: int = 8) -> np.ndarray:
        """Embed a batch of texts using Gemini embedding model."""
        from earnings_qa.core.cache import cache

        if not texts:
            return np.array([]).astype(np.float32)

        # Check cache
        uncached_indices = []
        uncached_texts = []
        results = [None] * len(texts)

        for i, text in enumerate(texts):
            cached = cache.get_embedding(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if not uncached_texts:
            return np.vstack(results)

        if USE_MOCK_LLM:
            mock_res = np.random.randn(len(uncached_texts), EMBEDDING_DIM).astype(np.float32)
            for i, idx in enumerate(uncached_indices):
                cache.set_embedding(uncached_texts[i], mock_res[i])
                results[idx] = mock_res[i]
            return np.vstack(results)

        if USE_OLLAMA_EMBED:
            for i, idx in enumerate(uncached_indices):
                res = self._embed_via_ollama(uncached_texts[i])
                cache.set_embedding(uncached_texts[i], res)
                results[idx] = res
            return np.vstack(results)

        for attempt in range(max_retries + 1):
            try:
                result = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=uncached_texts,
                )

                embeddings = []
                if hasattr(result, 'embeddings') and result.embeddings:
                    embeddings = [emb.values for emb in result.embeddings]
                elif isinstance(result, list):
                    # Sometimes the API returns a list of dicts
                    embeddings = [item['embedding'] if isinstance(item, dict) and 'embedding' in item else item for item in result]
                else:
                    raise ValueError(f"Unknown embedding response format: {type(result)}")

                vals = np.array(embeddings).astype(np.float32)
                faiss.normalize_L2(vals)
                
                for i, idx in enumerate(uncached_indices):
                    cache.set_embedding(uncached_texts[i], vals[i])
                    results[idx] = vals[i]
                    
                return np.vstack(results)

            except Exception as e:
                err_str = str(e)

                if "PerDay" in err_str:
                    raise RuntimeError(
                        "Gemini free-tier daily quota exhausted (1000 requests/day)."
                    ) from e

                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", err_str)
                    wait = float(match.group(1)) + 1.0 if match else (2 ** attempt) * 5
                    logger.warning(
                        f"Rate limited (429). Waiting {wait:.1f}s "
                        f"(retry {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait)
                    continue

                logger.error(f"Embedding error (attempt {attempt + 1}): {e}")
                raise

        raise RuntimeError(f"Failed to embed texts after {max_retries} retries.")

    def embed_documents(self, documents: List[Document], batch_size: int = 100) -> np.ndarray:
        """Embed a batch of documents using true batch requests.

        Args:
            documents: List of documents to embed
            batch_size: Number of documents to embed at once

        Returns:
            Embedding matrix (n_documents x embedding_dim)
        """
        all_embeddings = []

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            logger.info(f"Embedding documents {i+1} to {min(i+batch_size, len(documents))} / {len(documents)}")
            
            texts = [doc.content[:2000] for doc in batch_docs]
            batch_embeddings = self.embed_texts(texts)
            
            if batch_embeddings.ndim == 1:
                batch_embeddings = batch_embeddings.reshape(1, -1)
                
            all_embeddings.append(batch_embeddings)

        embeddings_array = np.vstack(all_embeddings) if all_embeddings else np.array([]).astype(np.float32)
        logger.info(
            f"Created {embeddings_array.shape[0]} embeddings of dimension {embeddings_array.shape[1] if embeddings_array.size > 0 else 0}")

        return embeddings_array

    def create_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """Create FAISS index from embeddings.

        Args:
            embeddings: Embedding matrix (n x d)

        Returns:
            FAISS index
        """
        embedding_dim = embeddings.shape[1]

        # Use L2 distance (Euclidean distance) on normalized vectors
        # This makes L2 distance equivalent to Cosine Similarity
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings)

        logger.info(
            f"Created FAISS index with {index.ntotal} vectors of dimension {embedding_dim}")

        return index

    def build_index(self, documents: List[Document], output_path: Path = None) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
        """Build complete embedding index from documents.

        Args:
            documents: List of documents
            output_path: Optional path to save index to disk

        Returns:
            Tuple of (FAISS index, embeddings array)
        """
        logger.info(f"Building index from {len(documents)} documents")

        # Store documents
        self.documents = documents

        # Embed documents
        embeddings = self.embed_documents(documents)
        self.embeddings = embeddings

        # Create FAISS index
        self.index = self.create_index(embeddings)

        # Save to disk if path provided
        if output_path:
            self.save_index(output_path)

        return self.index, embeddings

    def save_index(self, output_path: Path) -> None:
        """Save index and documents to disk.

        Args:
            output_path: Path to save index
        """
        import json
        from datetime import datetime
        import hashlib

        if self.index is None or self.embeddings is None:
            logger.error("No index to save")
            return

        output_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(output_path / "index.faiss"))

        # Save embeddings
        np.save(output_path / "embeddings.npy", self.embeddings)

        # Save documents metadata
        with open(output_path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        # Calculate manifest hash
        from earnings_qa.config import PACKAGE_ROOT
        manifest_path = PACKAGE_ROOT / "config" / "transcripts_manifest.json"
        manifest_hash = "unknown"
        if manifest_path.exists():
            with open(manifest_path, 'rb') as f:
                manifest_hash = hashlib.md5(f.read()).hexdigest()

        # Save index metadata
        metadata = {
            "build_time": datetime.now().isoformat(),
            "manifest_hash": manifest_hash,
            "model_version": self.embedding_model,
        }
        with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved index to {output_path}")

    def load_index(self, input_path: Path) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
        """Load index and documents from disk.

        Args:
            input_path: Path to load index from

        Returns:
            Tuple of (FAISS index, embeddings array)
        """
        import json

        if not input_path.exists():
            logger.error(f"Index path not found: {input_path}")
            return None, None

        # Load FAISS index
        self.index = faiss.read_index(str(input_path / "index.faiss"))

        # Load embeddings
        self.embeddings = np.load(input_path / "embeddings.npy")

        # Load documents metadata
        with open(input_path / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

        # Load metadata if exists
        self.metadata = {}
        metadata_path = input_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        logger.info(
            f"Loaded index from {input_path} ({len(self.documents)} documents)")

        return self.index, self.embeddings
