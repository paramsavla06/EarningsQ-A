"""Embedding pipeline: Embed documents and store in FAISS."""

import logging
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

try:
    import google.genai as genai
except ImportError:
    try:
        from google import genai
    except ImportError:
        import genai

import faiss

from src.config import GEMINI_API_KEY, EMBEDDING_MODEL, USE_MOCK_LLM
from src.rag.ingestion import Document

logger = logging.getLogger(__name__)


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

        # Initialize Gemini client if not in mock mode
        if not USE_MOCK_LLM:
            genai.configure(api_key=api_key)
            self.client = genai.Client(api_key=api_key)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text using Gemini embedding model.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (numpy array)
        """
        if USE_MOCK_LLM:
            # Return a random embedding for mock mode
            return np.random.randn(768).astype(np.float32)

        try:
            # Use Gemini embedding API
            result = self.client.models.embed_content(
                model=self.embedding_model,
                content=text,
            )

            embedding = np.array(result['embedding']).astype(np.float32)
            return embedding

        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            # Fallback to random embedding
            return np.random.randn(768).astype(np.float32)

    def embed_documents(self, documents: List[Document], batch_size: int = 10) -> np.ndarray:
        """Embed a batch of documents.

        Args:
            documents: List of documents to embed
            batch_size: Number of documents to embed at once

        Returns:
            Embedding matrix (n_documents x embedding_dim)
        """
        embeddings = []

        for i, doc in enumerate(documents):
            if i % batch_size == 0:
                logger.info(f"Embedding document {i+1}/{len(documents)}")

            # Embed document content (first 2000 chars to avoid token limits)
            embedding = self.embed_text(doc.content[:2000])
            embeddings.append(embedding)

        embeddings_array = np.array(embeddings).astype(np.float32)
        logger.info(
            f"Created {embeddings_array.shape[0]} embeddings of dimension {embeddings_array.shape[1]}")

        return embeddings_array

    def create_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """Create FAISS index from embeddings.

        Args:
            embeddings: Embedding matrix (n x d)

        Returns:
            FAISS index
        """
        embedding_dim = embeddings.shape[1]

        # Use L2 distance (Euclidean distance)
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

        logger.info(f"Saved index to {output_path}")

    def load_index(self, input_path: Path) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
        """Load index and documents from disk.

        Args:
            input_path: Path to load index from

        Returns:
            Tuple of (FAISS index, embeddings array)
        """
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

        logger.info(
            f"Loaded index from {input_path} ({len(self.documents)} documents)")

        return self.index, self.embeddings
