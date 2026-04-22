#!/usr/bin/env python
"""Quick import check to verify all modules can be loaded."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_imports():
    """Check that all modules can be imported."""
    print("Checking imports...")

    try:
        from src.config import USE_MOCK_LLM, DATA_DIR
        print("✓ src.config")
    except Exception as e:
        print(f"✗ src.config: {e}")
        return False

    try:
        from src.llm.client import get_llm_client, LLMClient, MockLLMClient
        print("✓ src.llm.client")
    except Exception as e:
        print(f"✗ src.llm.client: {e}")
        return False

    try:
        from src.llm.prompts import SYSTEM_PROMPT, get_retrieval_prompt
        print("✓ src.llm.prompts")
    except Exception as e:
        print(f"✗ src.llm.prompts: {e}")
        return False

    try:
        from src.rag.ingestion import TranscriptIngestionPipeline, Document
        print("✓ src.rag.ingestion")
    except Exception as e:
        print(f"✗ src.rag.ingestion: {e}")
        return False

    try:
        from src.rag.embeddings import EmbeddingPipeline
        print("✓ src.rag.embeddings")
    except Exception as e:
        print(f"✗ src.rag.embeddings: {e}")
        return False

    try:
        from src.rag.retriever import Retriever
        print("✓ src.rag.retriever")
    except Exception as e:
        print(f"✗ src.rag.retriever: {e}")
        return False

    try:
        from src.guardrails.validator import GuardrailValidator
        print("✓ src.guardrails.validator")
    except Exception as e:
        print(f"✗ src.guardrails.validator: {e}")
        return False

    try:
        from src.cli.interface import EarningsQACLI, main
        print("✓ src.cli.interface")
    except Exception as e:
        print(f"✗ src.cli.interface: {e}")
        return False

    print("\n✅ All imports successful!")
    print(f"Mock mode: {USE_MOCK_LLM}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Data directory exists: {DATA_DIR.exists()}")
    return True


if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)
