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
        from earnings_qa.config import USE_MOCK_LLM, DATA_DIR
        print("✓ src.config")
    except Exception as e:
        print(f"✗ src.config: {e}")
        return False

    try:
        from earnings_qa.llm import get_llm_client, LLMClient, MockLLMClient
        print("✓ src.llm")
    except Exception as e:
        print(f"✗ src.llm: {e}")
        return False

    try:
        from earnings_qa.llm.prompts import SYSTEM_PROMPT, get_retrieval_prompt
        print("✓ src.llm.prompts")
    except Exception as e:
        print(f"✗ src.llm.prompts: {e}")
        return False

    try:
        from earnings_qa.rag import TranscriptIngestionPipeline, Document, EmbeddingPipeline, Retriever
        print("✓ src.rag")
    except Exception as e:
        print(f"✗ src.rag: {e}")
        return False

    try:
        from earnings_qa.guardrails import GuardrailValidator
        print("✓ src.guardrails")
    except Exception as e:
        print(f"✗ src.guardrails: {e}")
        return False

    try:
        from earnings_qa.cli import EarningsQACLI, main
        print("✓ src.cli")
    except Exception as e:
        print(f"✗ src.cli: {e}")
        return False

    print("\n✅ All imports successful!")
    print(f"Mock mode: {USE_MOCK_LLM}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Data directory exists: {DATA_DIR.exists()}")
    return True


if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)
