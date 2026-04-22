"""Configuration and constants for earnings QA system."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent  # Parent of earnings-qa
DATA_DIR = WORKSPACE_ROOT / "files"  # Points to ../files (dataset folder)
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)

# Mock Mode (for development without API calls)
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "false").lower() == "true"

# Ollama (local) configuration
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:latest")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Granular Ollama control (override USE_OLLAMA per component)
# Defaults to USE_OLLAMA if not explicitly set
USE_OLLAMA_EMBED = os.getenv("USE_OLLAMA_EMBED", str(USE_OLLAMA)).lower() == "true"
USE_OLLAMA_LLM = os.getenv("USE_OLLAMA_LLM", str(USE_OLLAMA)).lower() == "true"

# Google Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# Application Configuration
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", 10))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 5))

# LLM Parameters
LLM_TEMPERATURE = 0.3   # Lower = more factual, less hallucination
LLM_MAX_TOKENS = 512    # Faster CPU inference for llama3.2

# Retry Configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1  # seconds

# Validation
if not USE_MOCK_LLM and not USE_OLLAMA and not USE_OLLAMA_LLM and not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not set. Set it in .env, or use USE_OLLAMA=true for local inference.")
