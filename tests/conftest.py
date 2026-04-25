import os
import pytest
from pathlib import Path

# Force mock mode and test environment variables
os.environ["USE_MOCK_LLM"] = "true"
os.environ["EARNINGS_QA_WORKSPACE"] = str(Path(__file__).parent / "test_workspace")

# Create test workspace directory
@pytest.fixture(autouse=True, scope="session")
def setup_test_env():
    test_workspace = Path(os.environ["EARNINGS_QA_WORKSPACE"])
    test_workspace.mkdir(parents=True, exist_ok=True)
    yield
    # Cleanup could go here if needed
