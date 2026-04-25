import pytest
from earnings_qa.core.chat_service import ChatService
from earnings_qa.rag.ingestion import Document

def make_doc(content: str, company_id: str = "c1", quarter: str = "Q1", year: str = "FY25") -> Document:
    return Document(content=content, company_id=company_id, quarter=quarter, year=year)

def test_extract_total_income_with_noise():
    """Test regex extraction for 'total income' with mid-sentence noise (e.g., Q1 FY2025)."""
    service = ChatService()
    
    # Real-world pattern from Medanta transcript
    text = "Our total income for Q1 FY2025 stood at Rs. 8,830 million compared to INR 8,000 million last year."
    doc = make_doc(text, company_id="543654", quarter="Q1", year="FY25")
    
    answer = service._try_direct_metric_answer("What was the total income for Medanta?", [(doc, 1.0)])
    
    assert answer is not None
    assert "8,830" in answer
    assert "million" in answer
    assert "Total income" in answer

def test_extract_total_income_reverse_pattern():
    """Test regex extraction where amount comes before label."""
    service = ChatService()
    text = "INR 10,000 million was the consolidated income for the period."
    doc = make_doc(text, company_id="543654", quarter="Q1", year="FY25")
    
    answer = service._try_direct_metric_answer("What was the total income?", [(doc, 1.0)])
    
    assert answer is not None
    assert "10,000" in answer
    assert "million" in answer

def test_extract_income_alias_mapping():
    """Test that 'total income' question maps to 'income' metric spec."""
    service = ChatService()
    metrics = service._detect_requested_metrics("What is the total income for Medanta?")
    assert "income" in metrics

def test_handle_numpy_float_serialization():
    """Verify that numpy.float32 confidence doesn't crash the system (via observability)."""
    import numpy as np
    from earnings_qa.core.observability import emit
    import tempfile
    from pathlib import Path
    import json
    
    with tempfile.TemporaryDirectory() as tmp:
        tel_file = Path(tmp) / "tel.jsonl"
        import earnings_qa.core.observability as obs
        original_tel = obs.TELEMETRY_LOG
        obs.TELEMETRY_LOG = tel_file
        
        try:
            # This should NOT throw 'float32 not JSON serializable'
            emit(
                request_id="test",
                question="test",
                query_type="test",
                company_id="test",
                quarter="test",
                retrieval_count=1,
                retrieval_confidence=np.float32(0.85), # The problematic type
                direct_answer_used=False,
                cache_hit=False,
                latency_ms=10.0,
                backend_llm="test",
                guardrail_version="test",
                guardrail_status="test"
            )
            assert tel_file.exists()
            content = tel_file.read_text()
            assert "0.85" in content
        finally:
            obs.TELEMETRY_LOG = original_tel
