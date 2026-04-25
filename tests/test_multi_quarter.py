import pytest
from earnings_qa.core.chat_service import ChatService
from earnings_qa.rag.ingestion import Document

def make_doc(content: str, company_id: str, quarter: str, year: str) -> Document:
    return Document(content=content, company_id=company_id, quarter=quarter, year=year)

def test_extract_all_medanta_quarters():
    """Verify that all 4 Medanta quarters are found when querying for Medanta generally."""
    service = ChatService()
    
    docs = [
        (make_doc("Our total income for Q1 FY2025 stood at Rs. 8,830 million", "543654", "Q1", "FY25"), 1.0),
        (make_doc("total income was INR 9,748 million in Q2", "543654", "Q2", "FY25"), 1.0),
        (make_doc("Total income was INR 9,595 million for Q3", "543654", "Q3", "FY24"), 1.0),
        (make_doc("During the year, Medanta delivered total consolidated income of INR 37,714 million", "543654", "Q4", "FY26"), 1.0),
    ]
    
    answer = service._try_direct_metric_answer("total income for Medanta", docs)
    assert answer is not None
    assert "Q1 FY25: INR8,830 million" in answer
    assert "Q2 FY25: INR9,748 million" in answer
    assert "Q3 FY24: INR9,595 million" in answer
    assert "Q4 FY26: INR37,714 million" in answer

def test_extract_income_with_inr_no_space():
    """Verify that INR9,748 (no space) is matched."""
    service = ChatService()
    doc = make_doc("Total income was INR9,748 million", "543654", "Q1", "FY25")
    answer = service._try_direct_metric_answer("income", [(doc, 1.0)])
    assert answer is not None
    assert "9,748" in answer
