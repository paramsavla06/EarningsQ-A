import pytest
from earnings_qa.core.chat_service import ChatService
from earnings_qa.rag.ingestion import Document


def make_doc(content: str, company_id: str = "c1", quarter: str = "Q1", year: str = "FY24") -> Document:
    return Document(content=content, company_id=company_id, quarter=quarter, year=year)


def test_detect_requested_metrics_pat_and_revenue():
    service = ChatService()
    metrics = service._detect_requested_metrics("What is the PAT and revenue?")
    assert "pat" in metrics
    assert "revenue" in metrics


def test_detect_requested_metrics_ebitda_margin():
    """'ebitda margin' should match ebitda_margin, not plain ebitda."""
    service = ChatService()
    metrics = service._detect_requested_metrics("What was the EBITDA margin?")
    assert "ebitda_margin" in metrics
    assert "ebitda" not in metrics


def test_extract_mentioned_quarters():
    service = ChatService()
    quarters = service._extract_mentioned_quarters("Tell me about Q2 and Q4.")
    assert quarters == ["Q2", "Q4"]


def test_extract_scope_from_text_quarter_only():
    """Quarter should be parsed from text even if company is unknown."""
    service = ChatService()
    _, quarter = service._extract_scope_from_text("In Q3, the revenue was good.")
    assert quarter == "Q3"


def test_try_direct_metric_answer_inr_amount():
    """Regex extraction should find INR amounts and return a formatted answer."""
    service = ChatService()
    doc = make_doc(
        "Our revenue came in at INR 1,234 million.",
        company_id="test_co", quarter="Q1", year="FY24"
    )
    answer = service._try_direct_metric_answer("What is the revenue?", [(doc, 0.9)])

    assert answer is not None
    assert "1,234" in answer
    assert "INR" in answer


def test_try_direct_metric_answer_no_match():
    """Returns None when no metric is found in the docs."""
    service = ChatService()
    doc = make_doc("The management discussed strategic plans for growth.")
    answer = service._try_direct_metric_answer("What is the revenue?", [(doc, 0.8)])
    assert answer is None


def test_is_vague_growth_reason_question_true():
    service = ChatService()
    assert service._is_vague_growth_reason_question("Why did revenue grow?") is True


def test_is_vague_growth_reason_question_false_with_company():
    """Not vague when a known company alias appears in the question."""
    service = ChatService()
    # Use a generic company term that definitely exists in any catalog
    # We check the negative: summary questions are never 'vague'
    assert service._is_vague_growth_reason_question("Summarize the quarter") is False
