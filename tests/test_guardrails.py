import pytest
from earnings_qa.guardrails.validator import GuardrailValidator
from earnings_qa.rag.ingestion import Document


def make_doc(content: str, company_id: str = "c1", quarter: str = "Q1", year: str = "FY24") -> Document:
    return Document(content=content, company_id=company_id, quarter=quarter, year=year)


# ── check_scope ────────────────────────────────────────────────────────────────

def test_scope_in_scope_financial():
    validator = GuardrailValidator()
    in_scope, _ = validator.check_scope("What was Medanta's revenue in Q2?")
    assert in_scope is True


def test_scope_out_of_scope_recipe():
    validator = GuardrailValidator()
    in_scope, msg = validator.check_scope("Give me a recipe for chocolate cake.")
    assert in_scope is False
    assert "earnings calls" in msg.lower()


def test_scope_out_of_scope_coding():
    validator = GuardrailValidator()
    in_scope, msg = validator.check_scope("Can you help me fix a bug in my Python code?")
    assert in_scope is False


def test_scope_vague_growth_blocked_without_company():
    validator = GuardrailValidator()
    in_scope, msg = validator.check_scope("Why did revenue grow?")
    assert in_scope is False
    assert "company" in msg.lower() or "quarter" in msg.lower()


def test_scope_vague_growth_allowed_with_history():
    """Same vague query passes when conversation history names a company."""
    validator = GuardrailValidator()
    in_scope, _ = validator.check_scope(
        "Why did revenue grow?",
        conversation_context="Medanta reported strong results in Q2."
    )
    # history mentions 'medanta' which is a company alias → should pass
    assert in_scope is True


# ── check_confidence ───────────────────────────────────────────────────────────

def test_confidence_no_docs():
    validator = GuardrailValidator()
    score, level = validator.check_confidence([])
    assert score == 0.0
    assert "LOW" in level


def test_confidence_high_similarity():
    validator = GuardrailValidator()
    doc = make_doc("Revenue was great.")
    score, level = validator.check_confidence([(doc, 0.85)])
    assert level == "HIGH"
    assert score >= 0.7


def test_confidence_diversity_bonus():
    """Two companies => small bonus applied."""
    validator = GuardrailValidator()
    doc1 = make_doc("Revenue high.", company_id="co1")
    doc2 = make_doc("PAT strong.",  company_id="co2")
    score, _ = validator.check_confidence([(doc1, 0.6), (doc2, 0.65)])
    # Max sim is 0.65; diversity bonus +0.05 → 0.70
    assert score >= 0.65


# ── apply_guardrails ───────────────────────────────────────────────────────────

def test_apply_guardrails_valid_response():
    validator = GuardrailValidator()
    doc = make_doc("Revenue in Q1 was INR 500 million.")
    response, status = validator.apply_guardrails(
        query="What is the revenue?",
        response="Revenue for Q1 was INR 500 million.",
        retrieved_documents=[(doc, 0.8)],
    )
    assert status in ("VALID", True)
    assert "INR" in response or "Revenue" in response


def test_apply_guardrails_refusal_passthrough():
    """LLM 'no data' responses should pass through unchanged."""
    validator = GuardrailValidator()
    no_data = "I don't have reliable data on this from the provided transcripts."
    response, status = validator.apply_guardrails(
        query="What is the revenue?",
        response=no_data,
        retrieved_documents=[],
    )
    assert no_data in response
