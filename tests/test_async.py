import pytest
import asyncio
from earnings_qa.llm.client import MockLLMClient
from earnings_qa.rag.ingestion import Document
from earnings_qa.rag.retriever import Retriever
from earnings_qa.rag.embeddings import EmbeddingPipeline
import tempfile
from pathlib import Path


def make_doc(content: str, company_id: str = "c1", quarter: str = "Q1", year: str = "FY24") -> Document:
    return Document(content=content, company_id=company_id, quarter=quarter, year=year)


def _build_retriever(docs):
    """Build a Retriever backed by a real in-memory FAISS index (mock embeddings)."""
    pipeline = EmbeddingPipeline()
    with tempfile.TemporaryDirectory() as tmp:
        pipeline.build_index(docs, output_path=Path(tmp))
    return Retriever(pipeline)


# ── LLM async ─────────────────────────────────────────────────────────────────

def test_mock_llm_answer_question_sync():
    client = MockLLMClient()
    response = client.answer_question("You are helpful.", "What is the revenue?")
    assert "revenue" in response.lower() or "MOCK" in response


def test_mock_llm_answer_question_async():
    client = MockLLMClient()
    response = asyncio.run(
        client.answer_question_async("You are helpful.", "What is the revenue?")
    )
    assert "revenue" in response.lower() or "MOCK" in response


# ── Retriever async ────────────────────────────────────────────────────────────

def test_retriever_retrieve_sync():
    docs = [make_doc("Revenue in Q1 was strong.", company_id="c1", quarter="Q1")]
    retriever = _build_retriever(docs)
    results = retriever.retrieve("revenue", top_k=1)
    assert isinstance(results, list)


def test_retriever_retrieve_async():
    docs = [make_doc("PAT grew in Q2.", company_id="c2", quarter="Q2")]
    retriever = _build_retriever(docs)
    results = asyncio.run(retriever.retrieve_async("PAT", top_k=1))
    assert isinstance(results, list)


def test_retriever_retrieve_by_filters_async():
    docs = [
        make_doc("Revenue in Q1.", company_id="c1", quarter="Q1"),
        make_doc("PAT in Q2.",     company_id="c1", quarter="Q2"),
    ]
    retriever = _build_retriever(docs)
    filtered = asyncio.run(
        retriever.retrieve_by_filters_async(company_ids=["c1"], quarters=["Q1"])
    )
    assert all(d.quarter == "Q1" for d in filtered)
    assert len(filtered) == 1
