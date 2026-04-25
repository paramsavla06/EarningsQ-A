"""Tests for Phase 8/9/10:
- Observability telemetry records
- Prompt and guardrail versioning
- Cache invalidation
- Conversation history handling
- End-to-end ChatService (mock LLM, no network)
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from earnings_qa.core.chat_service import ChatService, ChatResponse, ConversationHistory
from earnings_qa.core.cache import CacheManager
from earnings_qa.core.observability import emit, new_request_id
from earnings_qa.llm.prompts import PROMPT_VERSION
from earnings_qa.guardrails.validator import GUARDRAIL_VERSION
from earnings_qa.rag.ingestion import Document
from earnings_qa.rag.embeddings import EmbeddingPipeline
from earnings_qa.rag.retriever import Retriever


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_doc(content: str, company_id: str = "c1", quarter: str = "Q1", year: str = "FY24") -> Document:
    return Document(content=content, company_id=company_id, quarter=quarter, year=year)


def build_retriever(docs):
    pipeline = EmbeddingPipeline()
    with tempfile.TemporaryDirectory() as tmp:
        pipeline.build_index(docs, output_path=Path(tmp))
    return Retriever(pipeline)


# ── Phase 8: Observability ────────────────────────────────────────────────────

def test_new_request_id_is_unique():
    ids = {new_request_id() for _ in range(100)}
    assert len(ids) == 100


def test_emit_writes_jsonl(tmp_path, monkeypatch):
    """emit() should append a valid JSON line to the telemetry file."""
    import earnings_qa.core.observability as obs_mod
    monkeypatch.setattr(obs_mod, "TELEMETRY_LOG", tmp_path / "telemetry.jsonl")

    emit(
        request_id="test-rid",
        question="What is the revenue?",
        query_type="metric_lookup",
        company_id="c1",
        quarter="Q1",
        retrieval_count=3,
        retrieval_confidence=0.82,
        direct_answer_used=True,
        cache_hit=False,
        latency_ms=123.4,
        backend_llm="MockLLMClient",
        guardrail_version=GUARDRAIL_VERSION,
        guardrail_status="VALID",
    )

    lines = (tmp_path / "telemetry.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["request_id"] == "test-rid"
    assert record["query_type"] == "metric_lookup"
    assert record["retrieval_count"] == 3
    assert record["retrieval_confidence"] == 0.82
    assert record["direct_answer_used"] is True
    assert record["cache_hit"] is False
    assert record["latency_ms"] == 123.4
    assert record["prompt_version"] == PROMPT_VERSION
    assert record["guardrail_version"] == GUARDRAIL_VERSION
    assert "question_hash" in record          # PII not stored raw
    assert "question" not in record


def test_chat_service_response_has_request_id():
    """Every ChatResponse must carry a non-empty request_id."""
    svc = ChatService()
    resp = svc.process_message("What is the revenue for Q1?")
    assert isinstance(resp.request_id, str) and len(resp.request_id) > 0


def test_chat_service_scope_error_has_request_id():
    svc = ChatService()
    resp = svc.process_message("Give me a recipe for pasta.")
    assert resp.error_msg is not None
    assert isinstance(resp.request_id, str)


def test_telemetry_emitted_on_full_turn(tmp_path, monkeypatch):
    """A successful turn must produce exactly one telemetry record."""
    import earnings_qa.core.observability as obs_mod
    tel_file = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(obs_mod, "TELEMETRY_LOG", tel_file)

    svc = ChatService()
    svc.process_message("What is the EBITDA?")

    lines = tel_file.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert "latency_ms" in record
    assert record["latency_ms"] > 0


# ── Phase 9: Versioning ────────────────────────────────────────────────────────

def test_prompt_version_is_set():
    assert isinstance(PROMPT_VERSION, str) and PROMPT_VERSION.startswith("v")


def test_guardrail_version_is_set():
    assert isinstance(GUARDRAIL_VERSION, str) and GUARDRAIL_VERSION.startswith("v")


def test_telemetry_records_prompt_and_guardrail_version(tmp_path, monkeypatch):
    import earnings_qa.core.observability as obs_mod
    tel_file = tmp_path / "tel.jsonl"
    monkeypatch.setattr(obs_mod, "TELEMETRY_LOG", tel_file)

    svc = ChatService()
    svc.process_message("What is the revenue?")

    record = json.loads(tel_file.read_text().strip())
    assert record["prompt_version"] == PROMPT_VERSION
    assert record["guardrail_version"] == GUARDRAIL_VERSION


def test_cache_key_invalidated_on_prompt_version_change(tmp_path):
    """Changing PROMPT_VERSION must produce a cache miss."""
    mgr = CacheManager(cache_dir=tmp_path / "cache")

    mgr.set_answer("q", "co", "q1", "v_idx", "v1.0.0", "h1", {"direct_answer": "A", "llm_answer": "B"})
    assert mgr.get_answer("q", "co", "q1", "v_idx", "v1.0.0", "h1") is not None
    # Different prompt version → miss
    assert mgr.get_answer("q", "co", "q1", "v_idx", "v2.0.0", "h1") is None


def test_cache_key_invalidated_on_index_version_change(tmp_path):
    mgr = CacheManager(cache_dir=tmp_path / "cache")
    mgr.set_answer("q", "co", "q1", "hash_A", "v1.0.0", "h1", {"llm_answer": "old"})
    # New index hash → miss
    assert mgr.get_answer("q", "co", "q1", "hash_B", "v1.0.0", "h1") is None


def test_cache_hit_served_without_llm_call(tmp_path):
    """A primed cache entry must be served without touching the LLM."""
    mgr = CacheManager(cache_dir=tmp_path / "cache")

    # Prime the answer
    mgr.set_answer(
        "what is the revenue?", "None", "None", "unknown", PROMPT_VERSION, mgr._hash(""),
        {"direct_answer": None, "llm_answer": "Cached LLM answer"}
    )

    call_count = {"n": 0}

    class CountingLLM:
        def answer_question(self, system_prompt, user_message, max_tokens=512):
            call_count["n"] += 1
            return "SHOULD NOT BE CALLED"
        async def answer_question_async(self, *a, **kw):
            return self.answer_question(*a, **kw)
        def get_token_count_estimate(self, text):
            return 0

    import earnings_qa.core.chat_service as cs_mod
    original_cache = cs_mod.cache

    try:
        cs_mod.cache = mgr
        svc = ChatService()
        svc.llm = CountingLLM()
        resp = svc.process_message("What is the revenue?")
        assert call_count["n"] == 0, "LLM was called despite cache hit"
        assert resp.llm_answer == "Cached LLM answer"
    finally:
        cs_mod.cache = original_cache


# ── Phase 10: History handling ────────────────────────────────────────────────

def test_conversation_history_adds_messages():
    hist = ConversationHistory(max_history=5)
    hist.add("user", "What is revenue?")
    hist.add("assistant", "Revenue is INR 1,000 million.")
    assert len(hist.history) == 2


def test_conversation_history_truncates_at_max():
    hist = ConversationHistory(max_history=2)
    for i in range(10):
        hist.add("user", f"Q{i}")
        hist.add("assistant", f"A{i}")
    # max_history=2 → keep last 2*2=4 messages
    assert len(hist.history) == 4


def test_conversation_history_clear():
    hist = ConversationHistory()
    hist.add("user", "Hello")
    hist.clear()
    assert hist.history == []


def test_history_context_propagates_company():
    """A company mentioned in history should resolve in next turn."""
    svc = ChatService()
    # Seed history with a company context
    svc.conversation.add("user", "Tell me about Medanta Q1")
    svc.conversation.add("assistant", "Medanta Q1 revenue was INR 500 million.")

    company_id, quarter = svc._resolve_scope_from_history()
    # Should pick up 'Q1' from history
    assert quarter == "Q1"


def test_retrieval_filter_by_quarter():
    """retrieve_by_filters must only return docs matching the requested quarter."""
    docs = [
        make_doc("Revenue Q1.", company_id="c1", quarter="Q1"),
        make_doc("Revenue Q2.", company_id="c1", quarter="Q2"),
        make_doc("Revenue Q3.", company_id="c1", quarter="Q3"),
    ]
    retriever = build_retriever(docs)
    filtered = retriever.retrieve_by_filters(quarters=["Q2"])
    assert all(d.quarter == "Q2" for d in filtered)
    assert len(filtered) == 1


def test_retrieval_filter_by_company_and_quarter():
    docs = [
        make_doc("Revenue.", company_id="co_A", quarter="Q1"),
        make_doc("PAT.",     company_id="co_B", quarter="Q1"),
        make_doc("EBITDA.",  company_id="co_A", quarter="Q2"),
    ]
    retriever = build_retriever(docs)
    filtered = retriever.retrieve_by_filters(company_ids=["co_A"], quarters=["Q1"])
    assert len(filtered) == 1
    assert filtered[0].company_id == "co_A"
    assert filtered[0].quarter == "Q1"
