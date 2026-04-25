"""Structured observability for every chat turn.

Each call to `emit()` writes one JSON line to logs/telemetry.jsonl and
also emits a structured Python log record at DEBUG level so it appears in
any attached log handler without duplication.

Schema (all fields always present, missing values use None / empty string):
    request_id      uuid4 – unique per process_message call
    timestamp       ISO-8601 UTC
    question_hash   sha256 of the lowercased question (no raw PII in telemetry)
    query_type      "metric_lookup" | "qualitative" | "scope_error" | "cache_hit"
    company_id      resolved company filter (or None)
    quarter         resolved quarter filter (or None)
    retrieval_count number of docs returned by RAG
    retrieval_confidence   float (0-1) or None if not applicable
    direct_answer_used  bool
    cache_hit       bool
    latency_ms      wall-clock ms for the full process_message call
    backend_llm     class name of the LLM used
    prompt_version  from llm.prompts.PROMPT_VERSION
    guardrail_version from guardrails.validator.GUARDRAIL_VERSION
    guardrail_status  string returned by apply_guardrails
    error           error message string or None
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from earnings_qa.config import LOGS_DIR
from earnings_qa.llm.prompts import PROMPT_VERSION

logger = logging.getLogger(__name__)

TELEMETRY_LOG = LOGS_DIR / "telemetry.jsonl"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def new_request_id() -> str:
    return str(uuid.uuid4())


def emit(
    *,
    request_id: str,
    question: str,
    query_type: str,
    company_id: Optional[str],
    quarter: Optional[str],
    retrieval_count: int,
    retrieval_confidence: Optional[float],
    direct_answer_used: bool,
    cache_hit: bool,
    latency_ms: float,
    backend_llm: str,
    guardrail_version: str,
    guardrail_status: str,
    error: Optional[str] = None,
) -> None:
    """Write one structured telemetry record."""
    record = {
        "request_id": request_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "question_hash": _sha256(question.lower()),
        "query_type": query_type,
        "company_id": company_id,
        "quarter": quarter,
        "retrieval_count": retrieval_count,
        "retrieval_confidence": round(retrieval_confidence, 4) if retrieval_confidence is not None else None,
        "direct_answer_used": direct_answer_used,
        "cache_hit": cache_hit,
        "latency_ms": round(latency_ms, 1),
        "backend_llm": backend_llm,
        "prompt_version": PROMPT_VERSION,
        "guardrail_version": guardrail_version,
        "guardrail_status": guardrail_status,
        "error": error,
    }

    # Structured Python log (DEBUG)
    logger.debug("telemetry %s", json.dumps(record))

    # Persistent JSONL file
    try:
        with open(TELEMETRY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Could not write telemetry record: %s", exc)
