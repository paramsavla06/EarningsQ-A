import logging
import json
import re
from datetime import datetime
from typing import Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass

from src.config import LOGS_DIR, MAX_CONVERSATION_HISTORY
from src.llm import get_llm_client
from src.llm.prompts import SYSTEM_PROMPT, get_retrieval_prompt
from src.rag import Retriever
from src.guardrails import GuardrailValidator

logger = logging.getLogger(__name__)


class ConversationHistory:
    """Manage multi-turn conversation context with token-aware truncation."""

    def __init__(self, max_history: int = MAX_CONVERSATION_HISTORY):
        """Initialize conversation history.

        Args:
            max_history: Maximum number of exchanges to keep
        """
        self.history: List[dict] = []
        self.max_history = max_history

    def add(self, role: str, content: str) -> None:
        """Add message to history.

        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        self.history.append({"role": role, "content": content})

        # Truncate if exceeding max history
        if len(self.history) > self.max_history * 2:  # *2 because each exchange = 2 messages
            self.history = self.history[-(self.max_history * 2):]

    def get_context(self) -> str:
        """Get conversation history as formatted context.

        Returns:
            Formatted conversation history
        """
        if not self.history:
            return ""

        lines = ["Recent conversation context:"]
        for msg in self.history[-6:]:  # Last 3 exchanges
            role = "You" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content'][:200]}...")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear conversation history."""
        self.history = []


def log_query(question: str, answer: str, company: Optional[str], quarter: Optional[str]) -> None:
    """Log query and response to audit file.

    Args:
        question: User question
        answer: LLM answer
        company: Optional company filter
        quarter: Optional quarter filter
    """
    audit_log = LOGS_DIR / "audit.jsonl"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer[:500],  # First 500 chars
        "company_filter": company,
        "quarter_filter": quarter,
    }

    with open(audit_log, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


@dataclass
class ChatResponse:
    """Structured response from the chat service."""
    direct_answer: Optional[str] = None
    llm_answer: Optional[str] = None
    error_msg: Optional[str] = None


class ChatService:
    """Core service orchestrating retrieval, extraction, and generation."""

    def __init__(self, retriever: Optional[Retriever] = None):
        """Initialize chat service."""
        self.llm = get_llm_client()
        self.conversation = ConversationHistory()
        self.retriever = retriever
        self.validator = GuardrailValidator()

    @property
    def indexed(self) -> bool:
        return self.retriever is not None

    @staticmethod
    def _normalize_amount(amount: str, unit: str = "") -> str:
        """Format extracted numeric values in a readable INR-style string."""
        amount = amount.strip()
        unit = unit.strip()
        if unit:
            return f"INR{amount} {unit}"
        return f"INR{amount}"

    @staticmethod
    def _parse_amount(amount: str) -> float:
        """Convert a captured numeric string into a float for ranking."""
        return float(amount.replace(",", ""))

    @staticmethod
    def _metric_specs() -> dict:
        """Canonical finance metrics supported by the direct-answer path."""
        return {
            "revenue": {
                "aliases": ["revenue", "revenues", "turnover", "sales", "top line", "top-line"],
                "kind": "amount",
                "display": "Revenue",
            },
            "income": {
                "aliases": ["income", "total income", "consolidated income"],
                "kind": "amount",
                "display": "Total income",
            },
            "pat": {
                "aliases": ["pat", "profit after tax", "net profit"],
                "kind": "amount",
                "display": "PAT",
            },
            "ebitda": {
                "aliases": ["ebitda", "operating profit", "operating income"],
                "kind": "amount",
                "display": "EBITDA",
            },
            "ebitda_margin": {
                "aliases": ["ebitda margin", "margin"],
                "kind": "percent",
                "display": "EBITDA margin",
            },
            "gross_margin": {
                "aliases": ["gross margin"],
                "kind": "percent",
                "display": "Gross margin",
            },
            "operating_margin": {
                "aliases": ["operating margin"],
                "kind": "percent",
                "display": "Operating margin",
            },
            "cash_flow": {
                "aliases": ["cash flow", "operating cash flow", "free cash flow"],
                "kind": "amount",
                "display": "Cash flow",
            },
            "capex": {
                "aliases": ["capex", "capital expenditure", "capital spending"],
                "kind": "amount",
                "display": "Capex",
            },
            "pbt": {
                "aliases": ["pbt", "profit before tax"],
                "kind": "amount",
                "display": "PBT",
            },
            "arpob": {
                "aliases": ["arpob", "average revenue per occupied bed"],
                "kind": "amount",
                "display": "ARPOB",
            },
        }

    @classmethod
    def _detect_requested_metrics(cls, question: str) -> List[str]:
        """Return finance metrics explicitly asked in the question."""
        question_lower = question.lower()
        requested = []
        for metric_key, spec in cls._metric_specs().items():
            if metric_key == "ebitda" and "margin" in question_lower:
                continue
            if any(alias in question_lower for alias in spec["aliases"]):
                requested.append(metric_key)
        return requested

    @staticmethod
    def _extract_mentioned_quarters(question: str) -> List[str]:
        """Return quarters explicitly mentioned in the question, in order of appearance."""
        quarters = []
        for match in re.finditer(r"\bQ([1-4])\b", question, flags=re.IGNORECASE):
            quarter = f"Q{match.group(1)}"
            if quarter not in quarters:
                quarters.append(quarter)
        return quarters

    @staticmethod
    def _extract_scope_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract the most obvious company id and quarter from a text snippet."""
        from src.config import get_company_mapping
        text_lower = text.lower()
        company_mapping = get_company_mapping()

        company_id = None
        for cid, keywords in company_mapping.items():
            if any(keyword in text_lower for keyword in keywords):
                company_id = cid
                break

        quarter = None
        match = re.search(r"\bQ([1-4])\b", text, flags=re.IGNORECASE)
        if match:
            quarter = f"Q{match.group(1)}"

        return company_id, quarter

    def _resolve_scope_from_history(self) -> Tuple[Optional[str], Optional[str]]:
        """Resolve company and quarter from the recent conversation, if any."""
        company_id = None
        quarter = None

        for message in reversed(self.conversation.history):
            history_company, history_quarter = self._extract_scope_from_text(
                message["content"])
            if not company_id and history_company:
                company_id = history_company
            if not quarter and history_quarter:
                quarter = history_quarter
            if company_id and quarter:
                break

        return company_id, quarter

    def _is_vague_growth_reason_question(self, question: str, conversation_context: str = "") -> bool:
        """Detect growth/reason questions that do not name a company."""
        q = question.lower()
        growth_terms = ["why", "reason", "because", "grow", "growth",
                        "grew", "increase", "increased", "decline", "declined"]
        summary_terms = ["summarize", "summary", "overview", "recap"]
        from src.config import get_all_company_aliases
        company_terms = get_all_company_aliases()

        if any(term in q for term in summary_terms):
            return False

        if not any(term in q for term in growth_terms):
            return False

        if any(term in q for term in company_terms):
            return False

        if conversation_context and any(term in conversation_context.lower() for term in company_terms):
            return False

        return True

    def _extract_metric_candidates(self, retrieved_docs, metric: str):
        """Return candidate metric values from retrieved transcript chunks."""
        spec = self._metric_specs().get(metric)
        if not spec:
            return []

        label_regex = "|".join(
            sorted((re.escape(alias)
                   for alias in spec["aliases"]), key=len, reverse=True)
        )

        def _looks_like_money(amount_text: str, unit_text: str) -> bool:
            amount_clean = amount_text.replace(",", "")
            try:
                numeric = float(amount_clean)
            except ValueError:
                return False
            if unit_text:
                return True
            return numeric >= 100

        def _score_match(text: str, match_start: int, match_end: int, metric_key: str, amount_text: str, unit_text: str) -> int:
            snippet_start = max(0, match_start - 120)
            snippet_end = min(len(text), match_end + 120)
            snippet = text[snippet_start:snippet_end].lower()
            score = 0

            if metric_key == "pat":
                if "pat stood" in snippet or "profit after tax" in snippet:
                    score += 8
                if "has been at" in snippet:
                    score -= 4
                if "rupee terms" in snippet or "in rupee terms" in snippet:
                    score -= 2
                if unit_text:
                    score += 2
            elif metric_key == "revenue":
                if "revenue" in snippet or "revenues" in snippet or "income" in snippet:
                    score += 5
                if "has been at" in snippet:
                    score -= 2
                if unit_text:
                    score += 2
            else:
                if "%" in snippet:
                    score += 5

            if "reported" in snippet or "stood at" in snippet or "came in at" in snippet or "has been at" in snippet:
                score += 2
            if "usd" in snippet and not ("rupee terms" in snippet or "inr" in snippet):
                score -= 2

            try:
                numeric = float(amount_text.replace(",", ""))
            except ValueError:
                numeric = 0.0
            if metric_key == "revenue" and numeric >= 1000:
                score += 3
            if metric_key == "pat" and numeric < 100:
                score += 2

            return score

        if spec["kind"] == "amount":
            patterns = [
                rf"(?:{label_regex})\D{{0,120}}(?:was|is|at|of|stood at|reported|came in at|amounted to|totaled|totalled)?\D{{0,40}}(?:inr|rs\.?|₹|nr)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(million|crore|crores|bn|billion)?",
                rf"(?:inr|rs\.?|₹|nr)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(million|crore|crores|bn|billion)\D{{0,80}}(?:{label_regex})",
            ]
        else:
            patterns = [
                rf"(?:{label_regex})\D{{0,80}}(?:was|is|at|stood at|came in at|improved to|declined to|increased to)?\D{{0,20}}([0-9][0-9,]*(?:\.[0-9]+)?)\s*%",
                rf"([0-9][0-9,]*(?:\.[0-9]+)?)\s*%\D{{0,80}}(?:{label_regex})",
            ]

        candidates = []
        for doc, similarity in retrieved_docs:
            text = doc.content
            for pattern in patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                    amount = match.group(1)
                    unit = match.group(
                        2) if match.lastindex and match.lastindex >= 2 else ""
                    if spec["kind"] == "amount":
                        if not _looks_like_money(amount, unit):
                            continue
                        numeric_value = self._parse_amount(amount)
                        if unit == "" and numeric_value < 100:
                            continue
                        value_text = self._normalize_amount(amount, unit)
                    else:
                        numeric_value = self._parse_amount(amount)
                        value_text = f"{amount.strip()}%"

                    context_score = _score_match(
                        text=text,
                        match_start=match.start(),
                        match_end=match.end(),
                        metric_key=metric,
                        amount_text=amount,
                        unit_text=unit,
                    )

                    candidates.append(
                        (context_score, numeric_value, value_text, doc, similarity, match.start()))

        return candidates

    def _get_exact_lookup_documents(
        self,
        question: str,
        retrieved_docs,
        company_filter: Optional[str] = None,
        quarter_filter: Optional[str] = None,
    ):
        if not self.retriever:
            return retrieved_docs, [], []

        company_ids = [company_filter] if company_filter else self.retriever._detect_company_intents(
            question)
        quarters = [quarter_filter] if quarter_filter else self.retriever._detect_quarter_intents(
            question)

        if company_ids or quarters:
            docs = self.retriever.retrieve_by_filters(
                company_ids=company_ids,
                quarters=quarters,
            )
            return [(doc, 1.0) for doc in docs], company_ids, quarters

        return retrieved_docs, [], []

    def _try_direct_metric_answer(
        self,
        question: str,
        retrieved_docs,
        company_filter: Optional[str] = None,
        quarter_filter: Optional[str] = None,
    ) -> Optional[str]:
        requested_metrics = self._detect_requested_metrics(question)

        if not requested_metrics or not retrieved_docs:
            return None

        lookup_docs, detected_company_ids, detected_quarters = self._get_exact_lookup_documents(
            question,
            retrieved_docs,
            company_filter=company_filter,
            quarter_filter=quarter_filter,
        )
        if lookup_docs:
            retrieved_docs = lookup_docs

        lines = []

        for metric_key in requested_metrics:
            candidates = self._extract_metric_candidates(
                retrieved_docs, metric_key)
            if not candidates:
                continue

            metric_display = self._metric_specs()[metric_key]["display"]

            from src.config import get_company_names
            company_mapping = get_company_names()

            best_per_entity = {}
            for cand in candidates:
                entity_key = (cand[3].company_id,
                              f"{cand[3].quarter} {cand[3].year}")
                if entity_key not in best_per_entity or cand[0] > best_per_entity[entity_key][0]:
                    best_per_entity[entity_key] = cand

            sorted_results = sorted(best_per_entity.values(), key=lambda x: (
                x[3].company_id, x[3].year, x[3].quarter))

            current_company = None
            for cand in sorted_results:
                metric_score, metric_value, metric_text, metric_doc, metric_similarity, metric_position = cand
                company_name = company_mapping.get(
                    metric_doc.company_id, f"Company {metric_doc.company_id}")

                if company_name != current_company:
                    lines.append(f"\n{metric_display} for {company_name}:")
                    current_company = company_name

                lines.append(
                    f"• {metric_doc.quarter} {metric_doc.year}: {metric_text} [{metric_doc.company_id} {metric_doc.quarter}{metric_doc.year} - Match: {metric_similarity:.2%}]"
                )

        if not lines:
            return None
        return "\n".join(lines)

    @staticmethod
    def _is_exact_metric_question(question: str) -> bool:
        return len(ChatService._detect_requested_metrics(question)) > 0

    @staticmethod
    def _filter_conflict_message(company_filter: Optional[str], quarter_filter: Optional[str]) -> str:
        scope_parts = []
        if company_filter:
            scope_parts.append(f"company {company_filter}")
        if quarter_filter:
            scope_parts.append(f"quarter {quarter_filter}")

        scope_text = " and ".join(
            scope_parts) if scope_parts else "the current transcript scope"
        return (
            "Direct Answer:\n"
            f"I cannot answer that from the current filtered session because it is restricted to {scope_text}. "
            "Please rerun `python main.py` without those filters, or start a new session with the company and quarter you want to query."
        )

    @staticmethod
    def _is_outside_active_filters(
        question: str,
        company_filter: Optional[str],
        quarter_filter: Optional[str],
        retriever: Optional[Retriever],
    ) -> bool:
        if not retriever:
            return False

        requested_companies = retriever._detect_company_intents(question)
        requested_quarters = retriever._detect_quarter_intents(question)

        if company_filter and requested_companies and company_filter not in requested_companies:
            return True
        if quarter_filter and requested_quarters and quarter_filter not in requested_quarters:
            return True

        return False

    def process_message(
        self,
        question: str,
        company_filter: Optional[str] = None,
        quarter_filter: Optional[str] = None,
    ) -> ChatResponse:
        """Process a user question and generate a response."""
        conversation_context = self.conversation.get_context()
        history_company_filter, history_quarter_filter = self._resolve_scope_from_history()

        resolved_company_filter = company_filter or history_company_filter
        resolved_quarter_filter = quarter_filter or history_quarter_filter

        # Apply scope guardrail
        in_scope, scope_msg = self.validator.check_scope(
            question,
            conversation_context=conversation_context,
        )
        if not in_scope:
            return ChatResponse(error_msg=scope_msg)

        if self._is_vague_growth_reason_question(question, conversation_context=conversation_context):
            msg = (
                "Please specify the company and quarter for growth-related questions. "
                "For example: 'Why did Medanta revenue grow in Q2 FY2025?' or 'What drove Polycab's revenue in Q3?'"
            )
            self.conversation.add("user", question)
            self.conversation.add("assistant", msg)
            log_query(question, msg, company_filter, quarter_filter)
            return ChatResponse(error_msg=msg)

        # Retrieve relevant context from RAG
        context = ""
        retrieved_docs = []

        if self.indexed and self.retriever:
            if self._is_outside_active_filters(
                question=question,
                company_filter=company_filter,
                quarter_filter=quarter_filter,
                retriever=self.retriever,
            ):
                answer = self._filter_conflict_message(
                    company_filter, quarter_filter)
                answer, guardrail_status = self.validator.apply_guardrails(
                    query=question,
                    response=answer,
                    retrieved_documents=[],
                    conversation_context=conversation_context,
                )
                self.conversation.add("user", question)
                self.conversation.add("assistant", answer)
                log_query(question, answer,
                          company_filter, quarter_filter)
                return ChatResponse(error_msg=answer)

            retrieved_docs = self.retriever.retrieve(
                question,
                company_ids=[resolved_company_filter] if resolved_company_filter else None,
                quarters=[resolved_quarter_filter] if resolved_quarter_filter else None,
            )
            if retrieved_docs:
                context = self.retriever.format_context(retrieved_docs)

        direct_answer = self._try_direct_metric_answer(
            question,
            retrieved_docs,
            company_filter=resolved_company_filter,
            quarter_filter=resolved_quarter_filter,
        )
        
        final_answer = ""
        if direct_answer:
            final_answer += direct_answer + "\n\n"

        # Call the LLM to cover anything not answered by Regex.
        user_message = get_retrieval_prompt(
            question=question,
            context=context or "No relevant context found in transcripts.",
            company_filter=resolved_company_filter or "",
            quarter_filter=resolved_quarter_filter or "",
            conversation_context=conversation_context,
            already_extracted=direct_answer or "",
        )

        llm_answer = None
        try:
            raw_llm_answer = self.llm.answer_question(
                system_prompt=SYSTEM_PROMPT,
                user_message=user_message,
            )

            # Apply guardrails
            validated_answer, guardrail_status = self.validator.apply_guardrails(
                query=question,
                response=raw_llm_answer,
                retrieved_documents=retrieved_docs,
                conversation_context=conversation_context,
            )

            llm_answer = validated_answer
            final_answer += validated_answer

            # Log interaction
            self.conversation.add("user", question)
            self.conversation.add("assistant", final_answer.strip())
            log_query(question, final_answer.strip(),
                      resolved_company_filter, resolved_quarter_filter)

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ChatResponse(direct_answer=direct_answer, error_msg=f"Error calling LLM: {e}")

        return ChatResponse(direct_answer=direct_answer, llm_answer=llm_answer)
