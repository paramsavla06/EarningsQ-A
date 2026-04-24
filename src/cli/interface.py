"""Click-based CLI interface for earnings QA chatbot."""

import logging
import json
import re
from datetime import datetime
from typing import Optional, List
from pathlib import Path

import click

from src.config import LOGS_DIR, MAX_CONVERSATION_HISTORY, USE_MOCK_LLM, DATA_DIR
from src.llm import get_llm_client
from src.llm.prompts import SYSTEM_PROMPT, get_retrieval_prompt
from src.rag import TranscriptIngestionPipeline, EmbeddingPipeline, Retriever
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


def setup_logging() -> None:
    """Setup logging configuration."""
    import sys

    log_file = LOGS_DIR / \
        f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # File handler: full details with timestamp
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))

    # Console handler: simple format (no timestamp)
    # type: ignore[attr-defined]
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "%(message)s"
    ))

    # Clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
    )


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


class EarningsQACLI:
    """Main CLI interface for earnings QA chatbot."""

    def __init__(self):
        """Initialize CLI."""
        self.llm = get_llm_client()
        self.conversation = ConversationHistory()
        self.indexed = False
        self.retriever: Optional[Retriever] = None
        self.validator = GuardrailValidator()
        self.index_path = Path(".") / "rag_index"  # Local index storage

        # Try to load existing index
        self._load_index()

    def _load_index(self) -> None:
        """Load existing RAG index if available."""
        if self.index_path.exists():
            try:
                embedding_pipeline = EmbeddingPipeline()
                embedding_pipeline.load_index(self.index_path)
                self.retriever = Retriever(embedding_pipeline)
                self.indexed = True
                logger.info("✓ Loaded existing RAG index")
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}")
                self.indexed = False

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

    @classmethod
    def _has_mixed_quarter_request(cls, question: str) -> bool:
        """Detect whether a question mentions more than one quarter."""
        return len(cls._extract_mentioned_quarters(question)) > 1

    def _extract_metric_candidates(self, retrieved_docs, metric: str):
        """Return candidate metric values from retrieved transcript chunks.

        The goal is to prefer exact monetary amounts over nearby growth percentages.
        """
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
            # Accept bare numbers only if they are clearly monetary-scale amounts.
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
        """Prefer full company-quarter filtered docs for exact metric questions.

        Semantic top-k retrieval is useful for general questions, but exact revenue/PAT
        lookups should scan the entire company-quarter slice so the answer is not
        missed just because the relevant chunk was not among the top semantic hits.
        """
        if not self.retriever:
            return retrieved_docs, None, None

        company_id = company_filter or self.retriever._detect_company_intent(
            question)
        quarter = quarter_filter or self.retriever._detect_quarter_intent(
            question)

        if company_id or quarter:
            docs = self.retriever.retrieve_by_filters(
                company_id=company_id,
                quarter=quarter,
            )
            return [(doc, 1.0) for doc in docs], company_id, quarter

        return retrieved_docs, None, None

    def _try_direct_metric_answer(
        self,
        question: str,
        retrieved_docs,
        company_filter: Optional[str] = None,
        quarter_filter: Optional[str] = None,
    ) -> Optional[str]:
        """Build a deterministic answer for exact revenue/PAT questions when the context contains them."""
        requested_metrics = self._detect_requested_metrics(question)

        if not requested_metrics or not retrieved_docs:
            return None

        if self._has_mixed_quarter_request(question):
            return (
                "I can only answer for one quarter at a time. Please ask a separate question for each quarter, or narrow your request to a single quarter."
            )

        lookup_docs, detected_company_id, detected_quarter = self._get_exact_lookup_documents(
            question,
            retrieved_docs,
            company_filter=company_filter,
            quarter_filter=quarter_filter,
        )
        if lookup_docs:
            retrieved_docs = lookup_docs

        company_id = retrieved_docs[0][0].company_id
        
        company_mapping = {
            "532400": "Birlasoft Limited",
            "542652": "Polycab India",
            "543654": "Medanta",
            "544350": "Dr. Agarwal's Eye Hospital"
        }
        company_name = company_mapping.get(company_id, f"Company {company_id}")

        lines = []

        for metric_key in requested_metrics:
            candidates = self._extract_metric_candidates(
                retrieved_docs, metric_key)
            if not candidates:
                return (
                    f"I couldn't find reliable data on the exact {metric_key.replace('_', ' ')} for {company_name} from the provided transcripts."
                )

            metric_display = self._metric_specs()[metric_key]["display"]

            if detected_quarter:
                # If a specific quarter was asked for, just get the best candidate overall
                metric_score, metric_value, metric_text, metric_doc, metric_similarity, metric_position = max(
                    candidates,
                    key=lambda item: (item[0], item[4], item[1], -item[5]),
                )
                q_label = f"{metric_doc.quarter} {metric_doc.year}"
                lines.append(
                    f"Based on the transcript, {company_name} reported a {metric_display} of {metric_text} for {q_label} [{metric_doc.company_id} {metric_doc.quarter}{metric_doc.year} - Match: {metric_similarity:.2%}]."
                )
            else:
                # Group candidates by quarter and get the best one for EACH quarter
                best_per_q = {}
                for cand in candidates:
                    q_key = f"{cand[3].quarter} {cand[3].year}"
                    # cand[0] is metric_score. Keep the one with the highest score
                    if q_key not in best_per_q or cand[0] > best_per_q[q_key][0]:
                        best_per_q[q_key] = cand
                
                lines.append(f"Here is the {metric_display} for {company_name} across the available quarters:")
                for q_key, cand in sorted(best_per_q.items()):
                    metric_score, metric_value, metric_text, metric_doc, metric_similarity, metric_position = cand
                    lines.append(
                        f"• {q_key}: {metric_text} [{metric_doc.company_id} {metric_doc.quarter}{metric_doc.year} - Match: {metric_similarity:.2%}]"
                    )

        return "\n".join(lines)

    @staticmethod
    def _is_exact_metric_question(question: str) -> bool:
        """Detect questions that ask for an exact financial metric."""
        return len(EarningsQACLI._detect_requested_metrics(question)) > 0

    @staticmethod
    def _filter_conflict_message(company_filter: Optional[str], quarter_filter: Optional[str]) -> str:
        """Return a professional message for questions outside the active filters."""
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
        """Detect whether the question explicitly targets a company or quarter outside the active filters."""
        if not retriever:
            return False

        requested_company = retriever._detect_company_intent(question)
        requested_quarter = retriever._detect_quarter_intent(question)

        if company_filter and requested_company and requested_company != company_filter:
            return True
        if quarter_filter and requested_quarter and requested_quarter != quarter_filter:
            return True

        return False

    def _create_index(self) -> bool:
        """Create RAG index from transcripts.

        Returns:
            True if successful, False otherwise
        """
        try:
            click.echo("\n📑 Indexing earnings call transcripts...")

            # Ingest transcripts
            ingestion = TranscriptIngestionPipeline()
            documents = ingestion.ingest_transcripts(DATA_DIR)

            if not documents:
                click.echo("❌ No transcripts found in data directory")
                return False

            click.echo(f"✓ Ingested {len(documents)} document chunks")

            # Create embeddings and index
            embedding_pipeline = EmbeddingPipeline()
            embedding_pipeline.build_index(
                documents, output_path=self.index_path)

            # Initialize retriever
            self.retriever = Retriever(embedding_pipeline)
            self.indexed = True

            click.echo("✓ Index created successfully\n")
            return True

        except Exception as e:
            click.echo(f"❌ Error creating index: {e}\n", err=True)
            logger.error(f"Index creation error: {e}")
            return False

    def _chat_loop(
        self,
        company_filter: Optional[str] = None,
        quarter_filter: Optional[str] = None,
    ) -> None:
        """Interactive chat loop.

        Args:
            company_filter: Optional company to limit queries to
            quarter_filter: Optional quarter to limit queries to
        """
        click.echo("\n" + "="*60)
        click.echo("Earnings Call Q&A Chatbot")
        click.echo("="*60)

        click.echo("Type 'quit' or 'exit' to end conversation")
        click.echo("Type 'clear' to reset conversation history")
        click.echo("="*60 + "\n")

        while True:
            try:
                question = click.prompt("You").strip()

                if question.lower() in ["quit", "exit"]:
                    click.echo("\nGoodbye!")
                    break

                if question.lower() == "clear":
                    self.conversation.clear()
                    click.echo("Conversation history cleared.\n")
                    continue

                if not question:
                    continue

                # Apply scope guardrail
                in_scope, scope_msg = self.validator.check_scope(question)
                if not in_scope:
                    click.echo(f"\nAssistant: {scope_msg}\n")
                    continue

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
                        )
                        click.echo(answer + "\n")

                        self.conversation.add("user", question)
                        self.conversation.add("assistant", answer)
                        log_query(question, answer,
                                  company_filter, quarter_filter)
                        continue

                    retrieved_docs = self.retriever.retrieve(
                        question,
                        company_id=company_filter,
                        quarter=quarter_filter,
                    )
                    if retrieved_docs:
                        context = self.retriever.format_context(retrieved_docs)
                else:
                    click.echo(
                        "\n⚠️  RAG index not loaded. Run with --index first.\n")

                user_message = get_retrieval_prompt(
                    question=question,
                    context=context or "No relevant context found in transcripts.",
                    company_filter=company_filter or "",
                    quarter_filter=quarter_filter or "",
                )

                click.echo("\nAssistant: ", nl=False)

                direct_answer = self._try_direct_metric_answer(
                    question,
                    retrieved_docs,
                    company_filter=company_filter,
                    quarter_filter=quarter_filter,
                )
                if direct_answer:
                    answer = direct_answer
                    answer, guardrail_status = self.validator.apply_guardrails(
                        query=question,
                        response=answer,
                        retrieved_documents=retrieved_docs,
                    )
                    click.echo(answer + "\n")

                    self.conversation.add("user", question)
                    self.conversation.add("assistant", answer)
                    log_query(question, answer, company_filter, quarter_filter)
                    continue

                if self._is_exact_metric_question(question):
                    safe_refusal = (
                        "Direct Answer:\n"
                        "I don't have reliable data on this from the provided transcripts."
                    )
                    answer = safe_refusal
                    answer, guardrail_status = self.validator.apply_guardrails(
                        query=question,
                        response=answer,
                        retrieved_documents=retrieved_docs,
                    )
                    click.echo(answer + "\n")

                    self.conversation.add("user", question)
                    self.conversation.add("assistant", answer)
                    log_query(question, answer, company_filter, quarter_filter)
                    continue

                try:
                    answer = self.llm.answer_question(
                        system_prompt=SYSTEM_PROMPT,
                        user_message=user_message,
                    )

                    # Apply guardrails
                    answer, guardrail_status = self.validator.apply_guardrails(
                        query=question,
                        response=answer,
                        retrieved_documents=retrieved_docs,
                    )

                    click.echo(answer + "\n")

                    # Log interaction
                    self.conversation.add("user", question)
                    self.conversation.add("assistant", answer)
                    log_query(question, answer, company_filter, quarter_filter)

                except Exception as e:
                    click.echo(f"Error calling LLM: {e}\n", err=True)
                    logger.error(f"LLM error: {e}")

            except KeyboardInterrupt:
                click.echo("\n\nInterrupted by user.")
                break
            except Exception as e:
                click.echo(f"\nError: {e}\n", err=True)
                logger.error(f"Unexpected error: {e}")


@click.command()
@click.option(
    "--company",
    type=str,
    default=None,
    help="Filter queries to specific company ID (e.g., 532400)",
)
@click.option(
    "--quarter",
    type=str,
    default=None,
    help="Filter queries to specific quarter (e.g., Q1, Q2, Q3, Q4)",
)
@click.option(
    "--index",
    is_flag=True,
    help="Index transcripts (TODO: implement in Part 2)",
)
@click.option(
    "--list-docs",
    is_flag=True,
    help="List indexed documents (TODO: implement in Part 2)",
)
def main(company: Optional[str], quarter: Optional[str], index: bool, list_docs: bool):
    """Earnings Call Q&A Chatbot - Ask questions about company earnings calls."""

    setup_logging()
    cli = EarningsQACLI()

    if index:
        # Create RAG index
        success = cli._create_index()
        if not success:
            exit(1)
        click.echo("Index ready! Start chat with: python main.py")
        return

    if list_docs:
        # List indexed documents
        if not cli.indexed:
            click.echo("❌ No index loaded. Run with --index first.")
            return

        if not cli.retriever:
            click.echo("❌ Retriever not initialized.")
            return

        click.echo("\n📄 Indexed Documents:\n")
        documents = cli.retriever.documents

        # Group by company and quarter
        by_company = {}
        for doc in documents:
            key = (doc.company_id, doc.quarter)
            if key not in by_company:
                by_company[key] = 0
            by_company[key] += 1

        for (company_id, quarter), count in sorted(by_company.items()):
            click.echo(f"  {company_id} {quarter}: {count} chunks")

        click.echo(f"\nTotal: {len(documents)} chunks indexed\n")
        return

    # Default: Start chat loop
    if not cli.indexed:
        click.echo("⚠️  No RAG index loaded.")
        create_now = click.confirm("Would you like to index transcripts now?")
        if create_now:
            cli._create_index()

    cli._chat_loop(company_filter=company, quarter_filter=quarter)


if __name__ == "__main__":
    main()
