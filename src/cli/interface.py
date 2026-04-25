"""Click-based CLI interface for earnings QA chatbot."""

import logging
import json
import re
from datetime import datetime
from typing import Optional, List, Tuple
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

    @classmethod
    def _has_mixed_quarter_request(cls, question: str) -> bool:
        """Detect whether a question mentions more than one quarter."""
        return len(cls._extract_mentioned_quarters(question)) > 1

    @staticmethod
    def _extract_scope_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract the most obvious company id and quarter from a text snippet."""
        text_lower = text.lower()
        company_mapping = {
            "532400": ["bsoft", "birlasoft", "birla soft", "birla soft limited", "532400"],
            "542652": ["polycab", "wires", "542652"],
            "543654": ["medanta", "global health", "543654"],
            "544350": ["agarwal", "544350"],
        }

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
        company_terms = [
            "birlasoft", "bsoft", "medanta", "global health", "polycab", "wires",
            "agarwal", "dr. agarwal", "dr agarwal", "543654", "542652", "532400", "544350"
        ]

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
        """Build a deterministic answer for exact revenue/PAT questions when the context contains them."""
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
                # If we can't find it with Regex, stay silent and let the LLM handle it.
                continue

            metric_display = self._metric_specs()[metric_key]["display"]

            company_mapping = {
                "532400": "Birlasoft Limited",
                "542652": "Polycab India",
                "543654": "Medanta",
                "544350": "Dr. Agarwal's Eye Hospital"
            }

            # Group candidates by (Company, Quarter) and get the best one for EACH pair
            best_per_entity = {}
            for cand in candidates:
                entity_key = (cand[3].company_id,
                              f"{cand[3].quarter} {cand[3].year}")
                if entity_key not in best_per_entity or cand[0] > best_per_entity[entity_key][0]:
                    best_per_entity[entity_key] = cand

            # Sort by company name for organized output
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

        # Return None if no metrics were extracted, so the LLM can take over
        if not lines:
            return None
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

        requested_companies = retriever._detect_company_intents(question)
        requested_quarters = retriever._detect_quarter_intents(question)

        if company_filter and requested_companies and company_filter not in requested_companies:
            return True
        if quarter_filter and requested_quarters and quarter_filter not in requested_quarters:
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
                    click.echo(f"\nAssistant: {scope_msg}\n")
                    continue

                if self._is_vague_growth_reason_question(question, conversation_context=conversation_context):
                    msg = (
                        "Please specify the company and quarter for growth-related questions. "
                        "For example: 'Why did Medanta revenue grow in Q2 FY2025?' or 'What drove Polycab's revenue in Q3?'"
                    )
                    click.echo(f"\nAssistant: {msg}\n")
                    self.conversation.add("user", question)
                    self.conversation.add("assistant", msg)
                    log_query(question, msg, company_filter, quarter_filter)
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
                            conversation_context=conversation_context,
                        )
                        click.echo(answer + "\n")

                        self.conversation.add("user", question)
                        self.conversation.add("assistant", answer)
                        log_query(question, answer,
                                  company_filter, quarter_filter)
                        continue

                    retrieved_docs = self.retriever.retrieve(
                        question,
                        company_ids=[resolved_company_filter] if resolved_company_filter else None,
                        quarters=[resolved_quarter_filter] if resolved_quarter_filter else None,
                    )
                    if retrieved_docs:
                        context = self.retriever.format_context(retrieved_docs)
                else:
                    click.echo(
                        "\n⚠️  RAG index not loaded. Run with --index first.\n")

                click.echo("\nAssistant: ", nl=False)

                direct_answer = self._try_direct_metric_answer(
                    question,
                    retrieved_docs,
                    company_filter=resolved_company_filter,
                    quarter_filter=resolved_quarter_filter,
                )
                final_answer = ""

                if direct_answer:
                    # Print the exact mathematical figures immediately
                    click.echo(direct_answer + "\n")
                    final_answer += direct_answer + "\n\n"

                # If we're here, we need the LLM.
                if direct_answer:
                    # Add a separator to indicate the LLM is now answering the rest
                    click.echo("--- Qualitative Analysis ---\n", nl=False)

                # Call the LLM to cover anything not answered by Regex.
                user_message = get_retrieval_prompt(
                    question=question,
                    context=context or "No relevant context found in transcripts.",
                    company_filter=resolved_company_filter or "",
                    quarter_filter=resolved_quarter_filter or "",
                    conversation_context=conversation_context,
                    already_extracted=direct_answer or "",
                )

                try:
                    llm_answer = self.llm.answer_question(
                        system_prompt=SYSTEM_PROMPT,
                        user_message=user_message,
                    )

                    # Apply guardrails
                    validated_answer, guardrail_status = self.validator.apply_guardrails(
                        query=question,
                        response=llm_answer,
                        retrieved_documents=retrieved_docs,
                        conversation_context=conversation_context,
                    )

                    click.echo(validated_answer + "\n")
                    final_answer += validated_answer

                    # Log interaction
                    self.conversation.add("user", question)
                    self.conversation.add("assistant", final_answer.strip())
                    log_query(question, final_answer.strip(),
                              resolved_company_filter, resolved_quarter_filter)

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
