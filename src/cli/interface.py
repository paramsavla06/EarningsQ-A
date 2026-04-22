"""Click-based CLI interface for earnings QA chatbot."""

import logging
import json
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
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
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

        filters = []
        if company_filter:
            filters.append(f"Company: {company_filter}")
        if quarter_filter:
            filters.append(f"Quarter: {quarter_filter}")

        if filters:
            click.echo(f"Filters active: {', '.join(filters)}")

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
                    company_filter=company_filter,
                    quarter_filter=quarter_filter,
                )

                click.echo("\nAssistant: ", nl=False)

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
