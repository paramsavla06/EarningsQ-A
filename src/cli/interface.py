"""Click-based CLI interface for earnings QA chatbot."""

import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

import click

from src.config import LOGS_DIR, DATA_DIR
from src.rag import TranscriptIngestionPipeline, EmbeddingPipeline, Retriever
from src.core.chat_service import ChatService

logger = logging.getLogger(__name__)


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


class EarningsQACLI:
    """Main CLI interface for earnings QA chatbot."""

    def __init__(self):
        """Initialize CLI."""
        self.chat_service = ChatService()
        self.indexed = False
        self.index_path = Path(".") / "rag_index"  # Local index storage

        # Try to load existing index
        self._load_index()

    def _load_index(self) -> None:
        """Load existing RAG index if available."""
        if self.index_path.exists():
            try:
                embedding_pipeline = EmbeddingPipeline()
                embedding_pipeline.load_index(self.index_path)
                self.chat_service.retriever = Retriever(embedding_pipeline)
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
            self.chat_service.retriever = Retriever(embedding_pipeline)
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
                    self.chat_service.conversation.clear()
                    click.echo("Conversation history cleared.\n")
                    continue

                if not question:
                    continue

                if not self.indexed:
                    click.echo("\n⚠️  RAG index not loaded. Run with --index first.\n")

                response = self.chat_service.process_message(question, company_filter, quarter_filter)

                click.echo("\nAssistant: ", nl=False)
                
                if response.error_msg:
                    click.echo(response.error_msg + "\n")
                    continue

                if response.direct_answer:
                    click.echo(response.direct_answer + "\n")

                if response.llm_answer:
                    if response.direct_answer:
                        click.echo("--- Qualitative Analysis ---\n", nl=False)
                    click.echo(response.llm_answer + "\n")

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
    help="Index transcripts",
)
@click.option(
    "--list-docs",
    is_flag=True,
    help="List indexed documents",
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

        if not cli.chat_service.retriever:
            click.echo("❌ Retriever not initialized.")
            return

        click.echo("\n📄 Indexed Documents:\n")
        documents = cli.chat_service.retriever.documents

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
