"""Transcript ingestion: Parse PDFs, extract text, chunk into documents."""

import logging
from pathlib import Path
from typing import List, Tuple
import pdfplumber

from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class Document:
    """Represents a chunked document with metadata."""

    def __init__(self, content: str, company_id: str, quarter: str, year: int, section: str = ""):
        """Initialize document chunk.

        Args:
            content: Text content of the chunk
            company_id: Company identifier (e.g., 532400)
            quarter: Quarter (Q1, Q2, Q3, Q4)
            year: Year (e.g., 2024)
            section: Optional section name (e.g., "CEO_Statement", "Guidance")
        """
        self.content = content
        self.company_id = company_id
        self.quarter = quarter
        self.year = year
        self.section = section
        self.metadata = {
            "company_id": company_id,
            "quarter": quarter,
            "year": year,
            "section": section,
        }

    def __repr__(self) -> str:
        return f"<Document {self.company_id} {self.quarter}{self.year} ({len(self.content)} chars)>"


class TranscriptIngestionPipeline:
    """Pipeline for ingesting earnings call transcripts."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """Initialize ingestion pipeline.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Character overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text
        """
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            logger.info(f"Extracted {len(text)} chars from {pdf_path.name}")
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF {pdf_path}: {e}")
            return ""

    def parse_filename(self, filename: str) -> Tuple[str, str, int]:
        """Parse PDF filename to extract metadata.

        Format: YYYYMMDD_[company_id]_EarningsCallTranscript.pdf

        Args:
            filename: Filename to parse

        Returns:
            Tuple of (company_id, quarter, year)
        """
        try:
            parts = filename.replace(
                "_EarningsCallTranscript.pdf", "").split("_")
            if len(parts) < 2:
                logger.warning(f"Could not parse filename: {filename}")
                return None, None, None

            date_str = parts[0]  # YYYYMMDD
            company_id = parts[1]

            year = int(date_str[:4])
            month = int(date_str[4:6])

            # Determine quarter from month
            quarter = f"Q{(month - 1) // 3 + 1}"

            return company_id, quarter, year
        except Exception as e:
            logger.error(f"Error parsing filename {filename}: {e}")
            return None, None, None

    def chunk_text(self, text: str, company_id: str, quarter: str, year: int) -> List[Document]:
        """Split text into overlapping chunks.

        Args:
            text: Full text to chunk
            company_id: Company identifier
            quarter: Quarter
            year: Year

        Returns:
            List of Document chunks
        """
        chunks = []

        if len(text) == 0:
            return chunks

        # Simple splitting by chunk_size with overlap
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i: i + self.chunk_size]

            if len(chunk_text.strip()) > 50:  # Only keep non-trivial chunks
                doc = Document(
                    content=chunk_text,
                    company_id=company_id,
                    quarter=quarter,
                    year=year,
                )
                chunks.append(doc)

        logger.info(
            f"Created {len(chunks)} chunks for {company_id} {quarter}{year}")
        return chunks

    def ingest_transcripts(self, data_dir: Path = DATA_DIR) -> List[Document]:
        """Ingest all transcripts from data directory.

        Expected structure:
        data_dir/
            ├── 532400/
            │   ├── Q1/
            │   │   └── YYYYMMDD_532400_EarningsCallTranscript.pdf
            │   └── Q2/
            └── 542652/
                └── FY2025/
                    └── Q3/

        Args:
            data_dir: Root data directory

        Returns:
            List of Document chunks from all transcripts
        """
        all_documents = []

        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return all_documents

        # Iterate through company folders
        for company_folder in data_dir.iterdir():
            if not company_folder.is_dir():
                continue

            company_id = company_folder.name
            logger.info(f"Processing company: {company_id}")

            # Find all PDF files recursively
            pdf_files = list(company_folder.rglob("*_EarningsCallTranscript.pdf"))

            if not pdf_files:
                logger.warning(f"No PDFs found in {company_folder}")
                continue  # Skip empty companies

            for pdf_path in pdf_files:
                # Extract text from PDF
                text = self.extract_pdf_text(pdf_path)

                if text:
                    # Parse filename for metadata
                    parsed_company, parsed_quarter, year = self.parse_filename(
                        pdf_path.name)

                    if parsed_company and year:
                        # Chunk the text
                        chunks = self.chunk_text(
                            text, parsed_company, parsed_quarter, year)
                        all_documents.extend(chunks)

        logger.info(f"Total ingested documents: {len(all_documents)}")
        return all_documents
