"""Transcript ingestion: Parse PDFs, extract text, chunk into documents."""

import logging
from pathlib import Path
from typing import List, Tuple
import pdfplumber
import json

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
            
            # Calculate Fiscal Year (FY)
            # If month is April (4) or later, it belongs to the NEXT year's FY
            # e.g., August 2024 belongs to FY 2025
            fiscal_year = year + 1 if month >= 4 else year

            return company_id, None, fiscal_year
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
        """Ingest all transcripts from data directory using the manifest.

        Args:
            data_dir: Root data directory

        Returns:
            List of Document chunks from all transcripts
        """
        all_documents = []

        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return all_documents

        manifest_path = Path(__file__).parent.parent.parent / "config" / "transcripts_manifest.json"
        if not manifest_path.exists():
            logger.error(f"Manifest not found: {manifest_path}")
            return all_documents

        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading manifest: {e}")
            return all_documents

        transcripts = manifest_data.get("transcripts", [])
        if not transcripts:
            logger.warning("No transcripts found in manifest")
            return all_documents

        for entry in transcripts:
            company_id = entry.get("company_id")
            quarter = entry.get("quarter")
            fiscal_year = entry.get("fiscal_year")
            source_file_path = entry.get("source_file_path")
            
            if not all([company_id, quarter, fiscal_year, source_file_path]):
                logger.warning(f"Incomplete manifest entry: {entry}")
                continue
                
            pdf_path = data_dir / source_file_path
            
            if not pdf_path.exists():
                logger.warning(f"PDF not found: {pdf_path}")
                continue

            logger.info(f"Processing transcript: {pdf_path.name}")
            text = self.extract_pdf_text(pdf_path)

            if text:
                chunks = self.chunk_text(text, company_id, quarter, fiscal_year)
                all_documents.extend(chunks)

        logger.info(f"Total ingested documents: {len(all_documents)}")
        return all_documents
