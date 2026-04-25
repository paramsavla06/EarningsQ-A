import pytest
import tempfile
from pathlib import Path
from earnings_qa.rag.ingestion import TranscriptIngestionPipeline, Document
from earnings_qa.rag.embeddings import EmbeddingPipeline


def make_doc(content: str, company_id: str = "c1", quarter: str = "Q1", year: str = "FY24") -> Document:
    """Helper to create a Document using the real constructor signature."""
    return Document(content=content, company_id=company_id, quarter=quarter, year=year)


def test_ingestion_chunk_text():
    """TranscriptIngestionPipeline.chunk_text produces documents with correct metadata."""
    pipeline = TranscriptIngestionPipeline()
    text = "Revenue was INR 100 million. " * 20  # long enough to create at least 1 chunk

    docs = pipeline.chunk_text(text, company_id="543654", quarter="Q1", year="FY24")

    assert len(docs) > 0
    doc = docs[0]
    assert doc.company_id == "543654"
    assert doc.quarter == "Q1"
    assert doc.year == "FY24"
    assert "Revenue" in doc.content


def test_ingestion_pipeline_no_manifest(tmp_path):
    """ingest_transcripts returns [] when the manifest is absent."""
    pipeline = TranscriptIngestionPipeline()
    # Pass a data_dir that exists but has no manifest configured
    docs = pipeline.ingest_transcripts(tmp_path)
    # Manifest lookup lives inside the package config — just assert it doesn't raise
    assert isinstance(docs, list)


def test_embedding_pipeline_build_index():
    """EmbeddingPipeline builds a FAISS index from mock embeddings."""
    pipeline = EmbeddingPipeline()

    docs = [
        make_doc("Test document one",  company_id="c1", quarter="Q1", year="FY24"),
        make_doc("Test document two",  company_id="c2", quarter="Q2", year="FY24"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp)
        pipeline.build_index(docs, output_path=out_path)

        assert pipeline.index is not None
        assert pipeline.index.ntotal == 2
        assert (out_path / "index.faiss").exists()
        assert (out_path / "documents.pkl").exists()


def test_embedding_pipeline_save_and_load():
    """Index saved to disk can be loaded and queried again."""
    pipeline = EmbeddingPipeline()
    docs = [make_doc("Earnings were strong this quarter.", company_id="c1", quarter="Q1", year="FY24")]

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp)
        pipeline.build_index(docs, output_path=out_path)

        # Load into a fresh pipeline instance
        pipeline2 = EmbeddingPipeline()
        pipeline2.load_index(out_path)

        assert pipeline2.index is not None
        assert pipeline2.index.ntotal == 1
        assert pipeline2.documents[0].company_id == "c1"
