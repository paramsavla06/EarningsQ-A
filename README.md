# Earnings Call Q&A Chatbot

A production-ready RAG-based chatbot that answers questions about company earnings calls using semantic search and LLM-powered responses.

## Features

- **RAG Pipeline**: Semantic search with FAISS vector store for relevant context retrieval
- **Multi-turn Conversation**: Context-aware dialogue with sliding conversation history
- **PDF Ingestion**: Automatic PDF parsing, text extraction, and intelligent chunking
- **Embeddings**: Google Gemini embeddings with mock mode for development
- **Company & Quarter Filtering**: Filter queries to specific companies or quarters
- **Guardrails**: Scope validation, confidence scoring, response validation
- **Mock LLM Mode**: Develop without API quota limits
- **Audit Logging**: JSONL audit trail of all interactions


## Project Structure

```
ResearchFundamentalAssignment/
├── earnings-qa/                 # This project
│   ├── main.py                  # Entrypoint
│   ├── requirements.txt          # Dependencies
│   ├── .env                      # Configuration (mock mode enabled)
│   ├── .env.example              # Configuration template
│   ├── README.md                 # This file
│   ├── DESIGN.md                 # Architecture decisions
│   ├── IMPLEMENTATION.md         # Detailed implementation guide
│   ├── src/
│   │   ├── config.py             # Settings + constants
│   │   ├── cli/
│   │   │   └── interface.py      # Click CLI, chat loop, multi-turn conversation
│   │   ├── llm/
│   │   │   ├── client.py         # MockLLMClient + Gemini wrapper, retry logic
│   │   │   └── prompts.py        # System prompt + retrieval prompt builder
│   │   ├── rag/
│   │   │   ├── ingestion.py      # PDF extraction, chunking, Document class
│   │   │   ├── embeddings.py     # Embedding generation, FAISS indexing, persistence
│   │   │   └── retriever.py      # Query → top-K retrieval with filtering
│   │   ├── guardrails/
│   │   │   └── validator.py      # Scope checking, confidence scoring, validation
│   │   └── __init__.py
│   ├── logs/                     # Audit logs (JSONL format)
│   └── rag_index/                # Cached FAISS index (auto-created)
│       ├── index.faiss
│       ├── embeddings.npy
│       └── documents.pkl
│
└── files/                        # Dataset (earnings call transcripts)
    ├── 532400/
    │   ├── Q1/
    │   │   └── 20240808_532400_EarningsCallTranscript.pdf
    │   ├── Q2/
    │   ├── Q3/
    │   └── Q4/
    ├── 542652/
    │   └── ... (quarters with PDFs)
    ├── 543654/
    │   └── ... (quarters with PDFs)
    └── 544350/
        └── ... (quarters with PDFs)
```

## Quick Start

### Setup

```bash
# 1. Navigate to the project
cd earnings-qa

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (optional - mock mode enabled by default)
cp .env.example .env
# For production with real LLM: GEMINI_API_KEY=your_key
# For development: USE_MOCK_LLM=true (instant responses, no quota)
```

### Usage - Indexing

```bash
# Create RAG index from earnings call PDFs (../files/)
python main.py --index

# List all indexed documents
python main.py --list-docs

# The index is auto-loaded on subsequent runs
```

### Usage - Chat

```bash
# Interactive chat (all companies, all quarters)
python main.py

# Filter by company ID (e.g., 532400)
python main.py --company 532400

# Filter by company and quarter (e.g., 542652, Q2)
python main.py --company 542652 --quarter Q2
```

### Example Session

```
$ python main.py --company 532400
⚠️ No RAG index loaded.
Would you like to index transcripts now? [y/N]: y
📑 Indexing earnings call transcripts...
✓ Ingested 1,250 document chunks
✓ Index created successfully

============================================================
Earnings Call Q&A Chatbot
============================================================
Filters active: Company: 532400
Type 'quit' or 'exit' to end conversation
============================================================

You: What was the Q1 revenue?
Assistant: Revenue in Q1 was $119.6B, representing 2% YoY growth...

You: What about gross margins?
Assistant: [Context from retriever] Gross margin improved 150 bps to 46.2%...
📊 Confidence: 85% - Strong matching context found.
```

## Architecture Decisions

See [DESIGN.md](DESIGN.md) for detailed design rationale, scalability considerations, and tradeoff analysis.

## Development

```bash
# Run tests (TODO)
pytest

# Format code
black src/ main.py

# Check types (TODO)
mypy src/ main.py
```

## Future Enhancements

- Multi-turn RAG with context refinement
- Async FastAPI server for production deployment
- Redis caching for frequent queries
- Pinecone/Weaviate for enterprise-scale vector store
- Advanced query normalization (ticker → company name mapping)
- Fine-tuned embedding models for domain-specific matching
