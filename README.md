# Earnings Call Q&A Chatbot

A production-ready, modular **Hybrid Financial RAG** system that answers questions about company earnings-call transcripts.  
Exact financial figures (revenue, PAT, EBITDA, margins, etc.) are extracted deterministically via regex *before* the LLM is ever called, so factual answers are precise and do not hallucinate. The retriever and LLM then fill in context, reasoning, and follow-up analysis.

---

## Table of Contents

1. [What It Does](#what-it-does)  
2. [Architecture](#architecture)  
3. [Supported Companies](#supported-companies)  
4. [Project Layout](#project-layout)  
5. [Requirements](#requirements)  
6. [Setup & Installation](#setup--installation)  
7. [Configuration](#configuration)  
8. [Build the Index](#build-the-index)  
9. [Run the Chatbot](#run-the-chatbot)  
10. [Running Tests](#running-tests)  
11. [Observability](#observability)  
12. [Design Decisions & Assumptions](#design-decisions--assumptions)  
13. [Limitations](#limitations)  
14. [Module Reference](#module-reference)

---

## What It Does

- Answers questions about company earnings calls, organised by company and quarter
- **Deterministic metric extraction** for revenue, total income, PAT, EBITDA, gross/operating margin, cash flow, capex, PBT, and ARPOB — before any LLM call
- **Hybrid RAG**: FAISS semantic retrieval with keyword boosting and diversity-aware ranking
- Company and quarter **session filters** (`--company`, `--quarter`) for narrow, focused queries
- **Multi-turn conversation memory** — follow-up questions resolve company/quarter from context automatically
- **Three-layer caching**: embeddings → retrieval results → final LLM answers, all keyed by prompt version + index version + history hash
- **Background indexing** (`--index-bg`) so the chat loop is never blocked
- **Staleness detection** — warns when the transcript manifest has changed since the index was last built
- **Structured observability** — every turn emits a JSONL telemetry record to `logs/telemetry.jsonl`
- **Versioned prompts and guardrails** — changes to `PROMPT_VERSION` or `GUARDRAIL_VERSION` auto-invalidate cached answers
- Rejects vague or off-topic prompts instead of guessing
- Pluggable backends: **Google Gemini** (default), **Ollama** (local), or **Mock** (offline dev/test)

---

## Architecture

```
User question
  → scope / guardrail check       (earnings_qa/guardrails/validator.py)
  → cache lookup                  (earnings_qa/core/cache.py)
      cache hit → return immediately
  → exact metric extraction       (earnings_qa/core/chat_service.py)
      match found → direct answer (no LLM)
  → conversation history lookup   (ChatService._resolve_scope_from_history)
  → FAISS retrieval + boosting    (earnings_qa/rag/retriever.py)
  → context assembly
  → LLM response generation       (earnings_qa/llm/client.py)
  → guardrail validation
  → cache write + telemetry emit  (earnings_qa/core/observability.py)
  → ChatResponse(request_id, direct_answer, llm_answer)
```

The pipeline is split so that precise questions do **not** depend on free-form generation, reducing hallucination risk on factual queries.

---

## Supported Companies

| Company ID | Name | Sector |
|---|---|---|
| `532400` | Birlasoft | IT Services |
| `542652` | Polycab | Cable & Wire Manufacturing |
| `543654` | Medanta / Global Health | Healthcare |
| `544350` | Dr. Agarwal's Eye Hospital | Eye Care |

Companies and their aliases are loaded from `earnings_qa/config/company_catalog.json`.  
Transcripts are declared in `earnings_qa/config/transcripts_manifest.json`.  
**No code changes are needed to add a new company or quarter** — edit the manifest files only.

---

## Project Layout

```
earnings-qa/
├── main.py                          # Compatibility shim — runs the CLI
├── pyproject.toml                   # Package definition + console_scripts entry point
├── requirements.txt                 # Pinned runtime + dev dependencies
├── README.md
├── WALKTHROUGH.md
├── TESTING.md
├── check_imports.py                 # Quick sanity check for all imports
│
├── earnings_qa/                     # Installable Python package
│   ├── config.py                    # All settings, env-var driven paths
│   ├── config/
│   │   ├── company_catalog.json     # Company IDs, names, aliases, sectors
│   │   └── transcripts_manifest.json # Per-transcript metadata + checksums
│   │
│   ├── cli/
│   │   └── interface.py             # Click CLI, chat loop, staleness check
│   │
│   ├── core/
│   │   ├── chat_service.py          # Main orchestrator (retrieval → extraction → LLM)
│   │   ├── cache.py                 # Three-layer persistent cache (pickle + SHA-256 keys)
│   │   └── observability.py        # Structured JSONL telemetry per turn
│   │
│   ├── llm/
│   │   ├── backend.py               # Abstract LLMBackend interface (sync + async)
│   │   ├── client.py                # Gemini, Ollama, Mock implementations
│   │   └── prompts.py               # Versioned system + retrieval prompts
│   │
│   ├── rag/
│   │   ├── backend.py               # Abstract RetrieverBackend interface (sync + async)
│   │   ├── ingestion.py             # PDF parsing, chunking, manifest-driven ingestion
│   │   ├── embeddings.py            # Batch embedding, FAISS build/load, metadata.json
│   │   └── retriever.py             # Semantic retrieval, keyword boosting, diversity ranking
│   │
│   └── guardrails/
│       └── validator.py             # Scope check, confidence scoring, response validation
│
├── tests/
│   ├── conftest.py                  # Session-scoped mock env setup
│   ├── test_async.py                # Async adapter tests (LLM + retriever)
│   ├── test_extraction.py           # Deterministic metric regex tests
│   ├── test_guardrails.py           # Scope, confidence, guardrail tests
│   ├── test_indexing.py             # Ingestion, FAISS build/load tests
│   └── test_observability_versioning_cache.py  # Telemetry, versioning, cache invalidation
│
├── logs/
│   ├── audit.jsonl                  # Per-turn audit log (question + answer excerpt)
│   └── telemetry.jsonl              # Structured observability records
│
└── rag_index/                       # Built FAISS index (auto-created on --index)
    ├── index.faiss
    ├── documents.pkl
    └── metadata.json                # manifest_hash for staleness detection
```

> **Data directory**: transcript PDFs are read from `../files/` (one level above the project root) by default.  
> Override with the `EARNINGS_QA_DATA_DIR` environment variable.

---

## Requirements

**Python 3.10 or later** is required (uses `match` statements and `asyncio.to_thread`).

```
click>=8.2.1,<9.0.0
python-dotenv>=1.0.0,<2.0.0
pdfplumber>=0.10.0,<0.12.0
faiss-cpu>=1.7.4,<2.0.0
numpy>=2.0.0,<2.3.0
google-genai>=1.0.0,<2.0.0
google-api-core>=2.11.0,<3.0.0
tenacity>=8.2.3,<10.0.0
anyio>=4.8.0,<5.0.0
requests>=2.31.0,<3.0.0
pytest>=9.0.0,<10.0.0   # dev / test only
```

All versions are pinned in [`requirements.txt`](requirements.txt).

---

## Setup & Installation

### 1 — Clone or unzip the project

```bash
# From a zip file
unzip earnings-qa.zip
cd earnings-qa

# Or clone from GitHub
git clone <repo-url>
cd earnings-qa
```

### 2 — Create and activate a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — (Optional) Install as an editable package

This makes the `earnings-qa` console command available anywhere in the environment:

```bash
pip install -e .
```

### 5 — Configure your environment

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY (or choose Ollama / Mock mode)
```

---

## Configuration

All settings are read from environment variables (`.env` file via `python-dotenv`).

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | *(required unless mock/ollama)* | Google Gemini API key |
| `LLM_MODEL` | `gemini-2.0-flash` | Gemini chat model |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Gemini embedding model |
| `USE_MOCK_LLM` | `false` | Fully offline mock — no API calls, for testing |
| `USE_OLLAMA` | `false` | Use Ollama for both embeddings and generation |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_LLM_MODEL` | `llama3.2:latest` | Ollama chat model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `USE_OLLAMA_EMBED` | `$USE_OLLAMA` | Override: Ollama for embeddings only |
| `USE_OLLAMA_LLM` | `$USE_OLLAMA` | Override: Ollama for generation only |
| `MAX_CONVERSATION_HISTORY` | `10` | Exchanges kept in memory |
| `CHUNK_SIZE` | `500` | Transcript chunk size in characters |
| `CHUNK_OVERLAP` | `50` | Character overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Chunks returned by the retriever |
| `EARNINGS_QA_DATA_DIR` | `../files` | Override transcript PDF source directory |
| `EARNINGS_QA_LOGS_DIR` | `./logs` | Override log output directory |
| `EARNINGS_QA_CATALOG_PATH` | *(bundled)* | Override path to `company_catalog.json` |

If neither `USE_MOCK_LLM` nor `USE_OLLAMA` is set, `GEMINI_API_KEY` **must** be present.

---

## Build the Index

Transcripts are declared in `earnings_qa/config/transcripts_manifest.json`. The ingestion pipeline reads each entry, extracts text from the PDF, chunks it, and stores a FAISS vector index in `rag_index/`.

```bash
# Blocking — waits until indexing is complete before starting chat
python main.py --index

# Non-blocking — starts the chat loop immediately, indexes in the background
python main.py --index-bg

# List what is indexed (company / quarter / chunk count)
python main.py --list-docs
```

If the manifest has changed since the last index build, the chatbot will warn you on startup:

```
⚠️  WARNING: Transcripts manifest has changed since the index was built.
   Please run `python main.py --index` to update the RAG index.
```

---

## Run the Chatbot

### Using `python main.py` (always works)

```bash
# Full corpus, no filter
python main.py

# Restrict to one company
python main.py --company 543654

# Restrict to one company and quarter
python main.py --company 542652 --quarter Q3
```

### Using the installed CLI entry point (after `pip install -e .`)

```bash
earnings-qa
earnings-qa --company 543654
earnings-qa --company 542652 --quarter Q3
earnings-qa --index
earnings-qa --list-docs
```

### In-chat commands

| Command | Effect |
|---|---|
| `clear` | Reset conversation history |
| `quit` / `exit` | End the session |

### Example session

```
You: What was Medanta's revenue in Q1?

Direct Answer:
Revenue for Global Health:
• Q1 FY24: INR1,274 million [543654 Q1FY24 - Match: 87.42%]

[LLM analysis follows with qualitative context and growth drivers...]
```

---

## Running Tests

The test suite runs entirely offline — no API keys or network access required (`USE_MOCK_LLM=true` is set automatically by `conftest.py`).

```bash
# Run all 44 tests
python -m pytest tests/ -v

# Run a specific module
python -m pytest tests/test_extraction.py -v
python -m pytest tests/test_observability_versioning_cache.py -v
```

| Test file | What it covers |
|---|---|
| `test_extraction.py` | Deterministic metric regex, quarter parsing, vague-growth detection |
| `test_guardrails.py` | Scope allow/block, confidence scoring, history-aware override |
| `test_indexing.py` | PDF chunking, FAISS build, save/load roundtrip |
| `test_async.py` | `retrieve_async`, `retrieve_by_filters_async`, `answer_question_async` |
| `test_observability_versioning_cache.py` | Telemetry JSONL schema, request_id, cache invalidation on version change, cache-hit bypasses LLM, history propagation |

---

## Observability

Every chat turn writes one structured JSON record to `logs/telemetry.jsonl`:

```json
{
  "request_id": "f3a1...",
  "timestamp": "2024-10-01T10:05:23.412Z",
  "question_hash": "3fa2c...",
  "query_type": "metric_lookup",
  "company_id": "543654",
  "quarter": "Q1",
  "retrieval_count": 5,
  "retrieval_confidence": 0.872,
  "direct_answer_used": true,
  "cache_hit": false,
  "latency_ms": 312.4,
  "backend_llm": "LLMClient",
  "prompt_version": "v1.0.0",
  "guardrail_version": "v1.0.0",
  "guardrail_status": "VALID",
  "error": null
}
```

> Raw question text is **not** stored in telemetry — only a truncated SHA-256 hash — to avoid logging PII. Full question text is in `logs/audit.jsonl`.

---

## Design Decisions & Assumptions

- **Deterministic extraction before RAG**: regex patterns for each metric are applied to retrieved chunks before the LLM is called. This eliminates hallucination risk for simple factual questions.
- **Metadata-driven ingestion**: `transcripts_manifest.json` is the single source of truth. Adding a new company or quarter requires only a manifest entry — no code changes.
- **Manifest hash for staleness detection**: an MD5 of the manifest is stored in `rag_index/metadata.json` at build time and compared on every startup. A mismatch triggers a warning.
- **Three-layer cache**: embedding vectors, retrieval result sets, and final answers are all cached separately. Cache keys include `prompt_version` + `guardrail_version` + `index_version` (manifest hash) + conversation history hash, so any meaningful change forces a cache miss automatically.
- **Async adapters via `asyncio.to_thread`**: the synchronous retrieval and LLM logic is unchanged; the async methods wrap it in thread-pool calls, making it safe to call from any async framework (e.g. FastAPI) without refactoring the core.
- **FAISS for local retrieval**: keeps setup simple with no hosted vector database dependency. The `RetrieverBackend` interface allows swapping to Pinecone or Weaviate without touching the orchestration layer.
- **Assumption — data directory**: PDFs are expected at `../files/` relative to the project root. Override with `EARNINGS_QA_DATA_DIR`.
- **Assumption — fiscal year mapping**: months April–March map to the *next* calendar year's fiscal year (FY2025 = April 2024 – March 2025), matching Indian corporate reporting conventions.

---

## Limitations

- Conversation history is **in-memory only** and resets when `clear` is called or the process restarts
- The FAISS index is **local per environment** — different machines may diverge if built from different transcript sets
- Background indexing (`--index-bg`) uses a daemon thread; if the process exits before indexing completes, the index is not persisted
- Ollama streaming mode is CPU-bound; large models on CPU can be slow

---

## Module Reference

| Module | Purpose |
|---|---|
| [`earnings_qa/cli/interface.py`](earnings_qa/cli/interface.py) | Click CLI, chat loop, staleness check, background indexer |
| [`earnings_qa/core/chat_service.py`](earnings_qa/core/chat_service.py) | Main orchestrator: retrieval → extraction → LLM → guardrails → telemetry |
| [`earnings_qa/core/cache.py`](earnings_qa/core/cache.py) | Thread-safe three-layer persistent cache |
| [`earnings_qa/core/observability.py`](earnings_qa/core/observability.py) | Structured JSONL telemetry emitter |
| [`earnings_qa/rag/ingestion.py`](earnings_qa/rag/ingestion.py) | PDF parsing, text chunking, manifest-driven ingestion |
| [`earnings_qa/rag/embeddings.py`](earnings_qa/rag/embeddings.py) | Batch embedding, FAISS index build/load, metadata.json |
| [`earnings_qa/rag/retriever.py`](earnings_qa/rag/retriever.py) | Semantic retrieval, keyword boosting, diversity ranking, async adapters |
| [`earnings_qa/guardrails/validator.py`](earnings_qa/guardrails/validator.py) | Scope check, confidence scoring, response validation |
| [`earnings_qa/llm/client.py`](earnings_qa/llm/client.py) | Gemini, Ollama, Mock LLM backends with retry and async wrappers |
| [`earnings_qa/llm/prompts.py`](earnings_qa/llm/prompts.py) | Versioned system prompt and retrieval prompt templates |
| [`earnings_qa/config.py`](earnings_qa/config.py) | All settings; env-var driven paths; company catalog loader |
