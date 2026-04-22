# Implementation Summary - Earnings Call Q&A Chatbot

## Overview
Completed implementation of a production-quality earnings call Q&A chatbot with RAG pipeline and guardrails, demonstrating strong architecture and scalability thinking for an MLE take-home assignment.

**Project Structure:**
- **Part 1** (Complete): CLI skeleton with LLM integration ✅
- **Part 2** (Complete): RAG pipeline with PDF ingestion, embeddings, and retrieval ✅
- **Part 3** (Complete): Guardrails with scope validation, confidence scoring, and response validation ✅

---

## Architecture Overview

### Tech Stack
| Component | Technology | Why |
|-----------|-----------|-----|
| **LLM** | Google Gemini `gemini-2.0-flash` | Latest model, competitive API, cost-effective |
| **Embeddings** | Gemini `models/text-embedding-004` | Integrated with LLM provider, no extra setup |
| **Vector Store** | FAISS (Local) | No infrastructure, fast for small-medium datasets |
| **CLI** | Click 8.2.1+ | Cleaner than argparse, type hints support |
| **Data Pipeline** | pdfplumber + pandas | Robust PDF parsing, data manipulation |
| **Retry Logic** | tenacity | Exponential backoff for API reliability |
| **Dev Mode** | Mock LLM | Bypass API quota limits during development |

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      INITIALIZATION PHASE                       │
│  python main.py --index                                        │
│  ├─ TranscriptIngestionPipeline.ingest_transcripts()          │
│  │  └─ Extract PDFs → Parse metadata → Chunk text             │
│  │     (returns List[Document] with metadata)                 │
│  ├─ EmbeddingPipeline.build_index()                           │
│  │  └─ Embed each chunk → Create FAISS index → Save disk      │
│  └─ Index ready for queries                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      CHAT PHASE                                 │
│  python main.py                                                │
│  ├─ Load existing index (or create new)                        │
│  ├─ User enters query                                          │
│  ├─ GuardrailValidator.check_scope() → Is query in scope?     │
│  ├─ Retriever.retrieve()                                       │
│  │  └─ Embed query → Search FAISS → Get top-K → Format       │
│  ├─ LLM.answer_question(system_prompt + retrieval_prompt)     │
│  ├─ GuardrailValidator.apply_guardrails()                     │
│  │  └─ Check confidence → Add notes → Validate response       │
│  └─ Output answer with confidence metadata                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 1: CLI & LLM Integration ✅

### Files
- `main.py` - Entrypoint, delegates to CLI
- `src/config.py` - Centralized configuration (mock mode, API keys, paths)
- `src/llm/client.py` - MockLLMClient + real LLMClient with factory pattern
- `src/llm/prompts.py` - System prompt, retrieval prompt builder
- `src/cli/interface.py` - Click-based CLI with conversation history

### Key Features
✅ Interactive chat with multi-turn conversation history (sliding window)
✅ MockLLMClient with keyword-based responses (development mode)
✅ Real LLMClient using Gemini API with retry logic (production)
✅ Factory pattern for flexible LLM switching
✅ Audit logging of all interactions (JSONL format)
✅ Environment configuration with .env support

### Example: Start Chat
```bash
python main.py                    # Mock mode (no API calls)
python main.py --company 532400   # Filter by company
python main.py --quarter Q1       # Filter by quarter
```

---

## Part 2: RAG Pipeline ✅

### 2.1 Ingestion (`src/rag/ingestion.py`)

**Purpose:** Parse PDFs, extract text, chunk into overlapping documents

**Class: TranscriptIngestionPipeline**
```python
pipeline = TranscriptIngestionPipeline()
documents = pipeline.ingest_transcripts(Path("files"))
# Returns: List[Document] with content + metadata
```

**Process:**
1. Walk directory tree: `files/{company_id}/{quarter}/`
2. Find all `*_EarningsCallTranscript.pdf` files
3. Extract text using pdfplumber
4. Parse filename for metadata: `YYYYMMDD_{company_id}_...`
5. Split text into overlapping chunks (500 chars, 50 overlap)
6. Return Document objects with full metadata

**Document Structure:**
```python
Document(
    content="...",
    company_id="532400",
    quarter="Q1",
    year=2024,
    section="",  # Future: CEO_Statement, Guidance, etc.
    metadata={...}
)
```

### 2.2 Embeddings (`src/rag/embeddings.py`)

**Purpose:** Generate embeddings and create FAISS index

**Class: EmbeddingPipeline**
```python
pipeline = EmbeddingPipeline()
embeddings = pipeline.embed_documents(documents)  # Batch embed
index = pipeline.create_index(embeddings)         # Create FAISS
pipeline.save_index(Path("rag_index"))           # Persist
```

**Features:**
- **Mock Mode:** Random 768-dim vectors (no API calls)
- **Real Mode:** Gemini API embeddings with fallback
- **Batch Processing:** Efficient embedding of large document sets
- **Persistence:** Save/load index + embeddings + documents
- **Progress Logging:** Real-time embedding progress

**FAISS Index:**
- Type: `IndexFlatL2` (Euclidean distance)
- Dimension: 768
- Distance metric: Lower = more similar

### 2.3 Retrieval (`src/rag/retriever.py`)

**Purpose:** Query index and retrieve relevant documents

**Class: Retriever**
```python
retriever = Retriever(embedding_pipeline)
results = retriever.retrieve(
    query="What are revenue expectations?",
    top_k=5,
    company_id="532400",      # Optional filter
    quarter="Q1"              # Optional filter
)
# Returns: List[(Document, similarity_score)]

context = retriever.format_context(results)  # Format for LLM
```

**Process:**
1. Embed user query
2. Search FAISS index (return top-K×2 candidates for filtering)
3. Apply company/quarter filters if specified
4. Convert L2 distances to similarity scores: `1 / (1 + distance)`
5. Sort by relevance, return top-K
6. Format documents into context string for LLM

**Context Formatting:**
```
[532400 Q1 2024 - Match: 85%]
Revenue grew 2% YoY to $119.6B...

---

[532400 Q1 2024 - Match: 72%]
Gross margin improved 150 bps to 46.2%...
```

---

## Part 3: Guardrails ✅

### Files
- `src/guardrails/validator.py` - GuardrailValidator class

### Class: GuardrailValidator

**1. Scope Validation**
```python
in_scope, reason = validator.check_scope("What's the revenue?")
# Returns: (True, "Query is in scope.")

in_scope, reason = validator.check_scope("Tell me a joke")
# Returns: (False, "Your question contains 'joke', which is outside...")
```

**Scope Logic:**
- ✅ Earnings keywords: revenue, profit, margin, growth, guidance, etc.
- ❌ Out-of-scope: recipe, movie, weather, politics, jokes, etc.

**2. Confidence Scoring**
```python
confidence, level = validator.check_confidence(retrieved_docs)
# Returns: (0.75, "HIGH")
```

**Confidence Levels:**
- `HIGH` (≥0.7): Strong matches found
- `MEDIUM` (0.4-0.7): Moderate matches
- `LOW` (<0.4): Weak matches or no results

**3. Response Validation**
```python
is_valid, msg = validator.validate_response(answer)
# Checks: length, mock responses, uncertainty language, citations
```

**4. Guardrails Pipeline**
```python
final_answer, status = validator.apply_guardrails(
    query="...",
    response="...",
    retrieved_documents=[...]
)
# Returns: (answer_with_notes, "VALID"/"INVALID")
```

**Applied Transformations:**
- ⚠️ If no documents retrieved: "No relevant context found. For accurate information, please ensure the relevant earnings call PDF has been indexed."
- 📊 If confidence < 50%: "Confidence: 45% - Limited matching context found."

---

## CLI Integration

### `src/cli/interface.py` - Updated for RAG + Guardrails

**New Methods in EarningsQACLI:**
```python
def _load_index(self) -> None:
    """Load existing RAG index from disk if available."""

def _create_index(self) -> bool:
    """Create RAG index from transcripts in data directory."""
```

**Updated `_chat_loop()` Flow:**
```
1. Get user query
2. GuardrailValidator.check_scope() → Accept/reject
3. Retriever.retrieve(query, company_filter, quarter_filter)
4. LLM.answer_question(system_prompt + retrieval_prompt)
5. GuardrailValidator.apply_guardrails() → Add confidence notes
6. Output answer + log interaction
```

---

## Commands & Usage

### Index Transcripts
```bash
$ python main.py --index
📑 Indexing earnings call transcripts...
✓ Ingested 1,250 document chunks
✓ Index created successfully
```

### List Indexed Documents
```bash
$ python main.py --list-docs
📄 Indexed Documents:

  532400 Q1: 287 chunks
  532400 Q2: 295 chunks
  542652 Q1: 234 chunks
  542652 Q2: 241 chunks
  
Total: 1,057 chunks indexed
```

### Start Chat (Mock Mode)
```bash
$ python main.py
⚠️ No RAG index loaded.
Would you like to index transcripts now? [y/N]: y
📑 Indexing earnings call transcripts...
✓ Ingested 1,250 document chunks
✓ Index created successfully

============================================================
Earnings Call Q&A Chatbot
============================================================
Type 'quit' or 'exit' to end conversation
Type 'clear' to reset conversation history
============================================================

You: What was Q1 revenue?