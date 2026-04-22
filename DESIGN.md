# Design Document: Earnings Call Q&A Chatbot

## Executive Summary

This document outlines the architecture, design decisions, and tradeoff analysis for an earnings call Q&A chatbot. The system enables users to ask natural language questions about company earnings calls and receive context-grounded, sourced answers.

## 1. Architecture Overview

### Layered Architecture

```
┌─────────────────────────────────────────────┐
│       User Interface Layer (CLI)            │
│   - Click-based interactive chat loop       │
│   - Query filtering (company, quarter)      │
│   - Audit logging                           │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│    Orchestration Layer (Main Chat Logic)    │
│   - Conversation history management         │
│   - Prompt assembly                         │
│   - Response validation                     │
└─────────────────────────────────────────────┘
                    ↓
         ┌──────────┴──────────┐
         ↓                     ↓
┌──────────────────┐  ┌──────────────────┐
│   Guardrails     │  │   LLM Layer      │
│   - Scope check  │  │   - OpenAI GPT   │
│   - Confidence   │  │   - Retry logic  │
│   - Validation   │  │   - Tokenization │
└──────────────────┘  └──────────────────┘
         ↑                     ↑
         └──────────┬──────────┘
                    ↓
┌─────────────────────────────────────────────┐
│        RAG Pipeline (Part 2)                │
│   - Document ingestion & parsing            │
│   - Semantic embeddings (text-embedding-3)  │
│   - Vector storage (FAISS)                  │
│   - Retrieval with filtering                │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│       Data Layer                            │
│   - Raw transcripts (data/transcripts/)     │
│   - FAISS indexes                           │
│   - Metadata (company, quarter, section)    │
│   - Audit logs (JSONL format)               │
└─────────────────────────────────────────────┘
```

## 2. Core Components

### 2.1 CLI Interface (`src/cli/interface.py`)

**Responsibility**: User interaction and experience

**Features**:
- Interactive chat loop using Click
- Multi-turn conversation with history management
- Optional company ID/quarter filtering (e.g., --company 532400 --quarter Q1)
- Commands: `clear`, `quit`, `list-docs`, `--index`
- Graceful error handling with helpful error messages

**Design Rationale**:
- Click chosen over argparse for cleaner syntax and better UX
- Conversation history limited to configurable window (default: 10 exchanges) to manage token costs
- Audit logging on every query for debugging and monitoring

**Key Classes**:
- `ConversationHistory`: Manages multi-turn context with token-aware truncation
- `EarningsQACLI`: Main chat orchestration logic

### 2.2 LLM Layer (`src/llm/`)

**`client.py`**: OpenAI API wrapper

- Handles all API calls with exponential backoff retry logic
- Catches `RateLimitError` and `APIError` for graceful degradation
- Rough token counting for cost estimation
- Uses `tenacity` library for robust retry mechanisms

**Design Rationale**:
- Wrapper pattern allows easy swapping of LLM providers (e.g., Anthropic, Gemini)
- Retry logic with exponential backoff prevents API spam during rate limiting
- Token counting helps manage context size and costs

**`prompts.py`**: Prompt engineering

- `SYSTEM_PROMPT`: Core behavior definition
  - Emphasizes context-only answers (prevents hallucination)
  - Demands explicit "I don't know" responses
  - Sets formatting expectations (quotes, citations)
- `get_retrieval_prompt()`: Assembles user message with retrieved context
- `get_scope_check_prompt()`: Quick intent classification (not implemented in Part 1, for Part 3)

**Key Safeguards**:
```
IMPORTANT CONSTRAINTS:
1. Answer ONLY from provided context
2. Explicit "I don't have this data" responses
3. Citations with quarter/company
4. Acknowledge uncertainty
```

### 2.3 Configuration (`src/config.py`)

Centralized configuration with environment variable overrides:

| Setting | Default | Purpose |
|---------|---------|---------|
| `LLM_MODEL` | `gpt-4o-mini` | Model selection |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | For RAG embeddings |
| `MAX_CONVERSATION_HISTORY` | 10 | Conversation window |
| `CHUNK_SIZE` | 500 | Token size for chunking |
| `CHUNK_OVERLAP` | 50 | Overlap for context preservation |
| `TOP_K_RETRIEVAL` | 5 | Number of chunks to retrieve |
| `MAX_RETRIES` | 3 | API retry attempts |

**Design Rationale**:
- Single source of truth prevents hardcoding
- Environment variables allow production customization without code changes
- Default values chosen for balance between cost (gpt-4o-mini is cheap) and quality

### 2.4 Logging & Audit Trail

**Audit Log (`logs/audit.jsonl`)**:
```json
{
  "timestamp": "2024-01-30T14:23:45.123456",
  "question": "What was Apple's revenue growth?",
  "answer": "Apple reported revenue of $119.6 billion, up 2% year-over-year...",
  "company_filter": "AAPL",
  "quarter_filter": "Q1"
}
```

**Purpose**: 
- Track all queries for debugging
- Monitor usage patterns
- Support compliance/audit requirements
- Enable quality analysis (what users ask, what we answer)

**Design Rationale**:
- JSONL format allows easy streaming and processing
- Truncated answers (500 chars) balance storage and usefulness
- Privacy-safe by default (no PII collection)

## 3. Data Flow: Single Query

```
User Input
    ↓
[PART 1] CLI parses query + filters
    ↓
[PART 1] LLM client adds to conversation history
    ↓
[PART 2] RAG pipeline retrieves top-K relevant chunks
    ↓
[PART 1] Prompt assembly with context
    ↓
[PART 3] Scope guardrail: Is this about earnings?
    ↓
[PART 3] Confidence check: Retrieved chunks > threshold?
    ↓
OpenAI API Call (with retry logic)
    ↓
Response received
    ↓
[PART 1] Add to conversation history
    ↓
[PART 1] Log to audit trail
    ↓
Display to user
```

## 4. Technology Choices & Tradeoffs

### Why OpenAI gpt-4o-mini?

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **gemini-2.0-flash** | Fast, cheap, excellent reasoning | Newer model | ✅ CHOSEN |
| gemini-1.5-pro | Most capable | More expensive, slower | Use if budget allows |
| gpt-4o-mini | Good quality | Requires separate API key | Could swap in with adapter |
| Claude 3 | Great context window | Different API | Could swap in with adapter |

### Why FAISS for vector store?

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **FAISS** | Local, free, scales to 100M vectors | No persistence after restart | ✅ CHOSEN for take-home |
| Pinecone | Managed, persistent, scalable | Paid service, external dependency | Use in production |
| Weaviate | Open-source, full-featured | More complex setup | Use if multi-user |
| Milvus | Scalable, open-source | Requires Docker | Use if containerized |

**Decision**: FAISS is perfect for take-home (zero infra), but Part 2 will note: "For 100+ companies, upgrade to Pinecone; for production multi-user, use Weaviate."

### Why pure Python over LangChain?

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Pure Python** | Shows architecture understanding, minimal deps, full control | More code to write | ✅ CHOSEN |
| LangChain | Batteries included, SOTA integrations | Black-box abstractions, heavy | Indicates framework over fundamentals |
| LlamaIndex | RAG-focused, good docs | Similar to LangChain trade-offs | Same concern |

**Decision**: Pure Python shows I understand the fundamentals (chunking, embedding, retrieval, prompting). LangChain is great for production, but for eval, custom code is more impressive.

## 5. Scalability Roadmap

### Stage 1: Single-Company Take-Home (Current)
- FAISS in-memory
- Single-user CLI
- Hundreds of documents
- **Deployment**: Laptop/small server

### Stage 2: Multi-Company SaaS
- Upgrade: Pinecone + FastAPI
- Per-company vector namespacing
- User authentication (OAuth2)
- Caching with Redis for hot queries
- **Deployment**: Docker, AWS EC2

### Stage 3: Enterprise
- Upgrade: Weaviate (multi-tenant indexing)
- Async job queue for bulk ingestion (Celery + RabbitMQ)
- Fine-tuned embedding model for domain
- Query analytics dashboard
- **Deployment**: Kubernetes, multi-region

**Cost Evolution**:
- Stage 1: $0 (FAISS free) + API calls (~$1/1000 queries)
- Stage 2: $200/mo (Pinecone) + $5/mo (FastAPI hosting) + caching
- Stage 3: $1000+/mo (Weaviate enterprise + ML infra)

## 6. Quality & Safety

### Hallucination Prevention

1. **Prompt Design**: Explicit instruction to only use provided context
2. **Confidence Scoring** (Part 3): Flag low-confidence matches
3. **Citation Requirements**: Demand quarter/company attribution
4. **Retrieval Validation** (Part 3): Reject if top-K similarity < threshold

### Scope Validation

1. **Intent Classification** (Part 3): Quick LLM call to check if question is about earnings
2. **Graceful Decline**: "I specialize in earnings call analysis. This question is outside my scope."
3. **User Feedback**: Suggestions for rephrased questions

### Privacy & Compliance

- No PII collection by default
- Audit logs are immutable (append-only JSONL)
- Optional redaction for sensitive queries
- GDPR-ready: Can purge logs by user/timestamp

## 7. Implementation Phases

### Part 1: CLI + LLM Skeleton (2 hours) — COMPLETED
- [x] Click CLI with interactive loop
- [x] OpenAI client with retry logic
- [x] Prompt engineering (system + user)
- [x] Conversation history management
- [x] Audit logging
- [x] Config system

**Not implemented yet**:
- RAG pipeline (retrieval returns empty)
- Guardrails (scope checking, confidence scoring)
- Sample transcripts with real indexing

### Part 2: RAG Pipeline (3 hours) — TODO
- [ ] PDF parsing and text extraction (PyPDF2 or pdfplumber)
- [ ] Transcript ingestion (parse, chunk, metadata extraction)
- [ ] Embedding pipeline (text-embedding-3-small)
- [ ] FAISS indexing and retrieval
- [ ] Metadata filtering (company_id, quarter)
- [ ] Retrieval evaluation (MRR, NDCG)

**Deliverables**:
- `src/rag/ingestion.py`: PDF parser + chunker (handles ../files/{company_id}/{quarter}/... structure)
- `src/rag/embeddings.py`: Embedding + FAISS ops
- `src/rag/retriever.py`: Query → top-K chunks
- Updated `main.py --index` command

### Part 3: Guardrails (2 hours) — TODO
- [ ] Scope validation (intent classification)
- [ ] Confidence thresholding
- [ ] Query normalization (ticker aliases)
- [ ] Rate limiting stub
- [ ] Enhanced error messages

**Deliverables**:
- `src/guardrails/validator.py`
- Updated prompts for validation
- Updated CLI with safeguards

## 8. Testing Strategy (Future)

### Unit Tests
```python
# test_llm_client.py
def test_retry_on_rate_limit()
def test_token_count_estimation()

# test_prompts.py
def test_retrieval_prompt_formatting()

# test_ingestion.py
def test_chunk_overlap()
def test_metadata_parsing()
```

### Integration Tests
```python
# test_integration.py
def test_end_to_end_query()  # Index → Query → Answer
def test_conversation_history_truncation()
```

### Evaluation Metrics
- **Retrieval**: Mean Reciprocal Rank (MRR), NDCG@5
- **LLM**: BLEU (answer similarity), citation accuracy
- **Latency**: P50, P99 query time
- **Cost**: $/query, tokens/query

## 9. Known Limitations & Future Work

### Current (Part 1)
- ❌ No actual retrieval (placeholder empty context)
- ❌ No scope checking (accepts any question)
- ❌ No confidence thresholding
- ❌ Single-user CLI only

### After Part 2
- ❌ No multi-turn context refinement (each query independent)
- ❌ No query rewriting (typos, aliases)
- ❌ No semantic cache

### After Part 3
- ❌ No fine-tuned embeddings
- ❌ No multi-language support
- ❌ No real-time transcript updates

## 10. Deployment Checklist

- [ ] API key rotation policy
- [ ] Rate limiting (to prevent abuse)
- [ ] Monitoring & alerting (LLM failures)
- [ ] Backup of FAISS index
- [ ] Audit log retention policy
- [ ] Load testing (concurrent queries)
- [ ] Security scanning (dependencies)

---

**Document Version**: 1.0 (Part 1 Complete)  
**Last Updated**: 2024-01-30  
**Author**: MLE Candidate
