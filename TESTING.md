# Testing & Validation Guide

## Quick Validation

### 1. Check All Imports
```bash
python check_imports.py
```

Expected output:
```
✓ src.config
✓ src.llm.client
✓ src.llm.prompts
✓ src.rag.ingestion
✓ src.rag.embeddings
✓ src.rag.retriever
✓ src.guardrails.validator
✓ src.cli.interface

✅ All imports successful!
Mock mode: True
Data directory: ../files
Data directory exists: True
```

---

## Testing Phases

### Phase 1: Configuration & Setup
```bash
# Verify environment configuration
cat .env

# Check that DATA_DIR points to existing files
ls -la ../files/

# You should see company folders:
# 532400, 542652, 543654, 544350
```

### Phase 2: RAG Indexing

#### Step 2a: Create Index
```bash
python main.py --index
```

Expected output:
```
📑 Indexing earnings call transcripts...
✓ Ingested 1,250 document chunks
✓ Index created successfully
```

This creates `rag_index/` with:
- `index.faiss` - FAISS vector index
- `embeddings.npy` - Embedding vectors
- `documents.pkl` - Document metadata

#### Step 2b: List Documents
```bash
python main.py --list-docs
```

Expected output:
```
📄 Indexed Documents:

  532400 Q1: 287 chunks
  532400 Q2: 295 chunks
  542652 Q1: 234 chunks
  ...
  
Total: 1,057 chunks indexed
```

---

### Phase 3: Chat with Guardrails

#### Step 3a: Basic Chat (No Filters)
```bash
python main.py

You: What was the revenue in Q1?
Assistant: [Retrieved context + LLM answer]

You: Tell me a joke
Assistant: Your question contains 'joke', which is outside the scope of earnings call analysis.
```

#### Step 3b: Filtered Chat
```bash
python main.py --company 532400 --quarter Q1

You: What was revenue?
Assistant: [Only Q1 2024 company 532400 context used]

You: What about Q2?
Assistant: ⚠️ No relevant documents found (filtered to Q1 only)
```

---

## Monitoring & Debugging

### Check Audit Logs
```bash
# Show latest queries
tail -f logs/queries.jsonl

# Parse individual entries
jq . logs/queries.jsonl
```

Expected log format:
```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "query": "What was the revenue?",
  "answer": "Revenue was...",
  "company_filter": null,
  "quarter_filter": null
}
```

### View Mock Responses
When `USE_MOCK_LLM=true`, responses are marked with `[MOCK]`:

```
You: What was revenue?
Assistant: [MOCK] Revenue in Q1 was $119.6B, representing 2% YoY growth...
```

### Switch to Real LLM (Production)
```bash
# Edit .env
USE_MOCK_LLM=false
GEMINI_API_KEY=your_api_key_here

# Restart chat
python main.py
```

---

## Common Issues & Solutions

### Issue 1: "No index loaded"
```bash
# Solution: Create index first
python main.py --index
```

### Issue 2: "Data directory not found"
```bash
# Solution: Verify DATA_DIR in config.py points to ../files
# Check that files/ folder exists in parent directory
ls -la ../files/
```

### Issue 3: PDF extraction errors
```bash
# Solution: Verify pdfplumber is installed
pip install pdfplumber>=0.9.0

# Check PDF files are readable
file ../files/532400/Q1/*.pdf
```

### Issue 4: FAISS import error
```bash
# Solution: Install faiss
pip install faiss-cpu>=1.7.4
```

### Issue 5: Low confidence scores
```bash
# This is normal for:
# - Out-of-scope questions (falls back to general knowledge)
# - Questions with no matching context
# - Filtered queries that don't match filters

# View confidence scores in responses:
# "📊 Confidence: 35% - Limited matching context found."
```

---

## Performance Metrics

### Typical Performance (Mock Mode, 1000+ chunks)
- **Indexing time**: ~30 seconds (embedding the chunks)
- **Query latency**: <100ms (embedding + FAISS search)
- **LLM response time**: <2 seconds (with mock mode)
- **Total round-trip**: ~2-3 seconds per query

### Typical Performance (Real LLM, Gemini API)
- **Indexing time**: ~5 minutes (API rate-limited)
- **Query latency**: ~1-2 seconds (API calls)
- **LLM response time**: ~3-5 seconds
- **Total round-trip**: ~5-7 seconds per query

---

## Architecture Validation

### Part 1: CLI & LLM ✅
- [x] Click CLI with multi-turn conversation
- [x] MockLLMClient for development
- [x] Real LLMClient with Gemini API
- [x] Conversation history management
- [x] Audit logging

### Part 2: RAG Pipeline ✅
- [x] PDF ingestion with pdfplumber
- [x] Smart chunking with overlaps
- [x] Embedding generation (Gemini API / mock)
- [x] FAISS index creation
- [x] Index persistence (save/load)
- [x] Retrieval with filtering
- [x] Context formatting for LLM

### Part 3: Guardrails ✅
- [x] Scope validation (keyword-based)
- [x] Confidence scoring (similarity-based)
- [x] Response validation
- [x] Confidence notes in output
- [x] Integration into chat loop

---

## Example Test Scenarios

### Scenario 1: In-Scope Query with High Confidence
```
Input: "What was Apple's gross margin in Q1?"
Expected:
  - Scope check: PASS (contains "margin" keyword)
  - Retrieval: TOP-5 results with >0.8 similarity
  - LLM: Generates answer from context
  - Confidence: HIGH (>0.7)
  - Output: Answer + "Confidence: 85%"
```

### Scenario 2: Out-of-Scope Query
```
Input: "Tell me a joke about earnings"
Expected:
  - Scope check: FAIL (contains "joke")
  - Retrieval: SKIPPED
  - LLM: SKIPPED
  - Output: "Your question contains 'joke', which is outside..."
```

### Scenario 3: Partially Matching Query
```
Input: "What was the guidance for next year?"
Expected:
  - Scope check: PASS (contains "guidance")
  - Retrieval: TOP-5 results with 0.4-0.7 similarity
  - LLM: Generates answer with caveats
  - Confidence: MEDIUM (0.4-0.7)
  - Output: Answer + "Confidence: 50% - Limited matching context found."
```

### Scenario 4: Filtered Query
```
Input: python main.py --company 532400
User: "What were margins?"
Expected:
  - Retrieval: Only documents with company_id="532400"
  - Context: [532400 Q1 2024], [532400 Q2 2024], etc.
  - No context from other companies
```

---

## Continuous Integration (Future)

For CI/CD integration, add tests like:
```bash
# Quick smoke test
python check_imports.py

# Index integrity check
python main.py --index
python main.py --list-docs

# Query validation
# (Mock a few test queries and verify responses)
```

---

## Next Steps for Production

1. **Database Integration**: Replace pickle with PostgreSQL for documents
2. **Vector Store Upgrade**: Move to Pinecone/Weaviate for scaling
3. **API Server**: Wrap CLI in FastAPI for service deployment
4. **Caching**: Add Redis for frequent query results
5. **Monitoring**: Integrate with observability tools (Prometheus, etc.)
6. **Fine-tuning**: Train custom embeddings for domain-specific matching
7. **Query Expansion**: Add synonym expansion and typo tolerance
8. **Multi-language**: Support queries in different languages
