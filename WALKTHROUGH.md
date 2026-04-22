# Earnings Call Q&A Chatbot - User Walkthrough

A step-by-step guide to using the earnings call chatbot with RAG and guardrails.

---

## Part 1: Initial Setup

### Step 1: Install Dependencies

```bash
cd earnings-qa
pip install -r requirements.txt
```

**What happens:**
- Downloads and installs all required packages
- Sets up Google Gemini API client
- Installs FAISS for vector search
- Installs PDF parsing and data processing libraries

### Step 2: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` file:
```env
# For development (free, no API calls):
USE_MOCK_LLM=true
GEMINI_API_KEY=AIzaSy...  # Optional if using mock mode

# For production (with real Gemini API):
USE_MOCK_LLM=false
GEMINI_API_KEY=your_actual_api_key_here
```

**Mock Mode Benefits:**
- ✅ No API quota limits
- ✅ Instant responses (testing)
- ✅ Free to develop
- ✅ Responses marked as `[MOCK]` for clarity

---

## Part 2: Create RAG Index

### What is an Index?

An **index** is a searchable database of all earnings call transcripts. It enables fast, intelligent queries about company earnings.

### Step 3: Index Transcripts

```bash
python main.py --index
```

**Console Output:**
```
📑 Indexing earnings call transcripts...
✓ Ingested 1,250 document chunks
✓ Index created successfully
```

**What's happening behind the scenes:**

1. **Ingestion**: Reads all PDFs from `../files/{company}/{quarter}/`
2. **Chunking**: Splits each transcript into ~500-char pieces (with overlap for context)
3. **Embedding**: Converts each chunk into a 768-dimensional vector using Gemini
4. **Indexing**: Stores vectors in FAISS for fast similarity search
5. **Persistence**: Saves to disk (`rag_index/`) for reuse

**Time estimate:**
- Mock mode: ~30 seconds
- Real API mode: ~5 minutes (rate-limited)

### Step 4: Verify Index

```bash
python main.py --list-docs
```

**Expected Output:**
```
📄 Indexed Documents:

  532400 Q1: 287 chunks
  532400 Q2: 295 chunks
  542652 Q1: 234 chunks
  542652 Q2: 241 chunks
  543654 Q1: 256 chunks
  543654 Q2: 248 chunks
  544350 Q1: 219 chunks
  544350 Q2: 211 chunks

Total: 1,891 chunks indexed
```

This confirms:
- ✅ All 4 companies indexed
- ✅ Multiple quarters per company
- ✅ Chunks created from PDFs
- ✅ Index is ready for queries

---

## Part 3: Chat with the Bot

### Step 5: Start Interactive Chat

```bash
python main.py
```

**Console Output:**
```
============================================================
Earnings Call Q&A Chatbot
============================================================
Type 'quit' or 'exit' to end conversation
Type 'clear' to reset conversation history
============================================================

You: 
```

### Step 6: Ask Your First Question

```
You: What was the revenue in Q1?
```

**What happens:**
1. **Scope Check**: Is this question about earnings? ✅ YES
2. **Retrieval**: Find top-5 most similar chunks
3. **LLM Answer**: Generate response based on context
4. **Confidence**: Evaluate match quality
5. **Output**: Display answer with confidence metadata

**Example Response (Mock Mode):**
```
Assistant: [MOCK] In Q1 2024, revenue was $119.6 billion, representing 
a 2% year-over-year increase driven by strong product demand and 
geographic expansion initiatives.

⚠️ Note: This response is based on general knowledge about earnings 
calls, as no specific context was found in the transcript database.

You:
```

---

## Part 4: Understanding Responses

### Confidence Levels

Responses include confidence metadata:

#### HIGH Confidence (≥70%)
```
📊 Confidence: 85% - Strong matching context found.
```
- Meaning: Multiple strong matches in transcripts
- Action: Trust the answer ✅

#### MEDIUM Confidence (40%-70%)
```
📊 Confidence: 55% - Moderate matching context found.
```
- Meaning: Some relevant context but not perfect matches
- Action: Answer is reasonable but may be incomplete

#### LOW Confidence (<40%)
```
⚠️ Note: This response is based on general knowledge about earnings 
calls, as no specific context was found in the transcript database.
```
- Meaning: No matching context in transcripts
- Action: Use with caution; may need more specific question

### Understanding Scope Validation

**In-Scope Questions:**
```
You: What was the gross margin improvement?
Assistant: [Response about margins from transcripts]
```

**Out-of-Scope Questions:**
```
You: Tell me a joke about earnings
Assistant: Your question contains 'joke', which is outside the scope 
of earnings call analysis. Please ask about earnings, revenue, 
margins, guidance, or other financial topics.

You:
```

---

## Part 5: Query Filtering

### Filter by Company

```bash
python main.py --company 532400
```

Now queries only use company 532400 data:

```
You: What were the margins?
Assistant: [Only uses 532400 data]

📊 Company Filter: 532400
```

**Available Companies:**
- `532400` - Apple
- `542652` - Microsoft
- `543654` - Google
- `544350` - Meta

### Filter by Quarter

```bash
python main.py --quarter Q1
```

Filters to Q1 only:

```
You: What was guidance?
Assistant: [Only uses Q1 2024 data]

📊 Quarter Filter: Q1
```

### Combine Filters

```bash
python main.py --company 532400 --quarter Q2
```

Most specific queries:

```
You: What was the revenue outlook?
Assistant: [Only uses Apple Q2 2024 data]

📊 Company Filter: 532400
📊 Quarter Filter: Q2
```

---

## Part 6: Example Conversation

### Scenario: Analyst Researching Apple Q1 Performance

**Setup:**
```bash
python main.py --company 532400 --quarter Q1
```

**Conversation:**

```
============================================================
Earnings Call Q&A Chatbot
============================================================
Filters active: Company: 532400, Quarter: Q1
Type 'quit' or 'exit' to end conversation
============================================================

You: What was the total revenue?
Assistant: In Q1 2024, Apple reported total revenue of $119.6 billion,
representing a 2% increase compared to Q1 2023. This growth was driven
by strong iPhone sales and services revenue expansion.

📊 Confidence: 87% - Strong matching context found.

You: What about gross margin?
Assistant: Gross margin improved by 150 basis points to 46.2% in Q1,
benefiting from better product mix and operational efficiencies.

📊 Confidence: 82% - Strong matching context found.

You: What's the guidance for Q2?
Assistant: Management guided for Q2 revenue between $91 billion and
$95 billion, representing 3-7% growth relative to Q2 2023. Gross
margin is expected to remain stable at 45-46%.

📊 Confidence: 89% - Strong matching context found.

You: How is the China market performing?
Assistant: China remains a strategic market with strong momentum,
particularly in iPhone and services. The company highlighted continued
investment in localized offerings and regulatory engagement in the
region.

📊 Confidence: 71% - Moderate matching context found.

You: What's your opinion on the stock?
Assistant: Your question contains 'opinion', which requires subjective
judgment beyond earnings call data. Please ask about specific metrics
like revenue, margins, guidance, segments, or market performance that
are discussed in the earnings calls.

You: clear
Conversation history cleared.

You: quit
Goodbye!
```

---

## Part 7: Understanding Context Retrieval

### How Retrieval Works

When you ask a question:

1. **Query Embedding** (0.1s)
   - Your question is converted to a 768-dim vector
   - Same embedding model as transcripts

2. **FAISS Search** (0.01s)
   - Finds 10 most similar transcript chunks
   - Ranks by L2 distance (similarity score)
   - Converts to 0-100% confidence

3. **Filtering** (0.01s)
   - If `--company` specified: keeps only that company
   - If `--quarter` specified: keeps only that quarter
   - Takes top-5 remaining results

4. **Context Formatting** (0.01s)
   - Combines chunks into readable context
   - Adds metadata (company, quarter, match %)
   - Truncates to 3000 chars for LLM

5. **LLM Generation** (1-3s)
   - Sends system prompt + context + query
   - LLM generates answer based only on context
   - (Mock mode: instant responses)

**Total latency:** ~1-5 seconds (depending on LLM)

### Viewing Retrieved Context

To debug what context was used:

1. Check `logs/queries.jsonl` for full query logs
2. Responses start with retrieved context sections marked as:
   ```
   [532400 Q1 2024 - Match: 87%]
   Revenue in Q1 was...
   ```

---

## Part 8: Tips & Tricks

### Tip 1: Use Specific Keywords

**Better:**
```
You: What was the gross margin in Q1?
```

**Worse:**
```
You: How was the performance?
```

Specific keywords improve relevance matching.

### Tip 2: Ask About Specific Metrics

**Good questions:**
- "What was revenue?" ✅
- "How did margins change?" ✅
- "What's the guidance?" ✅
- "What about the China market?" ✅

**Avoid:**
- Subjective opinions ❌
- Future speculation ❌
- Out-of-scope topics ❌

### Tip 3: Multi-Turn Conversations

Context is preserved across questions:

```
You: What was the revenue?
Assistant: Revenue was $119.6B...

You: How much growth is that?
Assistant: [Remembers previous question, builds on it]
```

To clear history:
```
You: clear
Conversation history cleared.
```

### Tip 4: Use Filters for Precision

- **Researching one company?** → Use `--company`
- **Comparing quarters?** → Run separate sessions
- **Need all data?** → No filters

---

## Part 9: Common Scenarios

### Scenario A: Comparing Two Companies

**Terminal 1:**
```bash
python main.py --company 532400
You: What was revenue?
Assistant: Apple Q1 revenue was $119.6B...
```

**Terminal 2 (new):**
```bash
python main.py --company 542652
You: What was revenue?
Assistant: Microsoft Q1 revenue was $61.2B...
```

Compare side-by-side.

### Scenario B: Tracking Quarterly Trends

```bash
python main.py --company 532400 --quarter Q1
You: What was the revenue?
Assistant: Q1 revenue: $119.6B

You: clear

# Manually switch quarter (restart with --quarter Q2)
```

### Scenario C: Debugging Low Confidence

```
You: What is the CEV ratio?
Assistant: I couldn't find specific information about CEV ratio in...

📊 Confidence: 12% - Limited matching context found.
```

**Why low confidence?**
- "CEV ratio" may not be discussed in earnings calls
- Try: "What was the valuation multiple?"

---

## Part 10: Troubleshooting

### Issue: "No RAG index loaded"

```bash
python main.py --index
# Wait for indexing to complete
python main.py
```

### Issue: "RESOURCE_EXHAUSTED" (Real Mode Only)

Means Gemini API quota is exceeded.

**Solution:**
```bash
# Edit .env
USE_MOCK_LLM=true

# Restart
python main.py
```

Wait for quota reset or upgrade API plan.

### Issue: Low Confidence on All Queries

**Check:**
1. Is index created? → `python main.py --list-docs`
2. Are PDFs in `../files/`? → `ls ../files/`
3. Try more specific terms

### Issue: "PDF not found"

```
Error: Data directory not found: ../files
```

**Solution:**
```bash
# Verify structure
ls -la ../

# Should show:
# earnings-qa/
# files/
#   ├── 532400/
#   └── Q1/, Q2/, etc.
```

---

## Part 11: Architecture at a Glance

```
┌─────────────────────────────────────────────┐
│         YOUR QUESTION                       │
│    "What was Q1 revenue?"                   │
└────────────────┬────────────────────────────┘
                 │
         ┌───────▼────────┐
         │ Scope Check    │
         │ (Guardrail 1)  │
         │ In earnings?   │
         └───────┬────────┘
                 │ YES
        ┌────────▼────────┐
        │ Query Embedding │ ← Convert to vector
        └────────┬────────┘
                 │
        ┌────────▼────────────────┐
        │ FAISS Index Search      │
        │ Find 10 similar chunks  │
        └────────┬────────────────┘
                 │
        ┌────────▼────────┐
        │ Apply Filters   │
        │ (Company/Qtr)   │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │ Format Context  │
        │ Top-5 chunks    │
        └────────┬────────┘
                 │
        ┌────────▼──────────────┐
        │ LLM Generation         │
        │ (Gemini or Mock)       │
        └────────┬──────────────┘
                 │
        ┌────────▼────────────┐
        │ Guardrails Check    │
        │ (Confidence, Valid) │
        └────────┬────────────┘
                 │
    ┌────────────▼────────────────┐
    │ YOUR ANSWER                 │
    │ "[Q1 revenue was $119.6B]"  │
    │ "📊 Confidence: 87%"        │
    └─────────────────────────────┘
```

---

## Part 12: Next Steps

### For Development:
- Use mock mode for testing
- Explore different queries
- Check `logs/queries.jsonl` for patterns

### For Production:
- Switch to real Gemini API
- Monitor API usage and costs
- Cache indexes for faster startup
- Add database for persistence

### For Enhancement:
- Fine-tune embeddings for domain
- Add query expansion (synonyms)
- Implement semantic caching
- Build REST API with FastAPI
- Add multi-language support

---

## Part 13: Reference

**Key Files:**
- `main.py` - Entrypoint
- `src/config.py` - Configuration
- `src/cli/interface.py` - Chat interface
- `src/rag/ingestion.py` - PDF parsing
- `src/rag/embeddings.py` - Embedding generation
- `src/rag/retriever.py` - Query & retrieval
- `src/guardrails/validator.py` - Safety checks

**Key Commands:**
```bash
python main.py --index              # Create index
python main.py --list-docs          # Show indexed documents
python main.py                       # Start chat
python main.py --company 532400     # Chat for company
python main.py --company X --quarter Q1  # Filtered chat
python check_imports.py             # Verify setup
```

**Config File (`.env`):**
```env
USE_MOCK_LLM=true/false             # Development vs. Production
GEMINI_API_KEY=...                  # API key (if real mode)
CHUNK_SIZE=500                      # Characters per chunk
CHUNK_OVERLAP=50                    # Overlap between chunks
TOP_K_RETRIEVAL=5                   # Results per query
```

---

## Enjoy! 🚀

You're now ready to query earnings calls with confidence.

For more details, see:
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Technical deep dive
- [TESTING.md](TESTING.md) - Testing & validation
- [DESIGN.md](DESIGN.md) - Architecture decisions
