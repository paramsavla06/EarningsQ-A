# Earnings Call Q&A Chatbot

This project answers questions over earnings-call transcripts with a hybrid design: exact finance figures are extracted deterministically first, then the retriever and LLM fill in context, reasoning, and follow-up questions. The result is a small but practical RAG pipeline that aims to be accurate on factual queries, explicit about provenance, and easy to run locally.

## What It Does

- Answers questions about company earnings calls, quarter by quarter
- Extracts exact figures for revenue, total income, PAT, EBITDA, margins, cash flow, capex, PBT, and ARPOB before falling back to generation
- Uses company and quarter filters when you want a narrower search space
- Reuses recent conversation context for follow-up questions while that context is still available
- Rejects vague or off-topic prompts instead of guessing

## Architecture

```text
User question
	-> scope / guardrail check
	-> exact metric extraction for finance questions
	    ├─ match found -> direct answer
	    └─ no match -> conversation history lookup
	        -> retrieval from FAISS index
	        -> context assembly
	        -> LLM response generation
	        -> response validation + audit log
```

The pipeline is intentionally split so that precise questions do not depend on free-form generation. That reduces hallucination risk and makes exact-answer queries much more reliable.

## Supported Companies

The current dataset is organized by company ID, with the company context shown below.

- `532400` - Birlasoft, an IT services company
- `542652` - Polycab, a cable and wire manufacturer
- `543654` - Medanta / Global Health, a hospital and healthcare operator
- `544350` - Dr. Agarwal's Eye Hospital, an eye-care hospital chain

## Project Layout

```text
earnings-qa/
├── main.py
├── requirements.txt
├── README.md
├── WALKTHROUGH.md
├── src/
│   ├── config.py
│   ├── cli/
│   │   └── interface.py
│   ├── llm/
│   │   ├── client.py
│   │   └── prompts.py
│   ├── rag/
│   │   ├── ingestion.py
│   │   ├── embeddings.py
│   │   └── retriever.py
│   └── guardrails/
│       └── validator.py
├── logs/
├── rag_index/
└── ../files/
```

## Setup And Configuration

```bash
cd earnings-qa
pip install -r requirements.txt
cp .env.example .env
```

### Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `USE_MOCK_LLM` | `false` | Uses a mock local LLM client for development and testing without API calls |
| `USE_OLLAMA` | `false` | Enables Ollama for both embeddings and generation unless overridden |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_LLM_MODEL` | `llama3.2:latest` | Ollama chat model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `USE_OLLAMA_EMBED` | `USE_OLLAMA` | Overrides Ollama use for embeddings only |
| `USE_OLLAMA_LLM` | `USE_OLLAMA` | Overrides Ollama use for the chat model only |
| `GEMINI_API_KEY` | unset | Required for Gemini-based embeddings and LLM calls when not using mock or Ollama |
| `LLM_MODEL` | `gemini-2.0-flash` | Gemini chat model |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Gemini embedding model |
| `MAX_CONVERSATION_HISTORY` | `10` | Number of exchanges kept in memory for follow-up questions |
| `CHUNK_SIZE` | `500` | Transcript chunk size in characters |
| `CHUNK_OVERLAP` | `50` | Character overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Number of retrieved chunks returned to the LLM |
| `LLM_TEMPERATURE` | `0.3` | Lower temperature for more factual answers |
| `LLM_MAX_TOKENS` | `512` | Maximum output tokens for the LLM |

If neither `USE_MOCK_LLM` nor Ollama is enabled, you must provide `GEMINI_API_KEY`.

> Note: the transcript PDFs are read from a `files/` directory one level above the project root. That matches the current repository layout. If you move the data elsewhere, update `DATA_DIR` in [src/config.py](src/config.py).

## Build The Index

The chatbot reads transcript PDFs from `../files/` and stores the searchable FAISS index in `rag_index/`.

```bash
python main.py --index
python main.py --list-docs
```

`--index` ingests PDFs, chunks them, generates embeddings, and persists the local index. `--list-docs` shows the company and quarter chunks that are actually loaded, which is useful for confirming that the right dataset is available.

Chunking uses `CHUNK_SIZE=500` and `CHUNK_OVERLAP=50`, which keeps each passage small enough for targeted retrieval while preserving some boundary context between chunks.

## Run The Chatbot

```bash
python main.py
python main.py --company 543654
python main.py --company 542652 --quarter Q3
```

Inside chat you can type:

- `clear` to reset conversation history
- `quit` or `exit` to end the session

## How Answers Work

For exact finance questions, the CLI first tries deterministic extraction from the transcript text. If it can confidently identify the number, it returns that directly instead of asking the model to invent a response.

If the question is broader, the retriever pulls relevant transcript chunks from the FAISS index, the prompt includes recent conversation context, and the LLM explains the answer using only the provided excerpts. That makes the system better at both precision queries and follow-up questions.

Key behavior:

- Exact figures are preferred over paraphrases
- Company and quarter filters narrow retrieval before generation
- Vague follow-ups can reuse recent conversation history until `clear` is used
- If the app cannot infer the company or quarter safely, it asks for clarification instead of guessing

## Design Decisions

- **Deterministic extraction before RAG**: exact financial figures are best handled with regex-based extraction so the system does not rely on the LLM for simple factual answers.
- **FAISS for local retrieval**: a local vector index keeps setup simple and avoids requiring a hosted vector database for the assignment.
- **Transcript-aware chunking**: chunks are derived from the source transcripts so citations stay tied to the original company and quarter.
- **Layered guardrails**: scope checks, history-aware follow-up handling, and response validation reduce the chance of off-topic or cross-company answers.

## Validation

For the current snapshot, validation is documented in [TESTING.md](TESTING.md). The recommended checks are:

```bash
python check_imports.py
python main.py --index
python main.py --list-docs
python main.py
```

Use these checks to confirm import health, index creation, retrieval coverage, and the chat loop.

## Scalability Notes

The current setup is optimized for a local assignment workflow, not a multi-tenant production service. For hundreds of companies and thousands of users, the retriever layer can be swapped to a managed vector store, the CLI can be exposed behind a FastAPI service, and retrieval / generation can be separated into asynchronous workers with caching around embeddings and repeated transcript lookups.

## Limitations

- Conversation history is in-memory only and resets when `clear` is used or the process restarts
- The index is local, so different environments can diverge if they are built from different transcript sets

## Code Organisation

- [src/cli/interface.py](src/cli/interface.py) — chat loop, direct-answer routing, filter handling, history management
- [src/rag/ingestion.py](src/rag/ingestion.py) — PDF parsing, chunking, and transcript metadata tagging
- [src/rag/embeddings.py](src/rag/embeddings.py) — embedding generation, FAISS index build, and persistence
- [src/rag/retriever.py](src/rag/retriever.py) — query-time retrieval, company/quarter filtering, and context assembly
- [src/guardrails/validator.py](src/guardrails/validator.py) — scope enforcement, confidence scoring, and response validation
- [src/llm/client.py](src/llm/client.py) — Gemini, Ollama, and mock LLM backends with retry logic
- [src/llm/prompts.py](src/llm/prompts.py) — system prompt and retrieval prompt templates

