# Earnings Call Q&A Chatbot

This project is a retrieval-augmented chatbot for asking questions about earnings-call transcripts. It supports exact metric extraction, transcript search, and follow-up questions that reuse recent conversation context when the company or quarter was already established.

## Features

- **FAISS-backed retrieval** over chunked transcript PDFs
- **Exact metric answers** for revenue, total income, PAT, EBITDA, margins, cash flow, capex, PBT, and ARPOB
- **Company and quarter filtering** via CLI flags
- **History-aware follow-ups** for vague questions like “what was the reason behind its growth?” while history is still present
- **Guardrails** for scope checks, confidence handling, and fallback responses
- **Mock LLM mode** for fast local development without API calls
- **Audit logging** of questions and answers to JSONL

## Supported Companies

The current dataset and index are organized by company ID:

- `532400` - Birlasoft
- `542652` - Polycab
- `543654` - Medanta / Global Health
- `544350` - Dr. Agarwal's Eye Hospital

## Project Structure

```
earnings-qa/
├── main.py
├── requirements.txt
├── README.md
├── WALKTHROUGH.md
├── DESIGN.md
├── IMPLEMENTATION.md
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

## Setup

```bash
cd earnings-qa
pip install -r requirements.txt
cp .env.example .env
```

For local development, set:

```env
USE_MOCK_LLM=true
```

If you want a real LLM backend, set the appropriate API key or Ollama settings in `.env`.

## Build the Index

The chatbot reads transcripts from `../files/` and stores the searchable index in `rag_index/`.

```bash
python main.py --index
python main.py --list-docs
```

`--index` creates or refreshes the FAISS index. `--list-docs` prints the indexed company/quarter chunks so you can confirm the data that is actually loaded.

## Run Chat

```bash
python main.py
python main.py --company 543654
python main.py --company 542652 --quarter Q3
```

Inside chat you can type:

- `clear` to reset conversation history
- `quit` or `exit` to end the session

## How Answers Work

The CLI first tries deterministic extraction for exact finance questions. If the metric is clear, it answers directly from the transcript text. If not, it retrieves relevant chunks and asks the LLM to answer from that context.

Important behavior:

- Exact figures are preferred over summary language.
- Questions outside the active company or quarter filters are rejected.
- Vague growth questions can use recent conversation history only while that history still exists.
- After `clear`, the app no longer has prior context and will ask you to specify the company and quarter again.

## Development Notes

The most important source files are:

- [src/cli/interface.py](src/cli/interface.py) for chat flow, direct answers, filters, and history handling
- [src/rag/retriever.py](src/rag/retriever.py) for retrieval and company/quarter intent detection
- [src/guardrails/validator.py](src/guardrails/validator.py) for scope checks and response validation
- [src/llm/prompts.py](src/llm/prompts.py) for system and retrieval prompts

## Notes on Data and Caching

- The index is loaded from `rag_index/` if it already exists.
- Different environments can answer differently if they are using different cached indexes.
- The transcript source files live under `../files/` and are not embedded in the index until you run `--index`.

## Future Work

- Add automated tests for exact-answer ranking and history-aware follow-ups
- Add a startup summary of the loaded index contents
- Improve query normalization for more company aliases and shorthand references
