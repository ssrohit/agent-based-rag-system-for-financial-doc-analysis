# Agent-Based RAG System for Financial Document Analysis

An agent-based Retrieval-Augmented Generation (RAG) system that analyzes SEC filings (10-K,
earnings reports, annual reports) to answer investment-analysis questions with grounded,
source-cited answers. Given a free-text request, it identifies the relevant company and date
range, downloads the filings from SEC EDGAR, parses and embeds them, and answers follow-up
questions against that knowledge base.

## Architecture

FastAPI app (`main.py`) → routers (`src/routes/`) → services (`src/services/`) → core/document
processing (`src/core/`, `src/document_processor/`). See `AGENTS.md` for the detailed
file-by-file architecture notes (loader internals, LLM service resolution logic, singleton usage,
etc.) — this section stays high level.

- **Ingestion** (`POST /ingest`): free-text message → LLM extracts ticker + date range →
  `sec-edgar-downloader` fetches filings → a custom loader converts filing HTML (including
  financial tables) to clean, table-boundary-aware Markdown chunks → embedded
  (`sentence-transformers/all-mpnet-base-v2`) into a persistent Chroma vector store.
- **Chat** (`POST /chat/user-msg`): free-text question → LLM extracts which company (if any) the
  question is about → hybrid retrieval (vector similarity + BM25 keyword search, both scoped to
  that company's chunks when one was identified, fused via Reciprocal Rank Fusion, reordered by
  a local cross-encoder reranker) over the Chroma store → context-grounded prompt → LLM answer
  with key metrics/caveats → response includes the source filings the answer was drawn from,
  each with a relevance score.
- **Agent** (`POST /agent/ask`): a LangGraph-based alternative to `/chat/user-msg` for multi-part
  questions. A bounded ReAct-style loop plans and calls tools — document search, a dedicated
  company-comparison search, and a safe (no `eval()`) calculator — then drafts an answer and runs
  it through an LLM-judged reflection step (grounded/complete) that can send the loop back for one
  more evidence-gathering round before finalizing. Both loops are explicitly bounded (max tool
  rounds, max reflection retries), so a hard-to-answer question degrades to a best-effort answer
  with an honest caveat instead of hanging or erroring. Ships alongside `/chat/user-msg`, not as a
  replacement for it.
- **Observability**: every service entrypoint is wrapped in a Langfuse `@observe()` trace, and all
  LLM calls go through a single `LLMService` that auto-attaches Langfuse callbacks. Prompts are
  managed in Langfuse (not hardcoded), e.g. `dev/symbol-extractor`, `dev/financial-qa`,
  `dev/agent-system`, `dev/answer-grader`.

## Getting started

Requires Python >= 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
# create src/configs/.env with the required settings (see src/config.py for the full list)
uv run main.py
```

Then, e.g.:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"userMessage": "Get Apple 10-K filing from 2023"}'

curl -X POST http://127.0.0.1:8000/chat/user-msg \
  -H "Content-Type: application/json" \
  -d '{"userMessage": "What was Apple'\''s total revenue and how did it change year over year?"}'

curl -X POST http://127.0.0.1:8000/agent/ask \
  -H "Content-Type: application/json" \
  -d '{"userMessage": "Compare Apple and Microsoft'\''s revenue growth"}'
```

## Progress

This project is being built incrementally, one capability ("level") at a time — see
`CHANGELOG.md` for what each level actually added.

- [x] **Level 1 — Ingestion Pipeline**: SEC EDGAR download, table-aware document parsing,
      chunking, embedding into Chroma.
- [x] **Level 2 — Basic RAG QA Chain**: grounded retrieval + generation over ingested filings,
      with cited sources.
- [x] **Level 3 — Hybrid Retrieval + Reranking**: BM25 + vector ensemble retrieval, local
      cross-encoder reranking.
- [x] **Level 4 — Smart Generation / Query Understanding**: entity extraction from questions,
      ticker-scoped metadata-filtered retrieval.
- [x] **Level 5 — Agentic Multi-Hop Reasoning**: LangGraph ReAct-style agent with tools (document
      search, company comparison, calculator) and a bounded reflection/self-verification step.
- [ ] **Level 6 — Evaluation**: RAGAS-based evaluation harness, benchmarked across levels 2-5.
- [ ] **Level 7 — Frontend**: chat interface with a sources panel (framework TBD).
- [ ] **Level 8 — Deployment**: containerization and a hosted demo.
