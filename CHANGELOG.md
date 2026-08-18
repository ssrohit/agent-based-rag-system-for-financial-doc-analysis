# Changelog

Progress log for the Agent-Based Financial RAG System, organized by project "level" (see
`README.md` for the full roadmap). Each entry documents what was added and what it teaches/demonstrates,
so the history doubles as a record of how the system evolved.

## Level 1 — Ingestion Pipeline

- Set up the FastAPI + LangChain + ChromaDB + Langfuse foundation.
- Built an LLM-driven extraction step (`SymbolExtraction`) that pulls a stock ticker and date
  range out of a free-text user message using a Langfuse-managed prompt (`dev/symbol-extractor`).
- Added automated SEC EDGAR filing downloads (`SecFilingsDownloader`), including flattening the
  library's nested download layout into a consistent `{TICKER}_{FORM_TYPE}_{ACCESSION}_{filename}`
  naming scheme.
- Built a custom SEC-EDGAR-aware document loader (`SecEdgarAdvancedLoader`) that extracts specific
  filing types (e.g. 10-K) from the raw submission file, converts financial HTML tables to
  Markdown (instead of dropping table structure), and flattens layout-only tables.
- Implemented table-boundary-aware chunking and embedding (`sentence-transformers/all-mpnet-base-v2`)
  into a persistent Chroma vector store.
- Centralized all LLM calls behind a single `LLMService` with automatic Langfuse tracing and
  structured-output support.

## Level 2 — Basic RAG QA Chain

- Implemented `chat_service.py`: given a free-text question, retrieve the top-k relevant chunks
  from the Chroma collection populated in Level 1, ground an LLM answer in that context via a new
  Langfuse-managed prompt (`dev/financial-qa`), and return a structured answer (direct answer,
  key metrics, caveats) plus the source filings it drew from.
- Added `RetrievalService`, a dedicated read path over the vector store, kept separate from
  generation so Level 3's hybrid retrieval + reranking can swap in without touching the chat
  service.
- Citations (`SourceReference`) are built directly from retrieved-chunk metadata, never from the
  LLM, so sources can't be hallucinated.
- Handles the "nothing ingested yet" case explicitly instead of letting the LLM guess.
