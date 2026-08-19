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

## Level 3 — Hybrid Retrieval + Reranking

- Replaced single-signal vector similarity search with a hand-rolled hybrid retrieval pipeline
  in `RetrievalService`: `rank_bm25.BM25Okapi` keyword search runs alongside the existing Chroma
  vector search, the two candidate pools are fused with a custom Reciprocal Rank Fusion
  implementation (not LangChain's `EnsembleRetriever`), and the fused pool is reordered by a
  local `sentence-transformers` cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) before the
  final top-k is returned. Chosen deliberately over LangChain's retriever abstractions to keep
  every fusion/reranking step visible and understood, and to avoid an undeclared
  `langchain-classic` dependency.
- New `src/services/hybrid_retrieval.py`: the pure scoring/fusion/selection logic (tokenization,
  BM25 helpers, RRF, rerank-based top-k selection), deliberately free of any Chroma/embedding/
  cross-encoder imports so it stays fast and unit-testable.
- BM25 has no incremental-update API, so its in-memory index is rebuilt wholesale via a new
  `RetrievalService.refresh()` — called once at startup and again after every `/ingest` call, so
  newly-ingested filings are searchable without a server restart.
- `SourceReference` gained `relevance_score`, populated from the reranker at retrieval time
  (never LLM-generated), so citations now carry a measure of *why* a chunk was returned —
  preserving the anti-hallucination citation guarantee from Level 2.
- `ticker` is now threaded from the downloaded filing's own filename (not the raw, possibly
  multi-symbol LLM extraction) through `DocumentProcessor`/`SecEdgarAdvancedLoader` into every
  chunk's Chroma metadata — laying groundwork for Level 4's metadata-filtered retrieval without
  adding query-time filtering yet.
- First unit tests in the repo: `pytest` added as a dev dependency, with tests for the BM25/RRF/
  rerank-selection logic in `tests/services/test_hybrid_retrieval.py`, using fixture `Document`s
  only — no real model loads, kept fast by the `hybrid_retrieval.py` module split above.
