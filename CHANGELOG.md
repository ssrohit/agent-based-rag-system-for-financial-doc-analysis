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

## Level 4 — Smart Generation / Query Understanding

- Closed Level 3's explicitly-deferred limitation: retrieval used to search the entire Chroma
  collection regardless of which company a question was about. Now a chat question is first run
  through a new query-time entity extraction step — a new Langfuse-managed prompt
  (`dev/query-entity-extraction`) plus `llm_service.ainvoke_structured` against a new
  `QueryEntities` schema — reusing the same structured-output extraction pattern
  `SymbolExtraction` already used at ingestion, but tuned to tolerate questions that name no
  company at all (returns an empty ticker list rather than guessing).
- Extracted ticker(s) scope both retrieval signals: the Chroma vector search via a `where`
  filter (`hybrid_retrieval.build_ticker_filter`, `$eq`/`$in`), and the BM25 candidate pool via a
  small ephemeral index rebuilt from just the matching subset
  (`hybrid_retrieval.filter_documents_by_tickers`) — `BM25Okapi` has no query-time filtering of
  its own, so its fixed corpus is filtered before indexing rather than after searching.
  `RetrievalService.retrieve()` gained an optional `tickers` parameter; passing none preserves
  Level 3's unscoped behavior exactly.
- Deliberately **ticker-only filtering, not date-range**: the only date-like ingestion metadata
  is `filing_date` (when a filing was *submitted*, not the fiscal period it reports on), so
  filtering on it would silently produce misleading results for questions like "Apple's 2023
  revenue". Deferred until fiscal-period metadata is actually captured at ingestion.
- New explicit short-circuit in `chat_service.process_user_message`: when a question names a
  company that has no ingested chunks, the response now says so by name (e.g. "I don't have any
  ingested filings for TSLA yet") instead of the generic Level 2 fallback or an unfiltered answer
  that might cite an unrelated company.
- Ticker case is now normalized to uppercase on both the ingestion side
  (`ingestion_service._ticker_from_downloaded_file`) and the query side
  (`build_ticker_filter`/`filter_documents_by_tickers`), so `$eq`/`$in` filter matching doesn't
  silently miss chunks over a case mismatch.
- New unit tests for `build_ticker_filter` and `filter_documents_by_tickers` in
  `tests/services/test_hybrid_retrieval.py`, following the same fixture-`Document`,
  no-real-models approach as the existing hybrid-retrieval tests.
- End-to-end testing against the real store surfaced a data gap this level's filtering exposed
  rather than caused: the 598 Apple/ServiceNow chunks ingested in Level 1/2 predate Level 3's
  `ticker` metadata entirely, so ticker-scoped questions about them incorrectly hit the new
  "not ingested" short-circuit. Fixed with a one-off `scripts/backfill_ticker_metadata.py` that
  derives each untagged chunk's ticker from its existing `source` metadata (same filename
  convention `_ticker_from_downloaded_file` parses at ingestion) and writes it back onto the
  chunk's Chroma metadata in place — re-ingestion wasn't needed. Verified afterward: single- and
  multi-company questions about Apple/Microsoft/ServiceNow all retrieve correctly-scoped chunks.
- Rewrote the HTML-table-to-Markdown conversion in `SecEdgarAdvancedLoader` from a `markdownify`-based
  pass to a dedicated grid model, replacing the dependency entirely (dropped from `pyproject.toml`):
  - `table_to_grid` flattens a `<table>` into a rectangular cell grid, resolving `colspan`/`rowspan`
    itself instead of relying on `markdownify`'s document-order text extraction, which split a
    rowspan'd label away from every row after its first. `_collapse_redundant_columns` then drops
    always-empty spacer columns and merges columns that are identical in every row — both lossless
    cleanups of the pixel-alignment artifacts common in EDGAR's raw table HTML.
  - `is_data_table` (replacing the old `_is_data_table`) classifies a table from the resolved grid
    rather than raw digit-density of the table's full text: numeric-cell fraction *excluding header
    rows*, plus a financial-keyword check restricted to the actual header row(s) (via the new
    `_header_row_count`, which prefers `<thead>`/`<th>` signals over content sniffing so a
    purely-numeric header row like "2023 2022" isn't mistaken for a data row).
  - `grid_to_markdown_chunks` renders a table as one or more self-contained Markdown chunks, each
    under the chunker's `chunk_size`, repeating the merged header in every chunk and never splitting
    a data row — so a table too large for one chunk still leaves every chunk readable on its own
    instead of bare numbers with no column labels.
  - New `split_document_preserving_tables` replaces the old approach of handing
    `RecursiveCharacterTextSplitter` the whole document with `[START_TABLE]`/`[END_TABLE]` as split
    separators, which had a real bug: when a table plus its surrounding prose together exceeded
    `chunk_size`, the splitter could use `[END_TABLE]` as a separator against the *following* text,
    detaching it from its own table even though the table alone was well under `chunk_size`. The new
    function pulls out each already-pre-sized table span and emits it as its own chunk verbatim,
    running only the plain-text stretches around it through the recursive splitter — table chunks are
    tagged `metadata["is_table"] = True`.
  - Also fixed silent data loss on non-UTF-8 bytes in downloaded filings: file reads switched from
    `errors="ignore"` (drops bad bytes) to `errors="replace"`.
  - No before/after retrieval-quality benchmark has been run yet for this change (Level 6 will add
    a RAGAS-based harness for that); this entry is a correctness/coverage fix, not a measured
    improvement. Covered by new unit tests in `tests/document_processor/test_parse_documents.py`
    (grid resolution, header detection, chunk splitting, table-span preservation) with no real
    model loads.
