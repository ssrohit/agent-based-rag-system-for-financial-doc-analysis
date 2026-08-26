# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Cursor, Aider, Codex, etc.) when working
with code in this repository.

## Overview

An agent-based RAG system for analyzing financial documents (SEC filings). Given a natural-language
user message, it extracts a stock ticker and date range via an LLM, downloads the relevant SEC filings
(10-K by default), parses/chunks/embeds them into a Chroma vector store, and answers follow-up questions
against that store with a grounded, source-cited answer. See `README.md` for the level-by-level roadmap
and `CHANGELOG.md` for what's shipped at each level.

## Commands

This project uses `uv` for dependency management (Python >= 3.12, see `.python-version`).

- Install/sync dependencies: `uv sync`
- Run the API server: `uv run main.py` (starts uvicorn on `127.0.0.1:8000` with `--reload`)
  - Alternative: `uv run uvicorn main:app --reload`
- Lint: `uv run ruff check .` (fix with `uv run ruff check --fix .`)
- Run tests: `uv run pytest`

### Required environment

`src/config.py` loads settings from `src/configs/.env` (gitignored, must be created manually — not
committed). Required variables: `CHAT_MODEL`, `GOOGLE_API_KEY`, `CHROMA_PERSIST_PATH`,
`VECTOR_GENERATION_MODEL`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_BASE_URL`, `HF_TOKEN`.
Import `settings` from `src.config` rather than instantiating `Settings()` elsewhere — it's a
module-level singleton and also pushes several of these values into `os.environ` on import (for
Langfuse, Google, HF libs that read env vars directly).

## Architecture

Layered FastAPI app: `main.py` → `src/routes/*` → `src/services/*` → `src/core/*` / `src/document_processor/*`.

- **`main.py`** — FastAPI app entrypoint; mounts routers and runs uvicorn.
- **`src/routes/`** — thin FastAPI routers (`chat.py` at `/chat`, `ingest.py` at `/ingest`,
  `agent.py` at `/agent`) that just delegate to a service function. Keep route handlers thin;
  business logic belongs in services.
- **`src/services/llm_service.py`** — `llm_service`, a `Singleton`-metaclass instance centralizing all
  LLM calls (via `langchain_google_genai.ChatGoogleGenerativeAI`). All LLM invocation should go through
  this, not ad-hoc `ChatGoogleGenerativeAI` instances, because it:
  - auto-attaches a Langfuse `CallbackHandler` to every call for tracing (`_get_callbacks`);
  - resolves the model name with precedence: explicit `model_name` arg → model embedded in a Langfuse
    prompt object's `config`/`metadata`/`model` attr → `settings.CHAT_MODEL` default, stripping any
    `provider:` prefix (e.g. `google_genai:gemini-2.5-flash` → `gemini-2.5-flash`);
  - exposes `ainvoke_structured(prompt, schema, ...)` for Pydantic-schema-constrained output,
    `ainvoke(prompt, ...)` for raw string output, and `get_chat_model(prompt, ...)` (Level 5) which
    returns the resolved, callback-attached `ChatGoogleGenerativeAI` itself for callers that need the
    raw model object (e.g. to `.bind_tools(...)` for an agent loop) rather than one of the narrower
    invoke helpers — `ainvoke`/`ainvoke_structured` are themselves now thin wrappers over it.
- **`src/services/ingestion_service.py`** — `ingest_data()`, the pipeline entrypoint. Flow: fetch a
  Langfuse-managed prompt (`dev/symbol-extractor`) → `llm_service.ainvoke_structured` to extract a
  `SymbolExtraction` (ticker(s) + date range) from the user's message → download filings via
  `SecFilingsDownloader` → for each downloaded file, recover that file's own ticker from its filename
  (`{TICKER}_{FORM_TYPE}_{ACCESSION}_{filename}`, via `_ticker_from_downloaded_file`, more accurate than
  the raw possibly-multi-symbol LLM extraction for multi-company ingestion runs, and uppercased so it
  matches the case-normalized filters query-time retrieval builds) and feed it through
  `DocumentProcessor.extract_data(..., ticker=...)` into the Chroma collection named by
  `src.constants.DEFAULT_COLLECTION_NAME` → finally calls `retrieval_service.refresh()` so the BM25
  index picks up the newly-ingested chunks without a server restart.
- **`src/services/hybrid_retrieval.py`** — pure, dependency-light retrieval-fusion logic (no Chroma/
  embedding/cross-encoder imports): `tokenize`, `build_bm25_index`/`bm25_top_n` (thin wrappers over
  `rank_bm25.BM25Okapi`), `reciprocal_rank_fusion` (dedups by `Document.id`, content-hash fallback),
  `select_top_k_by_score` (attaches a score into `metadata["rerank_score"]` on fresh `Document` copies,
  never mutates inputs), `build_ticker_filter` (turns a ticker list into a Chroma `where` filter,
  `$eq`/`$in`, uppercased), `filter_documents_by_tickers` (case-insensitive in-memory filter of a
  document list by `metadata["ticker"]`, used to scope BM25 the same way the Chroma filter scopes
  vector search). Split out from `retrieval_service.py` specifically so it's unit-testable
  (`tests/services/test_hybrid_retrieval.py`) without triggering `RetrievalService`'s module-level
  singleton construction, which eagerly loads real embedding/cross-encoder models.
- **`src/services/retrieval_service.py`** — `retrieval_service`, a `Singleton` hybrid-retrieval wrapper
  around the same Chroma collection (built once, not per-request, unlike `DocumentProcessor`). Also
  loads a local `sentence_transformers.CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU) and
  maintains an in-memory `rank_bm25.BM25Okapi` index. `retrieve(query, k=5, tickers=None)`: vector search
  (`asimilarity_search`, top 20, filtered via `build_ticker_filter(tickers)` when `tickers` is given)
  and BM25 search (top 20, via `_bm25_candidates` — filters the in-memory document list first with
  `filter_documents_by_tickers` and rebuilds a small ephemeral index when `tickers` is given, since
  `BM25Okapi` has no query-time filtering of its own) run as parallel candidate pools, fused via
  `reciprocal_rank_fusion` down to the top 15, reranked by the cross-encoder, and truncated to the
  final `k`. `tickers=None`/empty preserves the exact pre-Level-4 unscoped behavior. Since BM25 has no
  incremental-update API, `refresh()` rebuilds the whole index from `vector_store.get(...)`; called
  once at the end of `__init__` and again by `ingestion_service.ingest_data` after every ingestion.
- **`src/services/chat_service.py`** — `process_user_message()`, the query-side entrypoint, wired with
  `@observe()`. Flow: `_extract_query_entities()` fetches the Langfuse-managed prompt
  `dev/query-entity-extraction` and calls `llm_service.ainvoke_structured` against the `QueryEntities`
  schema to pull the company ticker(s) (if any) out of the question, uppercased — the same
  structured-output pattern `SymbolExtraction` uses at ingestion, but tolerant of company-less
  questions (empty ticker list, never guessed) → `retrieval_service.retrieve(query, k=5, tickers=...)`
  scopes retrieval to those tickers → build a metadata-annotated context string → fetch the
  Langfuse-managed prompt `dev/financial-qa` → `llm_service.ainvoke_structured` against the
  `FinancialAnswer` schema (`answer`/`key_metrics`/`caveats`) → attach `SourceReference`s built directly
  from retrieved-chunk metadata (never LLM-generated, so citations can't be hallucinated — this now
  includes `relevance_score`, populated from the reranker's `metadata["rerank_score"]`, which is
  retrieval-time-deterministic rather than ingestion-time metadata but follows the same trust model)
  → return a `ChatResponse`. If retrieval returns nothing, short-circuits instead of calling the LLM:
  naming the extracted ticker(s) in the message when the question named a company with no ingested
  chunks, otherwise the generic "nothing ingested yet" answer from Level 2. `_build_context`/
  `_to_sources` were promoted out to `rag_formatting.py` in Level 5 (see below) so the agent path
  can reuse them; `process_user_message`'s own behavior is unchanged.
- **`src/services/rag_formatting.py`** (Level 5) — `build_context(chunks)`/`to_sources(chunks)`,
  extracted out of `chat_service.py` so the agent pipeline builds LLM-facing context strings and
  response-facing source citations identically to the Level 2-4 chat pipeline (same
  "citations come only from retrieved-chunk metadata, never the LLM" invariant) instead of a second
  copy of the same logic. Both `chat_service.py` and `agent_graph.py` import from here.
- **`src/services/calculator.py`** (Level 5) — `safe_eval(expression)`, a restricted arithmetic
  evaluator: walks a Python `ast` and only permits numeric literals plus `+ - * / % **`/parentheses;
  any other node (names, calls, attributes, comprehensions, ...) raises `ValueError`. Deliberately
  never calls `eval`/`exec`, since the expression is ultimately LLM-influenced input. Split into its
  own module (mirroring `hybrid_retrieval.py`'s split from `retrieval_service.py`) so it stays
  unit-testable (`tests/services/test_calculator.py`) without importing `agent_tools.py`, which pulls
  in `retrieval_service` and its real embedding/cross-encoder model loads at import time.
- **`src/services/agent_tools.py`** (Level 5) — the three tools available to the Level 5 agent, each
  a plain async function returning `(llm_facing_text, retrieved_documents)`: `search_filings(query,
  tickers=None)` (thin wrapper over `retrieval_service.retrieve`), `compare_companies(tickers,
  aspect)` (a dedicated tool — not left to the agent to improvise by calling `search_filings`
  per-company — fans out scoped retrieval per ticker in parallel via `asyncio.gather` and returns the
  evidence explicitly grouped by company), and `calculate(expression)` (thin wrapper over
  `calculator.safe_eval`). `build_agent_tool_schemas()` builds the `StructuredTool` wrappers bound to
  the model via `.bind_tools(...)` purely so it can produce valid tool-call arguments;
  `TOOL_FUNCTIONS` maps each tool's name back to its plain async function, which `agent_graph.py`'s
  `tools` node dispatches to directly (not through LangChain's generic tool-execution machinery),
  since it needs both the LLM-facing string *and* the raw `Document`s for state/citations.
- **`src/services/agent_graph.py`** (Level 5) — the LangGraph agent itself. `AgentState` (TypedDict)
  tracks the running `messages` list, accumulated `retrieved_chunks`, `draft_answer`,
  `tool_call_log`, and two explicit bounded counters: `loop_count` (agent<->tools rounds, capped at
  `MAX_TOOL_ITERATIONS = 4`) and `reflection_attempts` (grade-triggered retries, capped at
  `MAX_REFLECTION_RETRIES = 1`). These are checked in graph routing logic rather than left to
  LangGraph's `recursion_limit`, since hitting that raises and aborts the whole request — the goal is
  graceful degradation (best-effort answer with an honest caveat), never a 500. Flow: `agent` node
  (binds `tool_schemas` via `.bind_tools(...)`, decides to call tools or stop) <-> `tools` node
  (dispatches via `TOOL_FUNCTIONS`) in a loop, then `draft_answer` (structured `FinancialAnswer` from
  the deduped accumulated evidence, reusing the existing `dev/financial-qa` prompt) -> `grade`
  (structured `AnswerGrade` via the new `dev/answer-grader` prompt; accepts, or routes back to
  `agent` with injected feedback if under the retry cap, or finalizes anyway with an appended caveat
  if the cap is exhausted) -> `finalize`. `build_agent_graph(...)` is a factory (not a module-level
  singleton) taking every model/prompt dependency as an injectable factory function specifically so
  `tests/services/test_agent_graph.py` can swap in a hand-rolled fake chat model and exercise the
  loop/cap/retry control flow with no real LLM calls, no Langfuse prompt fetches, and no
  embedding/cross-encoder model loads (this module never imports `agent_tools.py`/
  `retrieval_service`).
- **`src/services/agent_service.py`** (Level 5) — `run_agent()`, the agent counterpart to
  `chat_service.process_user_message`, wired with `@observe()`. Builds the real (non-test)
  `agent_app` by wiring `build_agent_graph(...)` to `llm_service.get_chat_model` (per-node factories,
  one per Langfuse prompt so each node's model-name resolution reflects its own prompt's `config`)
  and the real tools/prompts, then on each request runs `agent_app.ainvoke(initial_state(question))`
  and maps the final state to an `AgentChatResponse` (sources via `rag_formatting.to_sources` on the
  deduped accumulated chunks — never LLM-generated, same as the chat path).
- **`src/services/chromadb_service.py`** — `ChromaDbService`, a `Singleton` wrapper around a
  `chromadb.PersistentClient` (path from `settings.CHROMA_PERSIST_PATH`) for collection
  create/get/delete. Note: `document_processor/parse_documents.py` currently talks to Chroma directly
  via `langchain_chroma.Chroma` instead of going through this service.
- **`src/core/sec_filings_downloader.py`** — `SecFilingsDownloader` wraps `sec_edgar_downloader`.
  Downloads to a temp dir (library layout: `sec-edgar-filings/<TICKER>/<FORM_TYPE>/<ACCESSION>/<file>`),
  then flattens and renames every file into `sec_files/` as
  `{TICKER}_{FORM_TYPE}_{ACCESSION}_{filename}`, deleting the temp dir afterward.
- **`src/document_processor/parse_documents.py`** — two pieces:
  - `SecEdgarAdvancedLoader` (a LangChain `BaseLoader`): parses SEC EDGAR `full-submission.txt` files by
    regexing out `<DOCUMENT><TYPE>...<TEXT>...</TEXT>` blocks matching `target_types` (e.g. `10-K`),
    pulls header metadata (CIK, company name, filing/fiscal dates) from the `<SEC-HEADER>` block, and
    cleans each document's HTML: tables with a high digit-density or financial-keyword headers are kept
    and converted to Markdown wrapped in `[START_TABLE]`/`[END_TABLE]` markers (so the chunker can
    respect table boundaries); other tables are treated as layout and unwrapped/flattened. Also accepts
    an optional `ticker` constructor arg, attached to every document's metadata as `ticker` when present.
  - `DocumentProcessor`: owns a `HuggingFaceEmbeddings` model
    (`sentence-transformers/all-mpnet-base-v2`, CPU, normalized) and a `langchain_chroma.Chroma` store
    bound to a given collection name. `extract_data(file_path, filings=["10-K"], ticker=None)` loads a
    filing via `SecEdgarAdvancedLoader` (passing `ticker` through), splits it with
    `RecursiveCharacterTextSplitter` (separators include the `[START_TABLE]`/`[END_TABLE]` markers,
    `chunk_size=4000`, `chunk_overlap=200`), and adds the chunks to the vector store.
- **`src/models/`** — Pydantic request/schema models. `chat_models.UserMessage` (accepts
  `userMessage` alias) is the base request shape for both `/chat` and `/ingest`.
  `chat_models.FinancialAnswer` is the structured-output schema the LLM fills in for chat answers;
  `chat_models.SourceReference`/`ChatResponse` wrap that plus retrieved-chunk citations for the API
  response (`SourceReference` includes `relevance_score`, the reranker's score for that chunk).
  `chat_models.QueryEntities` is the structured-output schema the LLM fills in at query time
  (`dev/query-entity-extraction`) to identify which company/companies (if any) a chat question is
  about, used to scope retrieval. `ingestion_models.SymbolExtraction` is the structured-output schema
  the LLM fills in during ingestion (ticker symbol(s), `from_date`, `to_date`). `agent_models.py`
  (Level 5): `AnswerGrade` is the structured-output schema the `grade` node fills in
  (`grounded`/`complete`/`feedback`); `AgentChatResponse` extends `ChatResponse` with `tool_calls`
  (human-readable log of tool invocations, for transparency into the agent's reasoning) and
  `reflection_retried`.
- **`src/constants.py`** — small shared constants, currently just `DEFAULT_COLLECTION_NAME`, the
  Chroma collection name used by both ingestion and retrieval so they can't drift apart.
- **`src/utils/singleton.py`** — `Singleton` metaclass used by `LLMService` and `ChromaDbService` to
  guarantee a single shared instance per process.
- **`tests/`** — mirrors the `src/` package layout (e.g. `tests/services/` for `src/services/`).
  Unit tests target pure, dependency-light logic only (see `hybrid_retrieval.py` above) — avoid
  importing modules that construct real embedding/LLM/cross-encoder models at import time.

### Tracing / observability

Langfuse is wired in throughout via `@observe()` decorators on service entrypoints (`ingest_data`,
`process_user_message`, `run_agent`) and via `llm_service`'s automatic callback attachment — new
service-level entrypoints that do meaningful work should follow the same `@observe()` convention, and
any LLM calls should go through `llm_service` rather than instantiating a chat model directly, to keep
tracing consistent. Prompts (e.g. `dev/symbol-extractor`, `dev/query-entity-extraction`,
`dev/financial-qa`, `dev/agent-system`, `dev/answer-grader`) are managed in Langfuse and fetched via
`langfuse.get_prompt(...)`, not hardcoded in Python.
