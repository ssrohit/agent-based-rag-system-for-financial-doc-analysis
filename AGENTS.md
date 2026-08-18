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
- There is no test suite in this repo yet.

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
- **`src/routes/`** — thin FastAPI routers (`chat.py` at `/chat`, `ingest.py` at `/ingest`) that just
  delegate to a service function. Keep route handlers thin; business logic belongs in services.
- **`src/services/llm_service.py`** — `llm_service`, a `Singleton`-metaclass instance centralizing all
  LLM calls (via `langchain_google_genai.ChatGoogleGenerativeAI`). All LLM invocation should go through
  this, not ad-hoc `ChatGoogleGenerativeAI` instances, because it:
  - auto-attaches a Langfuse `CallbackHandler` to every call for tracing (`_get_callbacks`);
  - resolves the model name with precedence: explicit `model_name` arg → model embedded in a Langfuse
    prompt object's `config`/`metadata`/`model` attr → `settings.CHAT_MODEL` default, stripping any
    `provider:` prefix (e.g. `google_genai:gemini-2.5-flash` → `gemini-2.5-flash`);
  - exposes `ainvoke_structured(prompt, schema, ...)` for Pydantic-schema-constrained output and
    `ainvoke(prompt, ...)` for raw string output.
- **`src/services/ingestion_service.py`** — `ingest_data()`, the pipeline entrypoint. Flow: fetch a
  Langfuse-managed prompt (`dev/symbol-extractor`) → `llm_service.ainvoke_structured` to extract a
  `SymbolExtraction` (ticker(s) + date range) from the user's message → download filings via
  `SecFilingsDownloader` → feed each downloaded file through `DocumentProcessor.extract_data` into the
  Chroma collection named by `src.constants.DEFAULT_COLLECTION_NAME`.
- **`src/services/retrieval_service.py`** — `retrieval_service`, a `Singleton` read-only wrapper around
  the same Chroma collection (built once, not per-request, unlike `DocumentProcessor`). Exposes
  `async retrieve(query, k)` via `Chroma.asimilarity_search`. Kept separate from `chat_service.py` so a
  future hybrid-retrieval/reranking implementation can swap in without touching the generation code.
- **`src/services/chat_service.py`** — `process_user_message()`, the query-side entrypoint, wired with
  `@observe()`. Flow: `retrieval_service.retrieve()` → build a metadata-annotated context string →
  fetch the Langfuse-managed prompt `dev/financial-qa` → `llm_service.ainvoke_structured` against the
  `FinancialAnswer` schema (`answer`/`key_metrics`/`caveats`) → attach `SourceReference`s built directly
  from retrieved-chunk metadata (never LLM-generated, so citations can't be hallucinated) → return a
  `ChatResponse`. If retrieval returns nothing, short-circuits with an explicit "nothing ingested yet"
  answer instead of calling the LLM.
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
    respect table boundaries); other tables are treated as layout and unwrapped/flattened.
  - `DocumentProcessor`: owns a `HuggingFaceEmbeddings` model
    (`sentence-transformers/all-mpnet-base-v2`, CPU, normalized) and a `langchain_chroma.Chroma` store
    bound to a given collection name. `extract_data(file_path)` loads a filing via
    `SecEdgarAdvancedLoader`, splits it with `RecursiveCharacterTextSplitter` (separators include the
    `[START_TABLE]`/`[END_TABLE]` markers, `chunk_size=4000`, `chunk_overlap=200`), and adds the chunks
    to the vector store.
- **`src/models/`** — Pydantic request/schema models. `chat_models.UserMessage` (accepts
  `userMessage` alias) is the base request shape for both `/chat` and `/ingest`.
  `chat_models.FinancialAnswer` is the structured-output schema the LLM fills in for chat answers;
  `chat_models.SourceReference`/`ChatResponse` wrap that plus retrieved-chunk citations for the API
  response. `ingestion_models.SymbolExtraction` is the structured-output schema the LLM fills in
  during ingestion (ticker symbol(s), `from_date`, `to_date`).
- **`src/constants.py`** — small shared constants, currently just `DEFAULT_COLLECTION_NAME`, the
  Chroma collection name used by both ingestion and retrieval so they can't drift apart.
- **`src/utils/singleton.py`** — `Singleton` metaclass used by `LLMService` and `ChromaDbService` to
  guarantee a single shared instance per process.

### Tracing / observability

Langfuse is wired in throughout via `@observe()` decorators on service entrypoints (`ingest_data`,
`process_user_message`) and via `llm_service`'s automatic callback attachment — new service-level
entrypoints that do meaningful work should follow the same `@observe()` convention, and any LLM calls
should go through `llm_service` rather than instantiating a chat model directly, to keep tracing
consistent. Prompts (e.g. `dev/symbol-extractor`) are managed in Langfuse and fetched via
`langfuse.get_prompt(...)`, not hardcoded in Python.
