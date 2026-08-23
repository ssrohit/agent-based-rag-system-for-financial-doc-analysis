"""
One-off backfill: attach `ticker` metadata to Chroma chunks ingested before Level 3 added it.

Level 3 started attaching `ticker` to every chunk at ingestion time, but chunks ingested before
that (Level 1/2) predate the field entirely. Once Level 4 started filtering retrieval by
`ticker`, those older chunks became invisible to company-scoped questions even though they were
genuinely ingested. This script derives each such chunk's ticker from its existing `source`
metadata (the downloaded filing's path, named `{TICKER}_{FORM_TYPE}_{ACCESSION}_{filename}` by
`SecFilingsDownloader`) using the same parsing rule `ingestion_service._ticker_from_downloaded_file`
applies at ingestion time, and writes it back onto each chunk's Chroma metadata.

Run once, manually: `uv run scripts/backfill_ticker_metadata.py`. Safe to re-run (only touches
chunks that still lack a `ticker`). After running, restart the API server so
`RetrievalService.refresh()` picks up the updated metadata into its in-memory BM25 index (Chroma
itself is updated immediately; only the in-process BM25 index needs a refresh).
"""
import logging
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.constants import DEFAULT_COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ticker_from_source(source: str) -> Optional[str]:
    """
    Recover a chunk's ticker from its `source` metadata (the downloaded filing's file path),
    mirroring `ingestion_service._ticker_from_downloaded_file`'s filename convention.

    Args:
        source: The chunk's `metadata["source"]` value, e.g.
            `sec_files\\AAPL_10-K_0000320193-23-000106_full-submission.txt`.

    Returns:
        The uppercased ticker prefix of the filename, or `None` if `source` is empty/unparseable.
    """
    if not source:
        return None
    parts = Path(source).name.split("_", 1)
    return parts[0].upper() if parts and parts[0] else None


def main() -> None:
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma(
        persist_directory=settings.CHROMA_PERSIST_PATH,
        embedding_function=embedding_function,
        collection_name=DEFAULT_COLLECTION_NAME,
    )
    data = vector_store.get(include=["metadatas"])

    ids_to_update = []
    metadatas_to_update = []
    skipped_no_source = 0
    for doc_id, metadata in zip(data["ids"], data["metadatas"]):
        metadata = metadata or {}
        if metadata.get("ticker"):
            continue
        ticker = _ticker_from_source(metadata.get("source", ""))
        if not ticker:
            skipped_no_source += 1
            continue
        ids_to_update.append(doc_id)
        # Pass the full existing metadata dict (not just {"ticker": ...}) so no other
        # ingestion-time metadata (company_name, doc_type, filing_date, cik, source) is lost,
        # regardless of whether Chroma's update() merges or replaces metadata per-id.
        metadatas_to_update.append({**metadata, "ticker": ticker})

    if ids_to_update:
        vector_store._collection.update(ids=ids_to_update, metadatas=metadatas_to_update)

    logger.info("Backfilled ticker metadata on %d chunk(s).", len(ids_to_update))
    if skipped_no_source:
        logger.warning(
            "Skipped %d chunk(s) with no usable 'source' metadata to derive a ticker from.",
            skipped_no_source,
        )


if __name__ == "__main__":
    main()
