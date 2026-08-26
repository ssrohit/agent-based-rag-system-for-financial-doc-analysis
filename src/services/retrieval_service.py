import asyncio
import logging
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.config import settings
from src.constants import DEFAULT_COLLECTION_NAME
from src.services.hybrid_retrieval import (
    bm25_top_n,
    build_bm25_index,
    build_ticker_filter,
    filter_documents_by_tickers,
    reciprocal_rank_fusion,
    select_top_k_by_score,
)
from src.utils.singleton import Singleton
from src.utils.timing import log_duration

logger = logging.getLogger(__name__)

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTOR_CANDIDATE_K = 20
BM25_CANDIDATE_K = 20
RERANK_CANDIDATE_POOL = 15


class RetrievalService(metaclass=Singleton):
    """
    Read-oriented access to the Chroma vector store populated by ingestion.

    Hybrid retrieval pipeline: vector similarity search and BM25 keyword search each
    produce a candidate pool, fused via Reciprocal Rank Fusion, then reordered by a local
    cross-encoder reranker before the top-k is returned. BM25 has no incremental-update API,
    so its in-memory index is rebuilt wholesale by `refresh()` rather than updated
    per-document; ingestion is responsible for calling `refresh()` once new chunks are added.
    """

    def __init__(self) -> None:
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store = Chroma(
            persist_directory=settings.CHROMA_PERSIST_PATH,
            embedding_function=self.embedding_function,
            collection_name=DEFAULT_COLLECTION_NAME,
        )
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device="cpu")
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_documents: List[Document] = []
        self.refresh()

    def refresh(self) -> None:
        """
        Rebuild the in-memory BM25 index from whatever is currently persisted in Chroma.

        Note: this reads via a separate Chroma/PersistentClient instance than the one
        DocumentProcessor writes through (same persist path/collection name) - both work off
        Chroma's durable on-disk writes, so this sees data written by any process, but keep in
        mind having multiple concurrently-open PersistentClients on the same path is a known
        Chroma footgun (SQLite locking), pre-existing in this codebase and not addressed here.

        Args:
            None.

        Returns:
            None. Updates `self._bm25_documents` and `self._bm25_index` in place; the latter is
            set to `None` when the collection is currently empty (see `build_bm25_index`).
        """
        try:
            with log_duration(logger, "BM25 index refresh"):
                data = self.vector_store.get(include=["documents", "metadatas"])
                documents = [
                    Document(page_content=text, metadata=metadata or {}, id=doc_id)
                    for doc_id, text, metadata in zip(
                        data["ids"], data["documents"], data["metadatas"]
                    )
                ]
                self._bm25_documents = documents
                self._bm25_index = build_bm25_index(documents)
            logger.info("BM25 index refreshed with %d chunks", len(documents))
        except Exception as error:
            logger.error("Error while refreshing BM25 index: %s", str(error), exc_info=True)
            raise error

    def _bm25_candidates(self, query: str, tickers: Optional[List[str]]) -> List[Document]:
        """
        Run BM25 candidate retrieval, optionally scoped to specific ticker(s).

        BM25Okapi has no query-time filtering of its own - its index is fixed to a corpus at
        construction time - so a ticker-scoped query rebuilds a small ephemeral index from just
        the matching subset of `self._bm25_documents` rather than filtering the whole-collection
        index built by `refresh()`.

        Args:
            query: Raw query text.
            tickers: Ticker symbols to scope the search to, or `None`/empty for no scoping
                (uses the pre-built whole-collection index from `refresh()`).

        Returns:
            Up to `BM25_CANDIDATE_K` documents ranked by BM25 score, highest first. Empty list
            if there are no documents in scope (empty collection, or no chunks match `tickers`).
        """
        if not tickers:
            return bm25_top_n(
                self._bm25_index, self._bm25_documents, query, BM25_CANDIDATE_K
            )
        scoped_documents = filter_documents_by_tickers(self._bm25_documents, tickers)
        scoped_index = build_bm25_index(scoped_documents)
        return bm25_top_n(scoped_index, scoped_documents, query, BM25_CANDIDATE_K)

    async def retrieve(
        self, query: str, k: int = 5, tickers: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Retrieve the top-k chunks for a query via hybrid (vector + BM25) search, fused by
        Reciprocal Rank Fusion and reordered by a local cross-encoder reranker.

        Args:
            query: The user's natural-language question or search text.
            k: Maximum number of chunks to return, after reranking.
            tickers: When given (non-empty), scopes both the vector search (via a Chroma
                metadata `where` filter, see `build_ticker_filter`) and the BM25 candidate pool
                (via `filter_documents_by_tickers`) to chunks whose `metadata["ticker"]` matches
                one of these tickers (case-insensitive). `None` or empty means search the whole
                collection, unscoped - unchanged from the pre-Level-4 behavior.

        Returns:
            Up to `k` `Document`s, best-first, each with a `metadata["rerank_score"]` set by the
            cross-encoder (see `select_top_k_by_score`). Empty list if nothing is retrievable -
            either the collection (or the `tickers`-scoped subset of it) is empty, or fusion
            produced no candidates.
        """
        try:
            with log_duration(logger, f"Hybrid retrieval total (query={query!r})"):
                with log_duration(logger, "Vector similarity search"):
                    vector_docs = await self.vector_store.asimilarity_search(
                        query, k=VECTOR_CANDIDATE_K, filter=build_ticker_filter(tickers)
                    )
                with log_duration(logger, "BM25 candidate search"):
                    bm25_docs = await asyncio.to_thread(self._bm25_candidates, query, tickers)

                fused = reciprocal_rank_fusion([vector_docs, bm25_docs])[:RERANK_CANDIDATE_POOL]
                if not fused:
                    return []

                with log_duration(logger, f"Cross-encoder rerank ({len(fused)} candidates)"):
                    scores = await asyncio.to_thread(
                        self.cross_encoder.predict, [(query, doc.page_content) for doc in fused]
                    )
                return select_top_k_by_score(fused, scores, k)
        except Exception as error:
            logger.error(
                "Error while retrieving chunks for query %r: %s",
                query,
                str(error),
                exc_info=True,
            )
            raise error


# Expose a singleton instance for standard imports
retrieval_service = RetrievalService()
